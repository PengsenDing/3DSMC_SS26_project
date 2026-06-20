#include "face_reconstruction/viewer.hpp"

#define GL_SILENCE_DEPRECATION

#include <GL/glew.h>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <SFML/Window/Window.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace face_reconstruction {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kNearPlane = 0.01f;
constexpr float kFarPlane = 100.0f;

constexpr std::string_view kVertexShader = R"(
#version 120

varying vec3 v_albedo;
varying vec3 v_normal;
varying vec3 v_model_position;
varying float v_view_depth;

void main() {
    vec4 view_position = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = gl_ProjectionMatrix * view_position;
    v_albedo = gl_Color.rgb;
    v_normal = normalize(gl_NormalMatrix * gl_Normal);
    v_model_position = gl_MultiTexCoord0.xyz;
    v_view_depth = -view_position.z;
}
)";

constexpr std::string_view kFragmentShader = R"(
#version 120

uniform int u_mode;
uniform float u_depth_near;
uniform float u_depth_far;
uniform float u_checker_frequency;

varying vec3 v_albedo;
varying vec3 v_normal;
varying vec3 v_model_position;
varying float v_view_depth;

void main() {
    if (u_mode == 0) {
        gl_FragColor = vec4(v_albedo, 1.0);
    } else if (u_mode == 1) {
        float range = max(u_depth_far - u_depth_near, 0.0001);
        float depth = clamp((v_view_depth - u_depth_near) / range, 0.0, 1.0);
        gl_FragColor = vec4(vec3(1.0 - depth), 1.0);
    } else if (u_mode == 2) {
        vec3 encoded_normal = 0.5 * normalize(v_normal) + vec3(0.5);
        gl_FragColor = vec4(encoded_normal, 1.0);
    } else {
        vec2 cell = floor(v_model_position.xy * u_checker_frequency);
        float parity = mod(cell.x + cell.y, 2.0);
        vec3 dark = vec3(0.08, 0.10, 0.13);
        vec3 light = vec3(0.92, 0.94, 0.96);
        gl_FragColor = vec4(mix(light, dark, parity), 1.0);
    }
}
)";

struct Camera {
    float yaw_degrees = 0.0f;
    float pitch_degrees = -10.0f;
    float distance = 3.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
};

struct MouseState {
    bool rotating = false;
    bool panning = false;
    sf::Vector2i last_position{};
};

struct ViewerState {
    RenderMode mode = RenderMode::Albedo;
    bool save_requested = false;
};

GLuint compile_shader(GLenum type, std::string_view source) {
    const GLuint shader = glCreateShader(type);
    const char* source_data = source.data();
    const GLint source_length = static_cast<GLint>(source.size());
    glShaderSource(shader, 1, &source_data, &source_length);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) {
        return shader;
    }

    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(std::max(log_length, 1)), '\0');
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error("Could not compile rendering shader:\n" + log);
}

GLuint create_render_program() {
    const GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return program;
    }

    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(std::max(log_length, 1)), '\0');
    glGetProgramInfoLog(program, log_length, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error("Could not link rendering shader:\n" + log);
}

void set_perspective(float fov_y_degrees, float aspect, float near_plane, float far_plane) {
    const float fov_y_radians = fov_y_degrees * kPi / 180.0f;
    const float top = std::tan(fov_y_radians * 0.5f) * near_plane;
    const float right = top * aspect;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-right, right, -top, top, near_plane, far_plane);
}

Eigen::Vector3f face_normal(const Mesh& mesh, const Triangle& triangle) {
    const Eigen::Vector3f& a = mesh.vertices[triangle.vertex_indices[0]];
    const Eigen::Vector3f& b = mesh.vertices[triangle.vertex_indices[1]];
    const Eigen::Vector3f& c = mesh.vertices[triangle.vertex_indices[2]];
    const Eigen::Vector3f normal = (b - a).cross(c - a);
    const float length = normal.norm();
    return length > 1.0e-8f ? normal / length : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
}

void draw_mesh(const Mesh& mesh) {
    const Eigen::Vector3f mesh_center = mesh.center();
    const float radius = mesh.bounding_radius();
    const Eigen::Vector3f default_albedo(0.72f, 0.55f, 0.46f);

    glBegin(GL_TRIANGLES);
    for (const Triangle& triangle : mesh.triangles) {
        const Eigen::Vector3f fallback_normal = face_normal(mesh, triangle);

        for (const std::uint32_t vertex_index : triangle.vertex_indices) {
            const Eigen::Vector3f& vertex = mesh.vertices[vertex_index];
            const Eigen::Vector3f& normal =
                mesh.has_normals() ? mesh.normals[vertex_index] : fallback_normal;
            const Eigen::Vector3f& albedo =
                mesh.has_colors() ? mesh.colors[vertex_index] : default_albedo;
            const Eigen::Vector3f normalized_position = (vertex - mesh_center) / radius;

            glColor3fv(albedo.data());
            glNormal3fv(normal.data());
            glMultiTexCoord3fv(GL_TEXTURE0, normalized_position.data());
            glVertex3fv(vertex.data());
        }
    }
    glEnd();
}

void reset_camera(Camera& camera) {
    camera = Camera{};
}

void select_mode(ViewerState& state, RenderMode mode) {
    if (state.mode == mode) {
        return;
    }
    state.mode = mode;
    std::cout << "Render mode: " << render_mode_name(mode) << '\n';
}

void handle_event(const sf::Event& event,
                  sf::Window& window,
                  Camera& camera,
                  MouseState& mouse,
                  ViewerState& state) {
    if (event.is<sf::Event::Closed>()) {
        window.close();
        return;
    }

    if (const auto* key_pressed = event.getIf<sf::Event::KeyPressed>()) {
        switch (key_pressed->code) {
            case sf::Keyboard::Key::Escape:
                window.close();
                break;
            case sf::Keyboard::Key::R:
                reset_camera(camera);
                break;
            case sf::Keyboard::Key::S:
                state.save_requested = true;
                break;
            case sf::Keyboard::Key::Num1:
                select_mode(state, RenderMode::Albedo);
                break;
            case sf::Keyboard::Key::Num2:
                select_mode(state, RenderMode::Depth);
                break;
            case sf::Keyboard::Key::Num3:
                select_mode(state, RenderMode::Normal);
                break;
            case sf::Keyboard::Key::Num4:
                select_mode(state, RenderMode::Checkerboard);
                break;
            default:
                break;
        }
    }

    if (const auto* mouse_pressed = event.getIf<sf::Event::MouseButtonPressed>()) {
        mouse.last_position = mouse_pressed->position;
        mouse.rotating = mouse_pressed->button == sf::Mouse::Button::Left;
        mouse.panning = mouse_pressed->button == sf::Mouse::Button::Right ||
                        mouse_pressed->button == sf::Mouse::Button::Middle;
    }

    if (const auto* mouse_released = event.getIf<sf::Event::MouseButtonReleased>()) {
        if (mouse_released->button == sf::Mouse::Button::Left) {
            mouse.rotating = false;
        }
        if (mouse_released->button == sf::Mouse::Button::Right ||
            mouse_released->button == sf::Mouse::Button::Middle) {
            mouse.panning = false;
        }
    }

    if (const auto* mouse_moved = event.getIf<sf::Event::MouseMoved>()) {
        const sf::Vector2i delta = mouse_moved->position - mouse.last_position;
        mouse.last_position = mouse_moved->position;

        if (mouse.rotating) {
            camera.yaw_degrees += static_cast<float>(delta.x) * 0.35f;
            camera.pitch_degrees =
                std::clamp(camera.pitch_degrees + static_cast<float>(delta.y) * 0.35f,
                           -85.0f,
                           85.0f);
        }

        if (mouse.panning) {
            camera.pan_x += static_cast<float>(delta.x) * 0.0025f * camera.distance;
            camera.pan_y -= static_cast<float>(delta.y) * 0.0025f * camera.distance;
        }
    }

    if (const auto* wheel = event.getIf<sf::Event::MouseWheelScrolled>()) {
        camera.distance =
            std::clamp(camera.distance * (1.0f - wheel->delta * 0.08f), 1.5f, 20.0f);
    }
}

void render_frame(const Mesh& mesh,
                  const Camera& camera,
                  sf::Vector2u window_size,
                  RenderMode mode,
                  GLuint program) {
    const int width = static_cast<int>(std::max(window_size.x, 1u));
    const int height = static_cast<int>(std::max(window_size.y, 1u));
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    if (mode == RenderMode::Depth || mode == RenderMode::Normal) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    } else {
        glClearColor(0.94f, 0.95f, 0.96f, 1.0f);
    }

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    set_perspective(45.0f, aspect, kNearPlane, kFarPlane);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(camera.pan_x, camera.pan_y, -camera.distance);
    glRotatef(camera.pitch_degrees, 1.0f, 0.0f, 0.0f);
    glRotatef(camera.yaw_degrees, 0.0f, 1.0f, 0.0f);

    const float mesh_scale = 1.25f / mesh.bounding_radius();
    const Eigen::Vector3f mesh_center = mesh.center();
    glScalef(mesh_scale, mesh_scale, mesh_scale);
    glTranslatef(-mesh_center.x(), -mesh_center.y(), -mesh_center.z());

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "u_mode"), static_cast<int>(mode));
    glUniform1f(glGetUniformLocation(program, "u_depth_near"),
                std::max(kNearPlane, camera.distance - 1.35f));
    glUniform1f(glGetUniformLocation(program, "u_depth_far"), camera.distance + 1.35f);
    glUniform1f(glGetUniformLocation(program, "u_checker_frequency"), 7.0f);
    draw_mesh(mesh);
    glUseProgram(0);
}

void save_framebuffer(const std::filesystem::path& path, sf::Vector2u window_size) {
    const int width = static_cast<int>(std::max(window_size.x, 1u));
    const int height = static_cast<int>(std::max(window_size.y, 1u));
    cv::Mat rgb_image(height, width, CV_8UC3);

    glFinish();
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb_image.data);

    cv::Mat top_down_rgb;
    cv::flip(rgb_image, top_down_rgb, 0);
    cv::Mat bgr_image;
    cv::cvtColor(top_down_rgb, bgr_image, cv::COLOR_RGB2BGR);

    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (!cv::imwrite(path.string(), bgr_image)) {
        throw std::runtime_error("Could not save rendered image: " + path.string());
    }

    std::cout << "Saved " << path.string() << '\n';
}

std::filesystem::path default_screenshot_path(RenderMode mode) {
    return std::filesystem::path("outputs") / (render_mode_name(mode) + ".png");
}

}  // namespace

std::string render_mode_name(RenderMode mode) {
    switch (mode) {
        case RenderMode::Albedo:
            return "albedo";
        case RenderMode::Depth:
            return "depth";
        case RenderMode::Normal:
            return "normal";
        case RenderMode::Checkerboard:
            return "checkerboard";
    }
    throw std::runtime_error("Unknown render mode");
}

RenderMode parse_render_mode(std::string_view name) {
    if (name == "albedo") {
        return RenderMode::Albedo;
    }
    if (name == "depth") {
        return RenderMode::Depth;
    }
    if (name == "normal") {
        return RenderMode::Normal;
    }
    if (name == "checkerboard" || name == "checker") {
        return RenderMode::Checkerboard;
    }
    throw std::runtime_error("Unknown render mode '" + std::string(name) +
                             "' (expected albedo, depth, normal, or checkerboard)");
}

int run_viewer(const Mesh& mesh, const ViewerOptions& options) {
    if (mesh.empty()) {
        throw std::runtime_error("Cannot view an empty mesh");
    }

    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antiAliasingLevel = 4;
    settings.majorVersion = 2;
    settings.minorVersion = 1;

    sf::Window window(sf::VideoMode({1280u, 900u}),
                      "Face Reconstruction - Basic Renderer",
                      sf::Style::Default,
                      sf::State::Windowed,
                      settings);
    window.setVerticalSyncEnabled(true);
    if (!window.setActive(true)) {
        throw std::runtime_error("Could not activate the SFML OpenGL context");
    }

    glewExperimental = GL_TRUE;
    const GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        throw std::runtime_error("Could not initialize GLEW");
    }

    const GLuint program = create_render_program();
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);

    Camera camera;
    MouseState mouse;
    ViewerState state{options.initial_mode, false};

    std::cout << "Viewer controls: 1 albedo, 2 depth, 3 normal, 4 checkerboard, "
                 "S save, left drag rotate, right/middle drag pan, wheel zoom, "
                 "R reset, Esc close\n";
    std::cout << "Render mode: " << render_mode_name(state.mode) << '\n';

    if (options.render_all_directory.has_value()) {
        const std::array modes = {
            RenderMode::Albedo,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Checkerboard,
        };
        for (const RenderMode mode : modes) {
            render_frame(mesh, camera, window.getSize(), mode, program);
            save_framebuffer(*options.render_all_directory / (render_mode_name(mode) + ".png"),
                             window.getSize());
        }
        glDeleteProgram(program);
        return 0;
    }

    int rendered_frames = 0;
    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            handle_event(*event, window, camera, mouse, state);
        }
        if (!window.isOpen()) {
            break;
        }

        render_frame(mesh, camera, window.getSize(), state.mode, program);

        if (options.screenshot_path.has_value() && rendered_frames == 0) {
            save_framebuffer(*options.screenshot_path, window.getSize());
            window.close();
        } else if (state.save_requested) {
            save_framebuffer(default_screenshot_path(state.mode), window.getSize());
            state.save_requested = false;
        }

        if (window.isOpen()) {
            window.display();
        }
        ++rendered_frames;
        if (options.max_frames > 0 && rendered_frames >= options.max_frames) {
            window.close();
        }
    }

    return 0;
}

}  // namespace face_reconstruction
