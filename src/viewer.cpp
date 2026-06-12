#include "face_reconstruction/viewer.hpp"

#define GL_SILENCE_DEPRECATION

#include <GL/glew.h>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <SFML/Window/Window.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace face_reconstruction {
namespace {

constexpr float kPi = 3.14159265358979323846f;

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
    Eigen::Vector3f normal = (b - a).cross(c - a);

    const float norm = normal.norm();
    if (norm <= 1.0e-8f) {
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }

    return normal / norm;
}

void draw_mesh(const Mesh& mesh) {
    glColor3f(0.72f, 0.76f, 0.80f);
    glBegin(GL_TRIANGLES);
    for (const Triangle& triangle : mesh.triangles) {
        const Eigen::Vector3f normal = face_normal(mesh, triangle);
        glNormal3f(normal.x(), normal.y(), normal.z());

        for (const std::uint32_t vertex_index : triangle.vertex_indices) {
            const Eigen::Vector3f& vertex = mesh.vertices[vertex_index];
            glVertex3f(vertex.x(), vertex.y(), vertex.z());
        }
    }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f(0.08f, 0.10f, 0.12f);
    glBegin(GL_TRIANGLES);
    for (const Triangle& triangle : mesh.triangles) {
        for (const std::uint32_t vertex_index : triangle.vertex_indices) {
            const Eigen::Vector3f& vertex = mesh.vertices[vertex_index];
            glVertex3f(vertex.x(), vertex.y(), vertex.z());
        }
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void reset_camera(Camera& camera) {
    camera = Camera{};
}

void handle_event(const sf::Event& event, sf::Window& window, Camera& camera, MouseState& mouse) {
    if (event.is<sf::Event::Closed>()) {
        window.close();
        return;
    }

    if (const auto* key_pressed = event.getIf<sf::Event::KeyPressed>()) {
        if (key_pressed->code == sf::Keyboard::Key::Escape) {
            window.close();
        }
        if (key_pressed->code == sf::Keyboard::Key::R) {
            reset_camera(camera);
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
            std::clamp(camera.distance * (1.0f - wheel->delta * 0.08f), 0.8f, 20.0f);
    }
}

void render_frame(const Mesh& mesh, const Camera& camera, const sf::Vector2u window_size) {
    const int width = static_cast<int>(std::max(window_size.x, 1u));
    const int height = static_cast<int>(std::max(window_size.y, 1u));
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    set_perspective(45.0f, aspect, 0.01f, 100.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(camera.pan_x, camera.pan_y, -camera.distance);
    glRotatef(camera.pitch_degrees, 1.0f, 0.0f, 0.0f);
    glRotatef(camera.yaw_degrees, 0.0f, 1.0f, 0.0f);

    const float mesh_scale = 1.25f / mesh.bounding_radius();
    const Eigen::Vector3f mesh_center = mesh.center();
    glScalef(mesh_scale, mesh_scale, mesh_scale);
    glTranslatef(-mesh_center.x(), -mesh_center.y(), -mesh_center.z());

    draw_mesh(mesh);
}

}  // namespace

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
                      "Face Reconstruction - Unlit Mesh Viewer",
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

    glClearColor(0.94f, 0.95f, 0.96f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    Camera camera;
    MouseState mouse;

    std::cout << "Viewer controls: left drag rotate, right/middle drag pan, wheel zoom, R reset, Esc close\n";

    int rendered_frames = 0;

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            handle_event(*event, window, camera, mouse);
        }

        render_frame(mesh, camera, window.getSize());
        window.display();

        ++rendered_frames;
        if (options.max_frames > 0 && rendered_frames >= options.max_frames) {
            window.close();
        }
    }

    return 0;
}

}  // namespace face_reconstruction
