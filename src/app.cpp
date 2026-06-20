#include "face_reconstruction/app.hpp"

#define GL_SILENCE_DEPRECATION

#include "face_reconstruction/landmarks.hpp"
#include "face_reconstruction/mesh.hpp"
#include "face_reconstruction/obj_loader.hpp"
#include "face_reconstruction/ply_loader.hpp"
#include "face_reconstruction/viewer.hpp"

#include <GL/glew.h>
#include <SFML/Config.hpp>
#include <ceres/version.h>
#include <opencv2/core/version.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace face_reconstruction {
namespace {

void print_usage(const char* executable_name) {
    std::cout << "Usage: " << executable_name
              << " [mesh.obj|mesh.ply] [--info] [--frames N] [--mode MODE]"
                 " [--output image.png] [--render-all DIR] [--landmarks landmarks.csv]"
                 " [--correspondences correspondences.csv]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  mesh path  OBJ or ASCII PLY mesh. Defaults to data/model.obj.\n";
    std::cout << "  --info    Load the mesh and print its summary without opening a window.\n";
    std::cout << "  --frames  Render N frames and exit. Useful for viewer smoke tests.\n";
    std::cout << "  --mode    Initial mode: albedo, depth, normal, or checkerboard.\n";
    std::cout << "  --output  Save the initial mode to a PNG and exit.\n";
    std::cout << "  --render-all  Save all four modes as PNG files in a directory and exit.\n";
    std::cout << "  --landmarks  Load a MediaPipe landmark CSV and print its summary.\n";
    std::cout << "  --correspondences  Load a BFM/MediaPipe correspondence CSV summary.\n";
    std::cout << "  --deps    Print the linked dependency report.\n";
    std::cout << "  --help    Show this help text.\n";
}

Mesh load_mesh(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (extension == ".obj") {
        return load_obj_mesh(path);
    }
    if (extension == ".ply") {
        return load_ply_mesh(path);
    }
    throw std::runtime_error("Unsupported mesh format '" + extension +
                             "' (expected .obj or .ply)");
}

}  // namespace

std::string dependency_report() {
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    std::ostringstream report;
    report << "Face Reconstruction project skeleton\n";
    report << "C++ target: C++20\n";
    report << "Eigen: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
           << EIGEN_MINOR_VERSION << " (identity trace = " << identity.trace() << ")\n";
    report << "OpenCV: " << CV_VERSION << "\n";
    report << "Ceres Solver: " << CERES_VERSION_STRING << "\n";
    report << "GLEW headers: " << GLEW_VERSION_MAJOR << "." << GLEW_VERSION_MINOR << "."
           << GLEW_VERSION_MICRO << "\n";
    report << "SFML: " << SFML_VERSION_MAJOR << "." << SFML_VERSION_MINOR << "."
           << SFML_VERSION_PATCH << "\n";
    report << "OpenGL: linked through CMake OpenGL::GL\n";
    return report.str();
}

int run(int argc, char** argv) {
    std::filesystem::path mesh_path = "data/model.obj";
    std::optional<std::filesystem::path> landmarks_path;
    std::optional<std::filesystem::path> correspondences_path;
    bool info_only = false;
    ViewerOptions viewer_options;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];

        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        }

        if (argument == "--deps") {
            std::cout << dependency_report();
            return 0;
        }

        if (argument == "--info" || argument == "--no-window") {
            info_only = true;
            continue;
        }

        if (argument == "--frames") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--frames requires a positive integer value");
            }
            viewer_options.max_frames = std::stoi(argv[++i]);
            if (viewer_options.max_frames <= 0) {
                throw std::runtime_error("--frames requires a positive integer value");
            }
            continue;
        }

        if (argument == "--mode") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--mode requires a mode name");
            }
            viewer_options.initial_mode = parse_render_mode(argv[++i]);
            continue;
        }

        if (argument == "--output") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--output requires an image path");
            }
            viewer_options.screenshot_path = argv[++i];
            continue;
        }

        if (argument == "--render-all") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--render-all requires an output directory");
            }
            viewer_options.render_all_directory = argv[++i];
            continue;
        }

        if (argument == "--landmarks") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--landmarks requires a CSV path");
            }
            landmarks_path = argv[++i];
            continue;
        }

        if (argument == "--correspondences") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--correspondences requires a CSV path");
            }
            correspondences_path = argv[++i];
            continue;
        }

        if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        }

        mesh_path = argument;
    }

    if (viewer_options.screenshot_path.has_value() &&
        viewer_options.render_all_directory.has_value()) {
        throw std::runtime_error("--output and --render-all cannot be used together");
    }

    const Mesh mesh = load_mesh(mesh_path);
    std::cout << mesh_summary(mesh);

    if (landmarks_path.has_value()) {
        const std::vector<Landmark2D> landmarks = load_landmarks_csv(*landmarks_path);
        std::cout << landmark_summary(landmarks, *landmarks_path);
    }

    if (correspondences_path.has_value()) {
        const std::vector<BfmMediaPipeCorrespondence> correspondences =
            load_bfm_mediapipe_correspondences(*correspondences_path);
        std::cout << bfm_mediapipe_correspondence_summary(correspondences,
                                                          *correspondences_path);
    }

    if (info_only) {
        return 0;
    }

    return run_viewer(mesh, viewer_options);
}

}  // namespace face_reconstruction
