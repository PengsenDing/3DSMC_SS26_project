#include "face_reconstruction/app.hpp"

#define GL_SILENCE_DEPRECATION

#include "face_reconstruction/mesh.hpp"
#include "face_reconstruction/obj_loader.hpp"
#include "face_reconstruction/viewer.hpp"

#include <GL/glew.h>
#include <SFML/Config.hpp>
#include <ceres/version.h>
#include <opencv2/core/version.hpp>

#include <Eigen/Dense>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace face_reconstruction {
namespace {

void print_usage(const char* executable_name) {
    std::cout << "Usage: " << executable_name << " [mesh.obj] [--info] [--frames N]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  mesh.obj  OBJ mesh to load. Defaults to data/model.obj.\n";
    std::cout << "  --info    Load the mesh and print its summary without opening a window.\n";
    std::cout << "  --frames  Render N frames and exit. Useful for viewer smoke tests.\n";
    std::cout << "  --deps    Print the linked dependency report.\n";
    std::cout << "  --help    Show this help text.\n";
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

        if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        }

        mesh_path = argument;
    }

    const Mesh mesh = load_obj_mesh(mesh_path);
    std::cout << mesh_summary(mesh);

    if (info_only) {
        return 0;
    }

    return run_viewer(mesh, viewer_options);
}

}  // namespace face_reconstruction
