#include "face_reconstruction/obj_loader.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace face_reconstruction {
namespace {

std::uint32_t parse_obj_index(std::string_view token, std::size_t vertex_count, int line_number) {
    const std::size_t slash = token.find('/');
    const std::string vertex_part(token.substr(0, slash));

    if (vertex_part.empty()) {
        throw std::runtime_error("OBJ line " + std::to_string(line_number) +
                                 ": face vertex is missing a position index");
    }

    const int raw_index = std::stoi(vertex_part);
    if (raw_index == 0) {
        throw std::runtime_error("OBJ line " + std::to_string(line_number) +
                                 ": OBJ indices are 1-based; index 0 is invalid");
    }

    const int resolved_index =
        raw_index > 0 ? raw_index - 1 : static_cast<int>(vertex_count) + raw_index;

    if (resolved_index < 0 || resolved_index >= static_cast<int>(vertex_count)) {
        throw std::runtime_error("OBJ line " + std::to_string(line_number) +
                                 ": face references a vertex outside the loaded range");
    }

    return static_cast<std::uint32_t>(resolved_index);
}

}  // namespace

Mesh load_obj_mesh(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open OBJ mesh: " + path.string());
    }

    Mesh mesh;
    mesh.source_path = path.string();

    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;

        std::istringstream line_stream(line);
        std::string tag;
        line_stream >> tag;

        if (tag.empty() || tag == "#") {
            continue;
        }

        if (tag == "v") {
            Eigen::Vector3f vertex;
            if (!(line_stream >> vertex.x() >> vertex.y() >> vertex.z())) {
                throw std::runtime_error("OBJ line " + std::to_string(line_number) +
                                         ": invalid vertex record");
            }
            mesh.vertices.push_back(vertex);

            Eigen::Vector3f color;
            if (line_stream >> color.x() >> color.y() >> color.z()) {
                if (color.maxCoeff() > 1.0f) {
                    color /= 255.0f;
                }
                mesh.colors.push_back(color.cwiseMax(0.0f).cwiseMin(1.0f));
            } else if (!mesh.colors.empty()) {
                throw std::runtime_error(
                    "OBJ mixes colored and uncolored vertex records at line " +
                    std::to_string(line_number));
            }
            continue;
        }

        if (tag == "f") {
            std::vector<std::uint32_t> face_indices;
            std::string token;

            while (line_stream >> token) {
                face_indices.push_back(parse_obj_index(token, mesh.vertices.size(), line_number));
            }

            if (face_indices.size() < 3) {
                throw std::runtime_error("OBJ line " + std::to_string(line_number) +
                                         ": face must contain at least 3 vertices");
            }

            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
                mesh.triangles.push_back(
                    Triangle{{face_indices[0], face_indices[i], face_indices[i + 1]}});
            }
        }
    }

    if (mesh.empty()) {
        throw std::runtime_error("OBJ mesh has no renderable triangles: " + path.string());
    }
    if (!mesh.colors.empty() && mesh.colors.size() != mesh.vertices.size()) {
        throw std::runtime_error("OBJ must provide colors for either every vertex or no vertices");
    }

    compute_vertex_normals(mesh);
    return mesh;
}

}  // namespace face_reconstruction
