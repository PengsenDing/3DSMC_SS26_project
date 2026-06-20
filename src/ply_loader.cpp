#include "face_reconstruction/ply_loader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace face_reconstruction {
namespace {

struct PlyHeader {
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::vector<std::string> vertex_properties;
};

PlyHeader read_header(std::istream& input, const std::filesystem::path& path) {
    std::string line;
    if (!std::getline(input, line) || line != "ply") {
        throw std::runtime_error("Not a PLY file: " + path.string());
    }

    PlyHeader header;
    std::string current_element;
    bool ascii_format = false;
    bool reached_end = false;

    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::string tag;
        stream >> tag;

        if (tag == "format") {
            std::string format;
            stream >> format;
            ascii_format = format == "ascii";
        } else if (tag == "element") {
            stream >> current_element;
            std::size_t count = 0;
            stream >> count;
            if (current_element == "vertex") {
                header.vertex_count = count;
            } else if (current_element == "face") {
                header.face_count = count;
            }
        } else if (tag == "property" && current_element == "vertex") {
            std::string type;
            std::string name;
            stream >> type >> name;
            if (type != "list" && !name.empty()) {
                header.vertex_properties.push_back(name);
            }
        } else if (tag == "end_header") {
            reached_end = true;
            break;
        }
    }

    if (!ascii_format) {
        throw std::runtime_error("Only ASCII PLY files are supported: " + path.string());
    }
    if (!reached_end || header.vertex_count == 0 || header.face_count == 0) {
        throw std::runtime_error("PLY header is incomplete: " + path.string());
    }

    return header;
}

int property_index(const std::vector<std::string>& properties, const std::string& name) {
    const auto iterator = std::find(properties.begin(), properties.end(), name);
    if (iterator == properties.end()) {
        return -1;
    }
    return static_cast<int>(std::distance(properties.begin(), iterator));
}

}  // namespace

Mesh load_ply_mesh(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open PLY mesh: " + path.string());
    }

    const PlyHeader header = read_header(input, path);
    const int x_index = property_index(header.vertex_properties, "x");
    const int y_index = property_index(header.vertex_properties, "y");
    const int z_index = property_index(header.vertex_properties, "z");
    const int red_index = property_index(header.vertex_properties, "red");
    const int green_index = property_index(header.vertex_properties, "green");
    const int blue_index = property_index(header.vertex_properties, "blue");

    if (x_index < 0 || y_index < 0 || z_index < 0) {
        throw std::runtime_error("PLY vertex properties must include x, y, and z");
    }

    const bool has_colors = red_index >= 0 && green_index >= 0 && blue_index >= 0;
    Mesh mesh;
    mesh.source_path = path.string();
    mesh.vertices.reserve(header.vertex_count);
    if (has_colors) {
        mesh.colors.reserve(header.vertex_count);
    }

    std::string line;
    for (std::size_t vertex_number = 0; vertex_number < header.vertex_count; ++vertex_number) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("PLY ended while reading vertices: " + path.string());
        }

        std::istringstream stream(line);
        std::vector<float> values;
        float value = 0.0f;
        while (stream >> value) {
            values.push_back(value);
        }
        if (values.size() < header.vertex_properties.size()) {
            throw std::runtime_error("Invalid PLY vertex record at index " +
                                     std::to_string(vertex_number));
        }

        mesh.vertices.emplace_back(values[x_index], values[y_index], values[z_index]);
        if (has_colors) {
            Eigen::Vector3f color(values[red_index], values[green_index], values[blue_index]);
            if (color.maxCoeff() > 1.0f) {
                color /= 255.0f;
            }
            mesh.colors.push_back(color.cwiseMax(0.0f).cwiseMin(1.0f));
        }
    }

    for (std::size_t face_number = 0; face_number < header.face_count; ++face_number) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("PLY ended while reading faces: " + path.string());
        }

        std::istringstream stream(line);
        std::size_t count = 0;
        stream >> count;
        if (count < 3) {
            throw std::runtime_error("PLY face has fewer than three vertices at index " +
                                     std::to_string(face_number));
        }

        std::vector<std::uint32_t> indices(count);
        for (std::uint32_t& index : indices) {
            if (!(stream >> index) || index >= mesh.vertices.size()) {
                throw std::runtime_error("Invalid PLY face index at face " +
                                         std::to_string(face_number));
            }
        }

        for (std::size_t i = 1; i + 1 < indices.size(); ++i) {
            mesh.triangles.push_back(Triangle{{indices[0], indices[i], indices[i + 1]}});
        }
    }

    if (mesh.empty()) {
        throw std::runtime_error("PLY mesh has no renderable triangles: " + path.string());
    }

    compute_vertex_normals(mesh);
    return mesh;
}

}  // namespace face_reconstruction
