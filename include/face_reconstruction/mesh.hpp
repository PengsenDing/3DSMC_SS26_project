#pragma once

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace face_reconstruction {

struct Triangle {
    std::array<std::uint32_t, 3> vertex_indices{};
};

struct Mesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> colors;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Triangle> triangles;
    std::string source_path;

    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool has_colors() const;
    [[nodiscard]] bool has_normals() const;
    [[nodiscard]] Eigen::Vector3f center() const;
    [[nodiscard]] Eigen::Vector3f extent() const;
    [[nodiscard]] float bounding_radius() const;
};

void compute_vertex_normals(Mesh& mesh);
std::string mesh_summary(const Mesh& mesh);

}  // namespace face_reconstruction
