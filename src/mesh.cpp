#include "face_reconstruction/mesh.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace face_reconstruction {

bool Mesh::empty() const {
    return vertices.empty() || triangles.empty();
}

bool Mesh::has_colors() const {
    return colors.size() == vertices.size();
}

bool Mesh::has_normals() const {
    return normals.size() == vertices.size();
}

Eigen::Vector3f Mesh::center() const {
    if (vertices.empty()) {
        return Eigen::Vector3f::Zero();
    }

    Eigen::Vector3f min_corner = vertices.front();
    Eigen::Vector3f max_corner = vertices.front();

    for (const Eigen::Vector3f& vertex : vertices) {
        min_corner = min_corner.cwiseMin(vertex);
        max_corner = max_corner.cwiseMax(vertex);
    }

    return 0.5f * (min_corner + max_corner);
}

Eigen::Vector3f Mesh::extent() const {
    if (vertices.empty()) {
        return Eigen::Vector3f::Zero();
    }

    Eigen::Vector3f min_corner = vertices.front();
    Eigen::Vector3f max_corner = vertices.front();

    for (const Eigen::Vector3f& vertex : vertices) {
        min_corner = min_corner.cwiseMin(vertex);
        max_corner = max_corner.cwiseMax(vertex);
    }

    return max_corner - min_corner;
}

float Mesh::bounding_radius() const {
    if (vertices.empty()) {
        return 1.0f;
    }

    const Eigen::Vector3f mesh_center = center();
    float radius = 0.0f;

    for (const Eigen::Vector3f& vertex : vertices) {
        radius = std::max(radius, (vertex - mesh_center).norm());
    }

    return std::max(radius, 1.0e-4f);
}

void compute_vertex_normals(Mesh& mesh) {
    mesh.normals.assign(mesh.vertices.size(), Eigen::Vector3f::Zero());

    for (const Triangle& triangle : mesh.triangles) {
        const Eigen::Vector3f& a = mesh.vertices[triangle.vertex_indices[0]];
        const Eigen::Vector3f& b = mesh.vertices[triangle.vertex_indices[1]];
        const Eigen::Vector3f& c = mesh.vertices[triangle.vertex_indices[2]];
        const Eigen::Vector3f weighted_normal = (b - a).cross(c - a);

        for (const std::uint32_t vertex_index : triangle.vertex_indices) {
            mesh.normals[vertex_index] += weighted_normal;
        }
    }

    for (Eigen::Vector3f& normal : mesh.normals) {
        const float length = normal.norm();
        normal = length > 1.0e-8f ? normal / length : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }
}

std::string mesh_summary(const Mesh& mesh) {
    const Eigen::Vector3f mesh_center = mesh.center();
    const Eigen::Vector3f mesh_extent = mesh.extent();

    std::ostringstream summary;
    summary << "Mesh: " << (mesh.source_path.empty() ? "<memory>" : mesh.source_path) << '\n';
    summary << "Vertices: " << mesh.vertices.size() << '\n';
    summary << "Triangles: " << mesh.triangles.size() << '\n';
    summary << "Vertex colors: " << (mesh.has_colors() ? "yes" : "no") << '\n';
    summary << "Vertex normals: " << (mesh.has_normals() ? "yes" : "no") << '\n';
    summary << std::fixed << std::setprecision(4);
    summary << "Center: [" << mesh_center.x() << ", " << mesh_center.y() << ", "
            << mesh_center.z() << "]\n";
    summary << "Extent: [" << mesh_extent.x() << ", " << mesh_extent.y() << ", "
            << mesh_extent.z() << "]\n";
    summary << "Bounding radius: " << mesh.bounding_radius() << '\n';
    return summary.str();
}

}  // namespace face_reconstruction
