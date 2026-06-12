#include "face_reconstruction/mesh.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace face_reconstruction {

bool Mesh::empty() const {
    return vertices.empty() || triangles.empty();
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

std::string mesh_summary(const Mesh& mesh) {
    const Eigen::Vector3f mesh_center = mesh.center();
    const Eigen::Vector3f mesh_extent = mesh.extent();

    std::ostringstream summary;
    summary << "Mesh: " << (mesh.source_path.empty() ? "<memory>" : mesh.source_path) << '\n';
    summary << "Vertices: " << mesh.vertices.size() << '\n';
    summary << "Triangles: " << mesh.triangles.size() << '\n';
    summary << std::fixed << std::setprecision(4);
    summary << "Center: [" << mesh_center.x() << ", " << mesh_center.y() << ", "
            << mesh_center.z() << "]\n";
    summary << "Extent: [" << mesh_extent.x() << ", " << mesh_extent.y() << ", "
            << mesh_extent.z() << "]\n";
    summary << "Bounding radius: " << mesh.bounding_radius() << '\n';
    return summary.str();
}

}  // namespace face_reconstruction
