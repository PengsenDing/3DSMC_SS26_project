#include "face_recon/rasterizer.h"

#include <ceres/rotation.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace face_recon {
namespace {

int TriangleCount(const Eigen::MatrixXi& triangles) {
  return triangles.cols() == 3 ? triangles.rows() : triangles.cols();
}

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

std::vector<ProjectedVertex> ProjectVertices(const Eigen::VectorXd& vertices,
                                             const CameraParameters& camera,
                                             int width,
                                             int height) {
  const int vertex_count = static_cast<int>(vertices.size() / 3);
  std::vector<ProjectedVertex> projected(vertex_count);

  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    double rotated[3];
    ceres::AngleAxisRotatePoint(camera.angle_axis.data(),
                                vertices.data() + 3 * vertex, rotated);
    const double x_norm =
        camera.scale * rotated[0] + camera.translation_x;
    const double y_norm =
        -camera.scale * rotated[1] + camera.translation_y;
    projected[vertex].pixel =
        Eigen::Vector2f(static_cast<float>(x_norm * (width - 1)),
                        static_cast<float>(y_norm * (height - 1)));
    // The BFM face points towards +Z, so larger camera-space Z is closer.
    projected[vertex].depth = static_cast<float>(rotated[2]);
  }
  return projected;
}

}  // namespace

bool ComputeBarycentricCoordinates(const Eigen::Vector2f& point,
                                   const Eigen::Vector2f& a,
                                   const Eigen::Vector2f& b,
                                   const Eigen::Vector2f& c,
                                   Eigen::Vector3f* barycentric) {
  const Eigen::Vector2f v0 = b - a;
  const Eigen::Vector2f v1 = c - a;
  const Eigen::Vector2f v2 = point - a;
  const float denominator =
      v0.x() * v1.y() - v1.x() * v0.y();
  if (std::abs(denominator) <= 1.0e-8f) {
    return false;
  }

  const float w1 =
      (v2.x() * v1.y() - v1.x() * v2.y()) / denominator;
  const float w2 =
      (v0.x() * v2.y() - v2.x() * v0.y()) / denominator;
  *barycentric = Eigen::Vector3f(1.0f - w1 - w2, w1, w2);
  return true;
}

RasterizationResult RasterizeMesh(const Eigen::VectorXd& vertices,
                                  const Eigen::MatrixXi& triangles,
                                  const CameraParameters& camera,
                                  int width,
                                  int height) {
  if (vertices.size() == 0 || vertices.size() % 3 != 0) {
    throw std::runtime_error("Rasterizer received an invalid vertex vector");
  }
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Rasterizer dimensions must be positive");
  }

  RasterizationResult result;
  result.width = width;
  result.height = height;
  result.projected_vertices =
      ProjectVertices(vertices, camera, width, height);
  result.pixels.resize(static_cast<std::size_t>(width) * height);

  constexpr float kInsideTolerance = -1.0e-5f;
  const int vertex_count = static_cast<int>(vertices.size() / 3);
  const int triangle_count = TriangleCount(triangles);
  for (int triangle = 0; triangle < triangle_count; ++triangle) {
    const int i0 = TriangleIndex(triangles, triangle, 0);
    const int i1 = TriangleIndex(triangles, triangle, 1);
    const int i2 = TriangleIndex(triangles, triangle, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
      throw std::runtime_error("Rasterizer encountered an invalid triangle index");
    }

    const ProjectedVertex& p0 = result.projected_vertices[i0];
    const ProjectedVertex& p1 = result.projected_vertices[i1];
    const ProjectedVertex& p2 = result.projected_vertices[i2];
    const float min_x_value =
        std::min({p0.pixel.x(), p1.pixel.x(), p2.pixel.x()});
    const float max_x_value =
        std::max({p0.pixel.x(), p1.pixel.x(), p2.pixel.x()});
    const float min_y_value =
        std::min({p0.pixel.y(), p1.pixel.y(), p2.pixel.y()});
    const float max_y_value =
        std::max({p0.pixel.y(), p1.pixel.y(), p2.pixel.y()});
    const int min_x =
        std::max(0, static_cast<int>(std::floor(min_x_value)));
    const int max_x =
        std::min(width - 1, static_cast<int>(std::ceil(max_x_value)));
    const int min_y =
        std::max(0, static_cast<int>(std::floor(min_y_value)));
    const int max_y =
        std::min(height - 1, static_cast<int>(std::ceil(max_y_value)));

    if (min_x > max_x || min_y > max_y) {
      continue;
    }

    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        Eigen::Vector3f barycentric;
        const Eigen::Vector2f pixel_center(
            static_cast<float>(x) + 0.5f,
            static_cast<float>(y) + 0.5f);
        if (!ComputeBarycentricCoordinates(
                pixel_center, p0.pixel, p1.pixel, p2.pixel,
                &barycentric) ||
            barycentric.minCoeff() < kInsideTolerance) {
          continue;
        }

        const float depth =
            barycentric[0] * p0.depth +
            barycentric[1] * p1.depth +
            barycentric[2] * p2.depth;
        RasterPixel& pixel = result.at(x, y);
        if (!pixel.visible() || depth > pixel.depth) {
          pixel.triangle_id = triangle;
          pixel.depth = depth;
          pixel.barycentric = barycentric;
        }
      }
    }
  }
  return result;
}

std::vector<bool> ComputeVisibleVertices(
    const RasterizationResult& rasterization,
    float depth_tolerance) {
  std::vector<bool> visible(
      rasterization.projected_vertices.size(), false);
  for (int vertex = 0;
       vertex < static_cast<int>(rasterization.projected_vertices.size());
       ++vertex) {
    const ProjectedVertex& projected =
        rasterization.projected_vertices[vertex];
    const int x = static_cast<int>(std::lround(projected.pixel.x()));
    const int y = static_cast<int>(std::lround(projected.pixel.y()));
    if (x < 0 || x >= rasterization.width ||
        y < 0 || y >= rasterization.height) {
      continue;
    }

    bool matches_depth = false;
    // A small neighborhood avoids rejecting vertices that lie exactly on a
    // shared raster edge and therefore miss the rounded center pixel.
    for (int dy = -1; dy <= 1 && !matches_depth; ++dy) {
      for (int dx = -1; dx <= 1 && !matches_depth; ++dx) {
        const int sample_x = x + dx;
        const int sample_y = y + dy;
        if (sample_x < 0 || sample_x >= rasterization.width ||
            sample_y < 0 || sample_y >= rasterization.height) {
          continue;
        }
        const RasterPixel& pixel = rasterization.at(sample_x, sample_y);
        matches_depth =
            pixel.visible() &&
            std::abs(projected.depth - pixel.depth) <= depth_tolerance;
      }
    }
    visible[vertex] = matches_depth;
  }
  return visible;
}

}  // namespace face_recon
