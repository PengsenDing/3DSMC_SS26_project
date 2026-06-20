#ifndef FACE_RECON_RASTERIZER_H_
#define FACE_RECON_RASTERIZER_H_

#include "face_recon/fitting.h"

#include <Eigen/Core>

#include <limits>
#include <vector>

namespace face_recon {

struct ProjectedVertex {
  Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
  float depth = -std::numeric_limits<float>::infinity();
};

struct RasterPixel {
  int triangle_id = -1;
  float depth = -std::numeric_limits<float>::infinity();
  Eigen::Vector3f barycentric = Eigen::Vector3f::Zero();

  bool visible() const { return triangle_id >= 0; }
};

struct RasterizationResult {
  int width = 0;
  int height = 0;
  std::vector<ProjectedVertex> projected_vertices;
  std::vector<RasterPixel> pixels;

  const RasterPixel& at(int x, int y) const {
    return pixels[static_cast<std::size_t>(y) * width + x];
  }
  RasterPixel& at(int x, int y) {
    return pixels[static_cast<std::size_t>(y) * width + x];
  }
};

bool ComputeBarycentricCoordinates(const Eigen::Vector2f& point,
                                   const Eigen::Vector2f& a,
                                   const Eigen::Vector2f& b,
                                   const Eigen::Vector2f& c,
                                   Eigen::Vector3f* barycentric);

RasterizationResult RasterizeMesh(const Eigen::VectorXd& vertices,
                                  const Eigen::MatrixXi& triangles,
                                  const CameraParameters& camera,
                                  int width,
                                  int height);

std::vector<bool> ComputeVisibleVertices(
    const RasterizationResult& rasterization,
    float depth_tolerance = 1.0f);

}  // namespace face_recon

#endif  // FACE_RECON_RASTERIZER_H_
