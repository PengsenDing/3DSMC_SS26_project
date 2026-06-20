#include "face_recon/rasterizer.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>

int main() {
  Eigen::Vector3f barycentric;
  if (!face_recon::ComputeBarycentricCoordinates(
          Eigen::Vector2f(0.25f, 0.25f),
          Eigen::Vector2f(0.0f, 0.0f),
          Eigen::Vector2f(1.0f, 0.0f),
          Eigen::Vector2f(0.0f, 1.0f),
          &barycentric) ||
      std::abs(barycentric.sum() - 1.0f) > 1.0e-5f ||
      barycentric.minCoeff() < 0.0f) {
    std::cerr << "Barycentric coordinate test failed\n";
    return 1;
  }

  // Two triangles have identical screen coordinates. The second triangle is
  // closer because larger camera-space Z points towards the frontal camera.
  Eigen::VectorXd vertices(18);
  vertices <<
      -0.3, 0.3, 1.0,
       0.3, 0.3, 1.0,
       0.0, -0.3, 1.0,
      -0.3, 0.3, 2.0,
       0.3, 0.3, 2.0,
       0.0, -0.3, 2.0;
  Eigen::MatrixXi triangles(2, 3);
  triangles << 0, 1, 2,
               3, 4, 5;

  face_recon::CameraParameters camera;
  camera.scale = 1.0;
  camera.translation_x = 0.5;
  camera.translation_y = 0.5;
  const face_recon::RasterizationResult rasterization =
      face_recon::RasterizeMesh(vertices, triangles, camera, 64, 64);

  int visible_pixels = 0;
  for (const face_recon::RasterPixel& pixel : rasterization.pixels) {
    if (!pixel.visible()) {
      continue;
    }
    ++visible_pixels;
    if (pixel.triangle_id != 1 || std::abs(pixel.depth - 2.0f) > 1.0e-5f ||
        std::abs(pixel.barycentric.sum() - 1.0f) > 1.0e-4f) {
      std::cerr << "Z-buffer or triangle coverage test failed\n";
      return 1;
    }
  }
  if (visible_pixels < 500) {
    std::cerr << "Rasterized triangle contains too few pixels\n";
    return 1;
  }

  const std::vector<bool> visible_vertices =
      face_recon::ComputeVisibleVertices(rasterization, 0.1f);
  if (visible_vertices[0] || visible_vertices[1] || visible_vertices[2] ||
      !visible_vertices[3] || !visible_vertices[4] || !visible_vertices[5]) {
    std::cerr << "Vertex visibility test failed\n";
    return 1;
  }

  std::cout << "Rasterizer test passed with " << visible_pixels
            << " visible pixels\n";
  return 0;
}
