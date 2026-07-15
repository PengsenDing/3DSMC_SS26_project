#include "face_recon/image_fitting.h"
#include "face_recon/rasterizer.h"

#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>

#include <cmath>
#include <filesystem>
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

  // Two triangles have identical perspective projections. The second
  // triangle is closer because it has the larger inverse optical depth.
  Eigen::VectorXd vertices(18);
  vertices <<
      -0.3, 0.3, 1.0,
       0.3, 0.3, 1.0,
       0.0, -0.3, 1.0,
      -0.2, 0.2, 2.0,
       0.2, 0.2, 2.0,
       0.0, -0.2, 2.0;
  Eigen::MatrixXi triangles(2, 3);
  triangles << 0, 1, 2,
               3, 4, 5;

  face_recon::CameraParameters camera;
  camera.translation = Eigen::Vector3d(0.0, 0.0, 4.0);
  camera.focal_length = 2.5;
  camera.aspect_ratio = 1.0;
  const face_recon::RasterizationResult rasterization =
      face_recon::RasterizeMesh(vertices, triangles, camera, 64, 64);

  int visible_pixels = 0;
  for (const face_recon::RasterPixel& pixel : rasterization.pixels) {
    if (!pixel.visible()) {
      continue;
    }
    ++visible_pixels;
    if (pixel.triangle_id != 1 ||
        std::abs(pixel.depth - 0.5f) > 1.0e-5f ||
        std::abs(pixel.barycentric.sum() - 1.0f) > 1.0e-4f) {
      std::cerr << "Z-buffer or triangle coverage test failed\n";
      return 1;
    }
  }
  if (visible_pixels < 400) {
    std::cerr << "Rasterized triangle contains too few pixels\n";
    return 1;
  }

  const std::vector<bool> visible_vertices =
      face_recon::ComputeVisibleVertices(rasterization, 1.0e-4f);
  if (visible_vertices[0] || visible_vertices[1] || visible_vertices[2] ||
      !visible_vertices[3] || !visible_vertices[4] || !visible_vertices[5]) {
    std::cerr << "Vertex visibility test failed\n";
    return 1;
  }

  const std::filesystem::path input_path =
      std::filesystem::temp_directory_path() /
      "face_reconstruction_surface_overlay_input.png";
  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() /
      "face_reconstruction_surface_overlay_output.png";
  const cv::Mat white_image(64, 64, CV_8UC3, cv::Scalar(255, 255, 255));
  if (!cv::imwrite(input_path.string(), white_image)) {
    std::cerr << "Could not create surface-overlay test image\n";
    return 1;
  }
  Eigen::VectorXd normals(18);
  for (int vertex = 0; vertex < 6; ++vertex) {
    normals.segment<3>(3 * vertex) = Eigen::Vector3d::UnitZ();
  }
  if (!face_recon::SaveBfmSurfaceOverlay(
          input_path.string(), triangles, normals, rasterization,
          output_path.string())) {
    std::cerr << "Surface-overlay export failed\n";
    return 1;
  }
  const cv::Mat surface_overlay =
      cv::imread(output_path.string(), cv::IMREAD_COLOR);
  std::filesystem::remove(input_path);
  std::filesystem::remove(output_path);
  if (surface_overlay.empty() ||
      surface_overlay.at<cv::Vec3b>(32, 32) == cv::Vec3b(255, 255, 255) ||
      surface_overlay.at<cv::Vec3b>(0, 0) != cv::Vec3b(255, 255, 255)) {
    std::cerr << "Surface-overlay pixels are incorrect\n";
    return 1;
  }

  std::cout << "Rasterizer test passed with " << visible_pixels
            << " visible pixels\n";
  return 0;
}
