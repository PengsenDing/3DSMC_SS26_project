#include "face_recon/rasterizer.h"
#include "face_recon/texture_fitting.h"

#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>

#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

cv::Vec3b ToBgr8(const Eigen::Vector3d& rgb) {
  const Eigen::Vector3d value = rgb.cwiseMax(0.0).cwiseMin(1.0) * 255.0;
  return cv::Vec3b(static_cast<unsigned char>(std::lround(value[2])),
                   static_cast<unsigned char>(std::lround(value[1])),
                   static_cast<unsigned char>(std::lround(value[0])));
}

}  // namespace

int main() {
  Eigen::VectorXd vertices(12);
  vertices <<
      -0.4,  0.4, 0.0,
       0.4,  0.4, 0.0,
       0.4, -0.4, 0.0,
      -0.4, -0.4, 0.0;
  Eigen::MatrixXi triangles(2, 3);
  triangles << 0, 1, 2,
               0, 2, 3;

  face_recon::CameraParameters camera;
  camera.scale = 1.0;
  camera.translation_x = 0.5;
  camera.translation_y = 0.5;
  const face_recon::RasterizationResult rasterization =
      face_recon::RasterizeMesh(vertices, triangles, camera, 64, 64);

  Eigen::VectorXd expected_colors(12);
  expected_colors <<
      0.9, 0.1, 0.1,
      0.1, 0.9, 0.1,
      0.1, 0.1, 0.9,
      0.9, 0.9, 0.1;
  cv::Mat image(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const face_recon::RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      Eigen::Vector3d color = Eigen::Vector3d::Zero();
      for (int corner = 0; corner < 3; ++corner) {
        const int vertex =
            TriangleIndex(triangles, pixel.triangle_id, corner);
        color += static_cast<double>(pixel.barycentric[corner]) *
                 expected_colors.segment<3>(3 * vertex);
      }
      image.at<cv::Vec3b>(y, x) = ToBgr8(color);
    }
  }

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      "face_reconstruction_texture_fitting_test";
  std::filesystem::create_directories(directory);
  const std::filesystem::path image_path = directory / "input.png";
  if (!cv::imwrite(image_path.string(), image)) {
    std::cerr << "Could not create synthetic texture input\n";
    return 1;
  }

  face_recon::TextureFittingOptions options;
  options.mask_erosion = 0;
  options.prior_weight = 1.0e-5;
  options.smoothness_weight = 0.0;
  Eigen::VectorXd initial_colors =
      Eigen::VectorXd::Constant(expected_colors.size(), 0.5);
  const face_recon::TextureFittingResult result =
      face_recon::FitDenseVertexColors(
          image_path.string(), triangles, rasterization, initial_colors,
          nullptr, options, directory.string());

  std::cout << "Dense texture RMSE: " << result.initial_rmse
            << " -> " << result.final_rmse << "\n";
  if (!result.usable || result.final_rmse >= 0.01 ||
      result.final_rmse >= result.initial_rmse * 0.1) {
    std::cerr << "Dense texture fitting did not recover the synthetic colors\n";
    return 1;
  }
  return 0;
}
