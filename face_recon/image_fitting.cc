#include "face_recon/image_fitting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace face_recon {
namespace {

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

}  // namespace

Eigen::Vector2i ReadImageSize(const std::string& image_path) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read fitting image: " + image_path);
  }
  return Eigen::Vector2i(image.cols, image.rows);
}

bool SaveRasterDepthImage(const RasterizationResult& rasterization,
                          const std::string& output_path) {
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  for (const RasterPixel& pixel : rasterization.pixels) {
    if (pixel.visible()) {
      minimum = std::min(minimum, pixel.depth);
      maximum = std::max(maximum, pixel.depth);
    }
  }
  if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
    std::cerr << "[ERROR] Rasterization has no visible pixels.\n";
    return false;
  }

  cv::Mat depth(rasterization.height, rasterization.width, CV_8UC1,
                cv::Scalar(0));
  const float range = std::max(maximum - minimum, 1.0e-6f);
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const RasterPixel& pixel = rasterization.at(x, y);
      if (pixel.visible()) {
        // Brighter pixels are closer to the camera.
        depth.at<unsigned char>(y, x) = static_cast<unsigned char>(
            std::clamp(255.0f * (pixel.depth - minimum) / range,
                       0.0f, 255.0f));
      }
    }
  }

  if (!std::filesystem::path(output_path).parent_path().empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path());
  }
  if (!cv::imwrite(output_path, depth)) {
    std::cerr << "[ERROR] Could not save raster depth image: "
              << output_path << "\n";
    return false;
  }
  std::cout << "[SUCCESS] Saved raster depth image: " << output_path << "\n";
  return true;
}

bool SaveVisibilityImage(const RasterizationResult& rasterization,
                         const std::string& output_path) {
  cv::Mat visibility(rasterization.height, rasterization.width, CV_8UC1,
                     cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        visibility.at<unsigned char>(y, x) = 255;
      }
    }
  }

  if (!std::filesystem::path(output_path).parent_path().empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path());
  }
  if (!cv::imwrite(output_path, visibility)) {
    std::cerr << "[ERROR] Could not save visibility image: "
              << output_path << "\n";
    return false;
  }
  std::cout << "[SUCCESS] Saved visibility image: " << output_path << "\n";
  return true;
}

bool SaveBfmSurfaceOverlay(const std::string& image_path,
                           const Eigen::MatrixXi& triangles,
                           const Eigen::VectorXd& camera_normals,
                           const RasterizationResult& rasterization,
                           const std::string& output_path,
                           double alpha) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "[ERROR] Could not read surface-overlay image: "
              << image_path << "\n";
    return false;
  }
  if (image.cols != rasterization.width ||
      image.rows != rasterization.height ||
      camera_normals.size() !=
          3 * static_cast<int>(rasterization.projected_vertices.size()) ||
      alpha < 0.0 || alpha > 1.0) {
    std::cerr << "[ERROR] Invalid BFM surface-overlay inputs.\n";
    return false;
  }

  cv::Mat overlay = image.clone();
  cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
  const Eigen::Vector3d base_bgr(235.0, 175.0, 55.0);
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = TriangleIndex(triangles, pixel.triangle_id, 0);
      const int i1 = TriangleIndex(triangles, pixel.triangle_id, 1);
      const int i2 = TriangleIndex(triangles, pixel.triangle_id, 2);
      Eigen::Vector3d normal =
          static_cast<double>(pixel.barycentric[0]) *
              camera_normals.segment<3>(3 * i0) +
          static_cast<double>(pixel.barycentric[1]) *
              camera_normals.segment<3>(3 * i1) +
          static_cast<double>(pixel.barycentric[2]) *
              camera_normals.segment<3>(3 * i2);
      if (normal.squaredNorm() > 1.0e-12) {
        normal.normalize();
      }
      const double shade = 0.45 + 0.55 * std::abs(normal.z());
      const Eigen::Vector3d surface = shade * base_bgr;
      const cv::Vec3b original = image.at<cv::Vec3b>(y, x);
      cv::Vec3b blended;
      for (int channel = 0; channel < 3; ++channel) {
        blended[channel] = static_cast<unsigned char>(std::lround(
            std::clamp((1.0 - alpha) * original[channel] +
                           alpha * surface[channel],
                       0.0, 255.0)));
      }
      overlay.at<cv::Vec3b>(y, x) = blended;
      mask.at<unsigned char>(y, x) = 255;
    }
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(overlay, contours, -1, cv::Scalar(255, 210, 70), 1,
                   cv::LINE_AA);

  if (!std::filesystem::path(output_path).parent_path().empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path());
  }
  if (!cv::imwrite(output_path, overlay)) {
    std::cerr << "[ERROR] Could not save BFM surface overlay: "
              << output_path << "\n";
    return false;
  }
  std::cout << "[SUCCESS] Saved BFM surface overlay: " << output_path << "\n";
  return true;
}

bool SaveReprojectionOverlay(const std::string& image_path,
                             const FittingResult& result,
                             const std::string& output_path) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "[ERROR] Could not read fitting image: " << image_path << "\n";
    return false;
  }

  for (const auto& reprojection : result.reprojections) {
    const cv::Point observed(
        static_cast<int>(std::lround(reprojection.observed.x() * image.cols)),
        static_cast<int>(std::lround(reprojection.observed.y() * image.rows)));
    const cv::Point projected(
        static_cast<int>(std::lround(reprojection.projected.x() * image.cols)),
        static_cast<int>(std::lround(reprojection.projected.y() * image.rows)));
    const bool contour = reprojection.name.rfind("contour_", 0) == 0;
    const cv::Scalar observed_color =
        contour ? cv::Scalar(255, 180, 0) : cv::Scalar(0, 220, 0);
    const cv::Scalar projected_color =
        contour ? cv::Scalar(255, 0, 180) : cv::Scalar(0, 0, 255);

    cv::line(image, observed, projected, cv::Scalar(0, 220, 255), 1,
             cv::LINE_AA);
    cv::circle(image, observed, 3, observed_color, -1, cv::LINE_AA);
    cv::circle(image, projected, 3, projected_color, 1, cv::LINE_AA);
  }

  if (!std::filesystem::path(output_path).parent_path().empty()) {
    std::filesystem::create_directories(
        std::filesystem::path(output_path).parent_path());
  }
  if (!cv::imwrite(output_path, image)) {
    std::cerr << "[ERROR] Could not save reprojection overlay: "
              << output_path << "\n";
    return false;
  }
  std::cout << "[SUCCESS] Saved reprojection overlay: " << output_path << "\n";
  return true;
}

}  // namespace face_recon
