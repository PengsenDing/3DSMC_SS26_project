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

namespace face_recon {
namespace {

Eigen::Vector3d BilinearRgb(const cv::Mat& image, double x, double y) {
  const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.cols - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.rows - 1);
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const double dx = x - x0;
  const double dy = y - y0;

  const cv::Vec3b c00 = image.at<cv::Vec3b>(y0, x0);
  const cv::Vec3b c10 = image.at<cv::Vec3b>(y0, x1);
  const cv::Vec3b c01 = image.at<cv::Vec3b>(y1, x0);
  const cv::Vec3b c11 = image.at<cv::Vec3b>(y1, x1);
  const cv::Vec3d bgr =
      (1.0 - dx) * (1.0 - dy) * cv::Vec3d(c00) +
      dx * (1.0 - dy) * cv::Vec3d(c10) +
      (1.0 - dx) * dy * cv::Vec3d(c01) +
      dx * dy * cv::Vec3d(c11);
  return Eigen::Vector3d(bgr[2], bgr[1], bgr[0]);
}

Eigen::Vector3d FallbackColor(const Eigen::VectorXd& colors, int vertex) {
  if (colors.size() >= 3 * vertex + 3) {
    Eigen::Vector3d color = colors.segment<3>(3 * vertex);
    if (color.maxCoeff() <= 1.05) {
      color *= 255.0;
    }
    return color.cwiseMax(0.0).cwiseMin(255.0);
  }
  return Eigen::Vector3d(184.0, 140.0, 117.0);
}

}  // namespace

Eigen::Vector2i ReadImageSize(const std::string& image_path) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read fitting image: " + image_path);
  }
  return Eigen::Vector2i(image.cols, image.rows);
}

Eigen::VectorXd SampleVisibleVertexColorsFromImage(
    const std::string& image_path,
    const Eigen::VectorXd& vertices,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& fallback_colors) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read fitting image: " + image_path);
  }

  if (rasterization.width != image.cols ||
      rasterization.height != image.rows ||
      rasterization.projected_vertices.size() !=
          static_cast<std::size_t>(vertices.size() / 3)) {
    throw std::runtime_error(
        "Rasterization dimensions do not match the fitting image or mesh");
  }

  const std::vector<bool> visible_vertices =
      ComputeVisibleVertices(rasterization, 2.0f);
  Eigen::VectorXd colors(vertices.size());
  for (int vertex = 0; vertex < vertices.size() / 3; ++vertex) {
    Eigen::Vector3d color = FallbackColor(fallback_colors, vertex);
    if (visible_vertices[vertex]) {
      const Eigen::Vector2f& pixel =
          rasterization.projected_vertices[vertex].pixel;
      const double x = pixel.x();
      const double y = pixel.y();
      color = BilinearRgb(image, x, y);
    }
    colors.segment<3>(3 * vertex) = color;
  }
  return colors;
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
