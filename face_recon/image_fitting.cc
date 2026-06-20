#include "face_recon/image_fitting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
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

Eigen::VectorXd SampleVertexColorsFromImage(
    const std::string& image_path,
    const Eigen::VectorXd& vertices,
    const CameraParameters& camera,
    const Eigen::VectorXd& fallback_colors) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read fitting image: " + image_path);
  }

  Eigen::VectorXd colors(vertices.size());
  for (int vertex = 0; vertex < vertices.size() / 3; ++vertex) {
    const Eigen::Vector3d point = vertices.segment<3>(3 * vertex);
    const Eigen::Vector2d projected = ProjectVertex(point, camera);
    const double x = projected.x() * static_cast<double>(image.cols - 1);
    const double y = projected.y() * static_cast<double>(image.rows - 1);

    Eigen::Vector3d color = FallbackColor(fallback_colors, vertex);
    if (x >= 0.0 && x <= image.cols - 1 &&
        y >= 0.0 && y <= image.rows - 1) {
      color = BilinearRgb(image, x, y);
    }
    colors.segment<3>(3 * vertex) = color;
  }
  return colors;
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
