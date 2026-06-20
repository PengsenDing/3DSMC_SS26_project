#include "face_recon/texture_fitting.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
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

Eigen::VectorXd NormalizeColors(const Eigen::VectorXd& colors,
                                int vertex_count) {
  if (colors.size() != 3 * vertex_count) {
    throw std::runtime_error(
        "Dense texture fitting received the wrong number of vertex colors");
  }
  Eigen::VectorXd normalized = colors;
  if (normalized.maxCoeff() > 1.05) {
    normalized /= 255.0;
  }
  return normalized.cwiseMax(0.0).cwiseMin(1.0);
}

cv::Mat BuildFaceOvalMask(
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    int width,
    int height) {
  static constexpr std::array<int, 36> kFaceOval = {
      10, 338, 297, 332, 284, 251, 389, 356, 454,
      323, 361, 288, 397, 365, 379, 378, 400, 377,
      152, 148, 176, 149, 150, 136, 172, 58, 132,
      93, 234, 127, 162, 21, 54, 103, 67, 109,
  };
  std::vector<const face_reconstruction::Landmark2D*> by_index(478, nullptr);
  for (const auto& landmark : landmarks) {
    if (landmark.index >= 0 &&
        landmark.index < static_cast<int>(by_index.size())) {
      by_index[landmark.index] = &landmark;
    }
  }
  std::vector<cv::Point> polygon;
  polygon.reserve(kFaceOval.size());
  for (const int index : kFaceOval) {
    if (by_index[index] == nullptr) {
      continue;
    }
    polygon.emplace_back(
        std::clamp(static_cast<int>(std::lround(
                       by_index[index]->x_norm * (width - 1))),
                   0, width - 1),
        std::clamp(static_cast<int>(std::lround(
                       by_index[index]->y_norm * (height - 1))),
                   0, height - 1));
  }
  cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
  if (polygon.size() >= 20) {
    cv::fillConvexPoly(mask, polygon, cv::Scalar(255), cv::LINE_AA);
  }
  return mask;
}

cv::Mat BuildMask(
    const RasterizationResult& rasterization,
    const std::vector<face_reconstruction::Landmark2D>* landmarks,
    int erosion) {
  cv::Mat mask(rasterization.height, rasterization.width, CV_8UC1,
               cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  if (landmarks != nullptr) {
    const cv::Mat face_oval =
        BuildFaceOvalMask(*landmarks, rasterization.width,
                          rasterization.height);
    if (cv::countNonZero(face_oval) > 0) {
      cv::bitwise_and(mask, face_oval, mask);
    }
  }
  if (erosion > 0) {
    const int size = 2 * erosion + 1;
    cv::erode(mask, mask,
              cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(size, size)));
  }
  return mask;
}

std::uint64_t EdgeKey(int first, int second) {
  const auto low = static_cast<std::uint32_t>(std::min(first, second));
  const auto high = static_cast<std::uint32_t>(std::max(first, second));
  return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::vector<std::array<int, 2>> UniqueEdges(
    const Eigen::MatrixXi& triangles) {
  std::unordered_set<std::uint64_t> keys;
  std::vector<std::array<int, 2>> edges;
  for (int triangle = 0; triangle < TriangleCount(triangles); ++triangle) {
    const int vertices[3] = {
        TriangleIndex(triangles, triangle, 0),
        TriangleIndex(triangles, triangle, 1),
        TriangleIndex(triangles, triangle, 2),
    };
    for (int edge = 0; edge < 3; ++edge) {
      const int first = vertices[edge];
      const int second = vertices[(edge + 1) % 3];
      const std::uint64_t key = EdgeKey(first, second);
      if (keys.insert(key).second) {
        edges.push_back({std::min(first, second), std::max(first, second)});
      }
    }
  }
  return edges;
}

Eigen::Vector3d PixelColor(const Eigen::VectorXd& colors,
                           const Eigen::MatrixXi& triangles,
                           const RasterPixel& pixel) {
  Eigen::Vector3d color = Eigen::Vector3d::Zero();
  for (int corner = 0; corner < 3; ++corner) {
    const int vertex =
        TriangleIndex(triangles, pixel.triangle_id, corner);
    color += static_cast<double>(pixel.barycentric[corner]) *
             colors.segment<3>(3 * vertex);
  }
  return color;
}

double ComputeRmse(const cv::Mat& image,
                   const cv::Mat& mask,
                   const Eigen::MatrixXi& triangles,
                   const RasterizationResult& rasterization,
                   const Eigen::VectorXd& colors,
                   int stride,
                   int* sample_count = nullptr) {
  double squared_error = 0.0;
  int count = 0;
  for (int y = 0; y < rasterization.height; y += stride) {
    for (int x = 0; x < rasterization.width; x += stride) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      const Eigen::Vector3d observed(bgr[2] / 255.0,
                                     bgr[1] / 255.0,
                                     bgr[0] / 255.0);
      squared_error +=
          (PixelColor(colors, triangles, pixel) - observed).squaredNorm();
      ++count;
    }
  }
  if (sample_count != nullptr) {
    *sample_count = count;
  }
  return count > 0
             ? std::sqrt(squared_error / (3.0 * static_cast<double>(count)))
             : std::numeric_limits<double>::infinity();
}

cv::Vec3b ToBgr8(const Eigen::Vector3d& rgb) {
  const Eigen::Vector3d value = rgb.cwiseMax(0.0).cwiseMin(1.0) * 255.0;
  return cv::Vec3b(static_cast<unsigned char>(std::lround(value[2])),
                   static_cast<unsigned char>(std::lround(value[1])),
                   static_cast<unsigned char>(std::lround(value[0])));
}

bool SaveRenderedResult(const cv::Mat& image,
                        const cv::Mat& mask,
                        const Eigen::MatrixXi& triangles,
                        const RasterizationResult& rasterization,
                        const Eigen::VectorXd& colors,
                        const std::string& output_directory,
                        bool save_diagnostics) {
  cv::Mat rendered(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat overlay = image.clone();
  cv::Mat residual(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const Eigen::Vector3d fitted =
          PixelColor(colors, triangles, pixel);
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      const Eigen::Vector3d observed(bgr[2] / 255.0,
                                     bgr[1] / 255.0,
                                     bgr[0] / 255.0);
      rendered.at<cv::Vec3b>(y, x) = ToBgr8(fitted);
      overlay.at<cv::Vec3b>(y, x) = ToBgr8(fitted);
      residual.at<cv::Vec3b>(y, x) =
          ToBgr8((fitted - observed).cwiseAbs());
    }
  }

  const std::filesystem::path directory(output_directory);
  std::filesystem::create_directories(directory);
  if (!cv::imwrite((directory / "rendered_final.png").string(), rendered) ||
      !cv::imwrite((directory / "rendered_final_overlay.png").string(),
                   overlay)) {
    return false;
  }
  return !save_diagnostics ||
         (cv::imwrite((directory / "texture_mask.png").string(), mask) &&
          cv::imwrite((directory / "texture_residual.png").string(),
                      residual));
}

}  // namespace

TextureFittingResult FitDenseVertexColors(
    const std::string& image_path,
    const Eigen::MatrixXi& triangles,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& initial_vertex_colors,
    const std::vector<face_reconstruction::Landmark2D>* landmarks,
    const TextureFittingOptions& options,
    const std::string& output_directory) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read dense texture input: " +
                             image_path);
  }
  if (image.cols != rasterization.width ||
      image.rows != rasterization.height) {
    throw std::runtime_error(
        "Dense texture input dimensions do not match rasterization");
  }
  const int vertex_count =
      static_cast<int>(rasterization.projected_vertices.size());
  const Eigen::VectorXd initial =
      NormalizeColors(initial_vertex_colors, vertex_count);
  const cv::Mat mask =
      BuildMask(rasterization, landmarks,
                std::max(options.mask_erosion, 0));
  const int stride = std::max(options.pixel_stride, 1);

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(
      rasterization.width * rasterization.height * 9 /
      (stride * stride) + vertex_count * 8));
  Eigen::MatrixXd right_hand_side =
      Eigen::MatrixXd::Zero(vertex_count, 3);
  std::vector<bool> observed_vertices(vertex_count, false);
  int sample_count = 0;

  for (int y = 0; y < rasterization.height; y += stride) {
    for (int x = 0; x < rasterization.width; x += stride) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      const Eigen::Vector3d observed(bgr[2] / 255.0,
                                     bgr[1] / 255.0,
                                     bgr[0] / 255.0);
      int indices[3];
      for (int corner = 0; corner < 3; ++corner) {
        indices[corner] =
            TriangleIndex(triangles, pixel.triangle_id, corner);
        observed_vertices[indices[corner]] = true;
      }
      for (int row = 0; row < 3; ++row) {
        const double row_weight = pixel.barycentric[row];
        right_hand_side.row(indices[row]) +=
            row_weight * observed.transpose();
        for (int column = 0; column < 3; ++column) {
          triplets.emplace_back(
              indices[row], indices[column],
              row_weight * static_cast<double>(pixel.barycentric[column]));
        }
      }
      ++sample_count;
    }
  }
  if (sample_count < 100) {
    throw std::runtime_error(
        "Too few visible pixels for dense texture fitting");
  }

  const double prior_weight = std::max(options.prior_weight, 1.0e-8);
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    triplets.emplace_back(vertex, vertex, prior_weight);
    right_hand_side.row(vertex) +=
        prior_weight * initial.segment<3>(3 * vertex).transpose();
  }

  const double smoothness_weight =
      std::max(options.smoothness_weight, 0.0);
  if (smoothness_weight > 0.0) {
    for (const auto& edge : UniqueEdges(triangles)) {
      const int first = edge[0];
      const int second = edge[1];
      triplets.emplace_back(first, first, smoothness_weight);
      triplets.emplace_back(second, second, smoothness_weight);
      triplets.emplace_back(first, second, -smoothness_weight);
      triplets.emplace_back(second, first, -smoothness_weight);
    }
  }

  Eigen::SparseMatrix<double> normal_matrix(vertex_count, vertex_count);
  normal_matrix.setFromTriplets(
      triplets.begin(), triplets.end(),
      [](double first, double second) { return first + second; });
  normal_matrix.makeCompressed();

  Eigen::ConjugateGradient<
      Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
      Eigen::IncompleteCholesky<double>>
      solver;
  solver.setMaxIterations(std::max(options.solver_iterations, 1));
  solver.setTolerance(std::max(options.solver_tolerance, 1.0e-12));
  solver.compute(normal_matrix);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "Could not factor dense texture normal equations");
  }

  Eigen::MatrixXd solved(vertex_count, 3);
  for (int channel = 0; channel < 3; ++channel) {
    Eigen::VectorXd initial_channel(vertex_count);
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
      initial_channel[vertex] = initial[3 * vertex + channel];
    }
    solved.col(channel) =
        solver.solveWithGuess(right_hand_side.col(channel), initial_channel);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "Dense texture iterative solver did not converge");
    }
  }
  solved = solved.cwiseMax(0.0).cwiseMin(1.0);

  TextureFittingResult result;
  result.sample_count = sample_count;
  result.visible_vertex_count =
      static_cast<int>(std::count(observed_vertices.begin(),
                                  observed_vertices.end(), true));
  result.vertex_colors.resize(3 * vertex_count);
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    result.vertex_colors.segment<3>(3 * vertex) = solved.row(vertex);
  }
  result.initial_rmse =
      ComputeRmse(image, mask, triangles, rasterization, initial, stride);
  result.final_rmse =
      ComputeRmse(image, mask, triangles, rasterization,
                  result.vertex_colors, stride);
  result.usable = std::isfinite(result.final_rmse) &&
                  result.final_rmse <= result.initial_rmse;

  if (!SaveRenderedResult(image, mask, triangles, rasterization,
                          result.vertex_colors, output_directory,
                          options.save_diagnostics)) {
    throw std::runtime_error(
        "Could not save dense texture diagnostic images");
  }
  std::cout << "[RESULT] Dense texture RMSE: " << result.initial_rmse
            << " -> " << result.final_rmse << "\n";
  return result;
}

bool SaveTextureFittingReport(const TextureFittingResult& result,
                              const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    return false;
  }
  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "sample_count: " << result.sample_count << "\n";
  output << "visible_vertex_count: " << result.visible_vertex_count << "\n";
  output << "initial_rmse_rgb: " << result.initial_rmse << "\n";
  output << "final_rmse_rgb: " << result.final_rmse << "\n";
  return true;
}

}  // namespace face_recon
