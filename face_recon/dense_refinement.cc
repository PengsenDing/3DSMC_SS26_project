#include "face_recon/dense_refinement.h"

#include "face_recon/rasterizer.h"

#include <ceres/rotation.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace face_recon {
namespace {

struct EvaluationContext {
  cv::Mat image;
  cv::Mat target_mask;
  cv::Mat region_weights;
  std::vector<LandmarkReprojection> reprojections;
  CameraParameters camera;
  Eigen::VectorXd vertex_albedo;
  Eigen::Matrix<double, 3, 4> illumination;
  DenseRefinementOptions options;
};

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

cv::Mat BuildTargetFaceMask(
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    int width,
    int height) {
  static constexpr std::array<int, 36> kFaceOval = {
      10, 338, 297, 332, 284, 251, 389, 356, 454,
      323, 361, 288, 397, 365, 379, 378, 400, 377,
      152, 148, 176, 149, 150, 136, 172, 58, 132,
      93, 234, 127, 162, 21, 54, 103, 67, 109,
  };
  std::unordered_map<int, const face_reconstruction::Landmark2D*> by_index;
  for (const auto& landmark : landmarks) {
    by_index[landmark.index] = &landmark;
  }
  std::vector<cv::Point> polygon;
  polygon.reserve(kFaceOval.size());
  for (const int index : kFaceOval) {
    const auto found = by_index.find(index);
    if (found == by_index.end()) {
      continue;
    }
    polygon.emplace_back(
        std::clamp(static_cast<int>(std::lround(
                       found->second->x_norm * (width - 1))),
                   0, width - 1),
        std::clamp(static_cast<int>(std::lround(
                       found->second->y_norm * (height - 1))),
                   0, height - 1));
  }
  if (polygon.size() < 20) {
    throw std::runtime_error("Not enough MediaPipe face-oval landmarks");
  }
  cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
  cv::fillConvexPoly(mask, polygon, cv::Scalar(255), cv::LINE_AA);
  return mask;
}

cv::Mat BuildRegionWeights(
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    int width,
    int height) {
  cv::Mat weights(height, width, CV_32FC1, cv::Scalar(1.0f));
  std::unordered_map<int, const face_reconstruction::Landmark2D*> by_index;
  for (const auto& landmark : landmarks) {
    by_index[landmark.index] = &landmark;
  }
  auto emphasize = [&](const std::vector<int>& indices,
                       float value,
                       double radius_fraction) {
    const int radius =
        std::max(2, static_cast<int>(std::lround(radius_fraction * width)));
    for (const int index : indices) {
      const auto found = by_index.find(index);
      if (found == by_index.end()) {
        continue;
      }
      const cv::Point center(
          std::clamp(static_cast<int>(std::lround(
                         found->second->x_norm * (width - 1))),
                     0, width - 1),
          std::clamp(static_cast<int>(std::lround(
                         found->second->y_norm * (height - 1))),
                     0, height - 1));
      cv::circle(weights, center, radius, cv::Scalar(value), -1,
                 cv::LINE_AA);
    }
  };
  emphasize({33, 133, 159, 145, 362, 263, 386, 374}, 3.0f, 0.035);
  emphasize({70, 63, 105, 66, 107, 336, 296, 334, 293, 300},
            2.5f, 0.025);
  emphasize({1, 2, 98, 327, 168}, 2.0f, 0.035);
  emphasize({61, 291, 13, 14, 78, 308}, 3.0f, 0.035);
  return weights;
}

cv::Mat RasterMask(const RasterizationResult& rasterization) {
  cv::Mat mask(rasterization.height, rasterization.width, CV_8UC1,
               cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  return mask;
}

double SilhouetteLoss(const cv::Mat& rendered, const cv::Mat& target) {
  cv::Mat intersection;
  cv::Mat union_mask;
  cv::bitwise_and(rendered, target, intersection);
  cv::bitwise_or(rendered, target, union_mask);
  const double union_pixels = cv::countNonZero(union_mask);
  if (union_pixels <= 0.0) {
    return 1.0;
  }
  return 1.0 - cv::countNonZero(intersection) / union_pixels;
}

Eigen::Vector4d ShBasis(const Eigen::Vector3d& normal) {
  return Eigen::Vector4d(1.0, normal.x(), normal.y(), normal.z());
}

Eigen::Vector3d Interpolate(const Eigen::VectorXd& values,
                            int i0,
                            int i1,
                            int i2,
                            const Eigen::Vector3f& barycentric) {
  return static_cast<double>(barycentric[0]) *
             values.segment<3>(3 * i0) +
         static_cast<double>(barycentric[1]) *
             values.segment<3>(3 * i1) +
         static_cast<double>(barycentric[2]) *
             values.segment<3>(3 * i2);
}

Eigen::VectorXd CameraNormals(const Eigen::VectorXd& vertices,
                              const Eigen::MatrixXi& triangles,
                              const CameraParameters& camera) {
  Eigen::VectorXd normals = ComputeVertexNormals(vertices, triangles);
  for (int vertex = 0; vertex < normals.size() / 3; ++vertex) {
    double rotated[3];
    ceres::AngleAxisRotatePoint(camera.angle_axis.data(),
                                normals.data() + 3 * vertex, rotated);
    Eigen::Vector3d normal(rotated[0], rotated[1], rotated[2]);
    normals.segment<3>(3 * vertex) =
        normal.norm() > 1.0e-12 ? normal.normalized()
                                : Eigen::Vector3d::UnitZ();
  }
  return normals;
}

double PhotometricRmse(const cv::Mat& image,
                       const cv::Mat& target_mask,
                       const cv::Mat& region_weights,
                       const BfmModel& model,
                       const RasterizationResult& rasterization,
                       const Eigen::VectorXd& camera_normals,
                       const Eigen::VectorXd& vertex_albedo,
                       const Eigen::Matrix<double, 3, 4>& illumination) {
  double squared_error = 0.0;
  double total_weight = 0.0;
  for (int y = 0; y < rasterization.height; y += 2) {
    for (int x = 0; x < rasterization.width; x += 2) {
      if (target_mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = TriangleIndex(model.triangles(), pixel.triangle_id, 0);
      const int i1 = TriangleIndex(model.triangles(), pixel.triangle_id, 1);
      const int i2 = TriangleIndex(model.triangles(), pixel.triangle_id, 2);
      Eigen::Vector3d normal =
          Interpolate(camera_normals, i0, i1, i2, pixel.barycentric);
      if (normal.norm() <= 1.0e-9) {
        continue;
      }
      normal.normalize();
      const Eigen::Vector3d albedo =
          Interpolate(vertex_albedo, i0, i1, i2, pixel.barycentric);
      const Eigen::Vector3d rendered =
          albedo.cwiseProduct(illumination * ShBasis(normal));
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      const Eigen::Vector3d observed(bgr[2] / 255.0,
                                     bgr[1] / 255.0,
                                     bgr[0] / 255.0);
      Eigen::Vector3d residual = rendered - observed;
      const double norm = residual.norm();
      if (norm > 0.12) {
        residual *= std::sqrt(0.12 / norm);
      }
      const double weight = region_weights.at<float>(y, x);
      squared_error += weight * residual.squaredNorm();
      total_weight += weight;
    }
  }
  return total_weight > 0.0
             ? std::sqrt(squared_error / (3.0 * total_weight))
             : 1.0;
}

double LandmarkRmse(const Eigen::VectorXd& vertices,
                    const CameraParameters& camera,
                    const std::vector<LandmarkReprojection>& reprojections) {
  double squared_error = 0.0;
  int count = 0;
  for (const auto& landmark : reprojections) {
    if (landmark.bfm_vertex_id < 0 ||
        3 * landmark.bfm_vertex_id + 2 >= vertices.size()) {
      continue;
    }
    const Eigen::Vector3d vertex =
        vertices.segment<3>(3 * landmark.bfm_vertex_id);
    squared_error +=
        (ProjectVertex(vertex, camera) - landmark.observed).squaredNorm();
    ++count;
  }
  return count > 0 ? std::sqrt(squared_error / count) : 1.0;
}

DenseObjective Evaluate(const BfmModel& model,
                        const EvaluationContext& context,
                        const Eigen::VectorXd& shape,
                        const Eigen::VectorXd& expression,
                        Eigen::VectorXd* vertices_out = nullptr,
                        RasterizationResult* rasterization_out = nullptr) {
  Eigen::VectorXd vertices = GenerateVertices(model, shape, expression);
  RasterizationResult rasterization =
      RasterizeMesh(vertices, model.triangles(), context.camera,
                    context.image.cols, context.image.rows);
  const cv::Mat rendered_mask = RasterMask(rasterization);
  const Eigen::VectorXd normals =
      CameraNormals(vertices, model.triangles(), context.camera);

  DenseObjective objective;
  objective.landmark_rmse =
      LandmarkRmse(vertices, context.camera, context.reprojections);
  objective.silhouette_loss =
      SilhouetteLoss(rendered_mask, context.target_mask);
  objective.photometric_rmse =
      PhotometricRmse(context.image, context.target_mask,
                      context.region_weights, model,
                      rasterization, normals, context.vertex_albedo,
                      context.illumination);
  objective.shape_prior =
      shape.size() > 0 ? shape.squaredNorm() / shape.size() : 0.0;
  objective.expression_prior =
      expression.size() > 0
          ? expression.squaredNorm() / expression.size()
          : 0.0;
  objective.total =
      context.options.landmark_weight * objective.landmark_rmse +
      context.options.silhouette_weight * objective.silhouette_loss +
      context.options.photometric_weight * objective.photometric_rmse +
      context.options.shape_prior_weight * objective.shape_prior +
      context.options.expression_prior_weight * objective.expression_prior;
  if (vertices_out != nullptr) {
    *vertices_out = std::move(vertices);
  }
  if (rasterization_out != nullptr) {
    *rasterization_out = std::move(rasterization);
  }
  return objective;
}

bool BetterAndSafe(const DenseObjective& candidate,
                   const DenseObjective& current,
                   double maximum_landmark_rmse) {
  return std::isfinite(candidate.total) &&
         candidate.total + 1.0e-8 < current.total &&
         candidate.landmark_rmse <= maximum_landmark_rmse;
}

void SaveRefinementDiagnostics(const EvaluationContext& context,
                               const BfmModel& model,
                               const Eigen::VectorXd& vertices,
                               const std::string& output_directory) {
  const RasterizationResult rasterization =
      RasterizeMesh(vertices, model.triangles(), context.camera,
                    context.image.cols, context.image.rows);
  const cv::Mat rendered_mask = RasterMask(rasterization);
  cv::Mat overlay = context.image.clone();
  overlay.setTo(cv::Scalar(0, 100, 255), rendered_mask);
  cv::addWeighted(context.image, 0.65, overlay, 0.35, 0.0, overlay);
  const std::filesystem::path directory(output_directory);
  std::filesystem::create_directories(directory);
  cv::imwrite((directory / "target_silhouette.png").string(),
              context.target_mask);
  cv::imwrite((directory / "refined_silhouette.png").string(),
              rendered_mask);
  cv::imwrite((directory / "refined_geometry_overlay.png").string(),
              overlay);
}

}  // namespace

DenseRefinementResult RefineGeometryDense(
    const std::string& image_path,
    const BfmModel& model,
    const FittingResult& initial_fitting,
    const PhotometricResult& appearance,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const DenseRefinementOptions& options,
    const std::string& output_directory) {
  const cv::Mat full_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (full_image.empty()) {
    throw std::runtime_error("Could not read dense-refinement image");
  }
  const double scale =
      std::min(1.0, options.resolution /
                        static_cast<double>(std::max(full_image.cols,
                                                     full_image.rows)));
  EvaluationContext context;
  cv::resize(full_image, context.image, cv::Size(), scale, scale,
             cv::INTER_AREA);
  context.target_mask =
      BuildTargetFaceMask(landmarks, context.image.cols, context.image.rows);
  context.region_weights =
      BuildRegionWeights(landmarks, context.image.cols, context.image.rows);
  context.reprojections = initial_fitting.reprojections;
  context.camera = initial_fitting.camera;
  context.vertex_albedo = appearance.fitted_vertex_albedo;
  context.illumination = appearance.illumination;
  context.options = options;

  DenseRefinementResult result;
  result.shape_coefficients = initial_fitting.shape_coefficients;
  result.expression_coefficients = initial_fitting.expression_coefficients;
  result.initial = Evaluate(model, context, result.shape_coefficients,
                            result.expression_coefficients);
  DenseObjective current = result.initial;
  const double maximum_landmark_rmse =
      result.initial.landmark_rmse * options.landmark_degradation_limit;
  const int shape_count =
      std::min(options.shape_coefficients,
               static_cast<int>(result.shape_coefficients.size()));
  const int expression_count =
      std::min(options.expression_coefficients,
               static_cast<int>(result.expression_coefficients.size()));
  double step = options.coefficient_step;

  for (int pass = 0; pass < std::max(options.passes, 1); ++pass) {
    int pass_updates = 0;
    auto refine_block = [&](Eigen::VectorXd* coefficients, int count) {
      for (int index = 0; index < count; ++index) {
        const double original = (*coefficients)[index];
        double best_value = original;
        DenseObjective best = current;
        for (const double direction : {-1.0, 1.0}) {
          (*coefficients)[index] =
              std::clamp(original + direction * step, -3.0, 3.0);
          const DenseObjective candidate =
              Evaluate(model, context, result.shape_coefficients,
                       result.expression_coefficients);
          if (BetterAndSafe(candidate, best, maximum_landmark_rmse)) {
            best = candidate;
            best_value = (*coefficients)[index];
          }
        }
        (*coefficients)[index] = best_value;
        if (best_value != original) {
          current = best;
          ++result.accepted_updates;
          ++pass_updates;
        }
      }
    };
    refine_block(&result.expression_coefficients, expression_count);
    refine_block(&result.shape_coefficients, shape_count);
    std::cout << "[INFO] Dense refinement pass " << pass + 1
              << ": " << pass_updates << " accepted updates, objective "
              << current.total << "\n";
    if (pass_updates == 0) {
      step *= 0.5;
      if (step < options.minimum_step) {
        break;
      }
    }
  }

  result.final = Evaluate(model, context, result.shape_coefficients,
                          result.expression_coefficients, &result.vertices);
  result.usable = result.final.total <= result.initial.total &&
                  result.final.landmark_rmse <= maximum_landmark_rmse;
  SaveRefinementDiagnostics(context, model, result.vertices, output_directory);
  return result;
}

bool SaveDenseRefinementReport(const DenseRefinementResult& result,
                               const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    return false;
  }
  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "accepted_updates: " << result.accepted_updates << "\n";
  auto write_objective = [&](const char* prefix,
                             const DenseObjective& objective) {
    output << prefix << "_landmark_rmse: " << objective.landmark_rmse << "\n";
    output << prefix << "_silhouette_loss: "
           << objective.silhouette_loss << "\n";
    output << prefix << "_photometric_rmse: "
           << objective.photometric_rmse << "\n";
    output << prefix << "_shape_prior: " << objective.shape_prior << "\n";
    output << prefix << "_expression_prior: "
           << objective.expression_prior << "\n";
    output << prefix << "_total: " << objective.total << "\n";
  };
  write_objective("initial", result.initial);
  write_objective("final", result.final);
  return true;
}

}  // namespace face_recon
