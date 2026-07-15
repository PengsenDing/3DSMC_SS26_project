#include "face_recon/silhouette_fitting.h"

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "face_recon/rasterizer.h"

namespace face_recon {
namespace {

struct ShapeSample {
  Eigen::Vector3d base = Eigen::Vector3d::Zero();
  Eigen::MatrixXd basis;
};

struct RenderedContourSample {
  cv::Point pixel;
  ShapeSample surface;
};

struct SilhouetteMatch {
  ShapeSample surface;
  Eigen::Vector2d target = Eigen::Vector2d::Zero();
};

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner) : triangles(corner, triangle);
}

Eigen::MatrixXd ScaledShapeBasis(const BfmModel& model, int vertex_id, int count) {
  Eigen::MatrixXd basis = model.shape().pca_basis.block(3 * vertex_id, 0, 3, count);
  for (int index = 0; index < count; ++index) {
    basis.col(index) *= std::sqrt(std::max(model.shape().pca_variance[index], 1.0e-12));
  }
  return basis;
}

cv::Mat LoadBinaryMask(const std::string& path) {
  cv::Mat mask = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) {
    throw std::runtime_error("Could not read silhouette mask: " + path);
  }
  cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
  if (cv::countNonZero(mask) == 0) {
    throw std::runtime_error("Silhouette mask is empty: " + path);
  }
  return mask;
}

cv::Mat ResizeMask(const cv::Mat& mask, int maximum_dimension) {
  const double scale =
      std::min(1.0, maximum_dimension / static_cast<double>(std::max(mask.cols, mask.rows)));
  cv::Mat resized;
  cv::resize(mask, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
  return resized;
}

// Loads the external-occluder mask (hair, hands, clothes, accessories),
// resized to the silhouette working resolution and dilated a little so
// imprecise segmentation boundaries still count as occluded. Returns an
// empty matrix when no occluder mask was supplied.
cv::Mat LoadOccluderMask(const std::string& path, const cv::Size& size) {
  if (path.empty()) {
    return {};
  }
  cv::Mat occluder = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (occluder.empty()) {
    throw std::runtime_error("Could not read occluder mask: " + path);
  }
  if (occluder.size() != size) {
    cv::resize(occluder, occluder, size, 0.0, 0.0, cv::INTER_NEAREST);
  }
  cv::threshold(occluder, occluder, 127, 255, cv::THRESH_BINARY);
  const int radius = std::max(
      2, static_cast<int>(std::lround(0.02 * std::max(size.width, size.height))));
  cv::dilate(occluder, occluder,
             cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                       cv::Size(2 * radius + 1, 2 * radius + 1)));
  return occluder;
}

// Boundary pixels inside the occluder region are occlusion boundaries
// (face-vs-hand, face-vs-hair), not the face contour, and must not be
// matched or measured.
std::vector<cv::Point> RemoveOccludedContourPoints(
    const std::vector<cv::Point>& contour, const cv::Mat& occluder) {
  if (occluder.empty()) {
    return contour;
  }
  std::vector<cv::Point> visible;
  visible.reserve(contour.size());
  for (const cv::Point& point : contour) {
    if (occluder.at<unsigned char>(point) == 0) {
      visible.push_back(point);
    }
  }
  return visible;
}

cv::Mat RasterMask(const RasterizationResult& rasterization) {
  cv::Mat mask(rasterization.height, rasterization.width, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  return mask;
}

std::vector<cv::Point> LargestContour(const cv::Mat& mask) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.empty()) {
    return {};
  }
  return *std::max_element(
      contours.begin(), contours.end(),
      [](const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) {
        return std::abs(cv::contourArea(lhs)) < std::abs(cv::contourArea(rhs));
      });
}

double SquaredDistance(const cv::Point& lhs, const cv::Point& rhs) {
  const double dx = static_cast<double>(lhs.x - rhs.x);
  const double dy = static_cast<double>(lhs.y - rhs.y);
  return dx * dx + dy * dy;
}

int NearestPoint(const cv::Point& query, const std::vector<cv::Point>& candidates,
                 double* squared_distance) {
  int best = -1;
  *squared_distance = std::numeric_limits<double>::max();
  for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
    const double distance = SquaredDistance(query, candidates[index]);
    if (distance < *squared_distance) {
      *squared_distance = distance;
      best = index;
    }
  }
  return best;
}

Eigen::Vector2d NormalizePixel(const cv::Point& point, int width, int height) {
  return Eigen::Vector2d(point.x / static_cast<double>(std::max(width - 1, 1)),
                         point.y / static_cast<double>(std::max(height - 1, 1)));
}

const RasterPixel* FindVisiblePixel(const RasterizationResult& rasterization,
                                    const cv::Point& point, cv::Point* visible_point) {
  for (int radius = 0; radius <= 2; ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const int x = point.x + dx;
        const int y = point.y + dy;
        if (x < 0 || x >= rasterization.width || y < 0 || y >= rasterization.height) {
          continue;
        }
        const RasterPixel& pixel = rasterization.at(x, y);
        if (pixel.visible()) {
          *visible_point = cv::Point(x, y);
          return &pixel;
        }
      }
    }
  }
  return nullptr;
}

ShapeSample VertexSample(const BfmModel& model, const Eigen::VectorXd& vertices,
                         const Eigen::VectorXd& shape, int vertex_id) {
  ShapeSample sample;
  sample.basis = ScaledShapeBasis(model, vertex_id, shape.size());
  sample.base = vertices.segment<3>(3 * vertex_id) - sample.basis * shape;
  return sample;
}

ShapeSample SurfaceSample(const BfmModel& model, const Eigen::VectorXd& vertices,
                          const Eigen::VectorXd& shape, int triangle_id,
                          const Eigen::Vector3f& barycentric) {
  ShapeSample sample;
  sample.basis = Eigen::MatrixXd::Zero(3, shape.size());
  for (int corner = 0; corner < 3; ++corner) {
    const int vertex_id = TriangleIndex(model.triangles(), triangle_id, corner);
    const double weight = barycentric[corner];
    const Eigen::MatrixXd basis = ScaledShapeBasis(model, vertex_id, shape.size());
    sample.basis += weight * basis;
    sample.base += weight * (vertices.segment<3>(3 * vertex_id) - basis * shape);
  }
  return sample;
}

std::vector<RenderedContourSample> BuildRenderedSamples(const BfmModel& model,
                                                        const RasterizationResult& rasterization,
                                                        const std::vector<cv::Point>& contour,
                                                        const Eigen::VectorXd& vertices,
                                                        const Eigen::VectorXd& shape, int stride) {
  std::vector<RenderedContourSample> samples;
  for (int index = 0; index < static_cast<int>(contour.size()); index += std::max(stride, 1)) {
    cv::Point visible_point;
    const RasterPixel* pixel = FindVisiblePixel(rasterization, contour[index], &visible_point);
    if (pixel == nullptr) {
      continue;
    }
    RenderedContourSample sample;
    sample.pixel = visible_point;
    sample.surface = SurfaceSample(model, vertices, shape, pixel->triangle_id, pixel->barycentric);
    samples.push_back(std::move(sample));
  }
  return samples;
}

std::vector<SilhouetteMatch> BuildSilhouetteMatches(
    const std::vector<cv::Point>& target_contour,
    const std::vector<RenderedContourSample>& rendered_samples, int width, int height,
    int target_stride, double maximum_distance_fraction) {
  std::vector<cv::Point> rendered_points;
  rendered_points.reserve(rendered_samples.size());
  for (const auto& sample : rendered_samples) {
    rendered_points.push_back(sample.pixel);
  }

  const double maximum_distance = maximum_distance_fraction * std::max(width, height);
  const double maximum_squared_distance = maximum_distance * maximum_distance;
  std::vector<SilhouetteMatch> matches;

  // Rendered-to-target prevents the model outline from extending outside the
  // observation.
  for (const auto& sample : rendered_samples) {
    double squared_distance = 0.0;
    const int nearest = NearestPoint(sample.pixel, target_contour, &squared_distance);
    if (nearest >= 0 && squared_distance <= maximum_squared_distance) {
      matches.push_back({sample.surface, NormalizePixel(target_contour[nearest], width, height)});
    }
  }

  // Target-to-rendered prevents the model from shrinking inside the target.
  for (int index = 0; index < static_cast<int>(target_contour.size());
       index += std::max(target_stride, 1)) {
    double squared_distance = 0.0;
    const int nearest = NearestPoint(target_contour[index], rendered_points, &squared_distance);
    if (nearest >= 0 && squared_distance <= maximum_squared_distance) {
      matches.push_back({rendered_samples[nearest].surface,
                         NormalizePixel(target_contour[index], width, height)});
    }
  }
  return matches;
}

class ShapeProjectionResidual {
 public:
  ShapeProjectionResidual(ShapeSample sample, Eigen::Vector2d target, CameraParameters camera,
                          double weight)
      : sample_(std::move(sample)),
        target_(std::move(target)),
        camera_(std::move(camera)),
        sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* shape = parameters[0];
    T point[3] = {T(sample_.base.x()), T(sample_.base.y()), T(sample_.base.z())};
    for (int coefficient = 0; coefficient < sample_.basis.cols(); ++coefficient) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] += T(sample_.basis(axis, coefficient)) * shape[coefficient];
      }
    }
    T rotated[3];
    const T angle_axis[3] = {T(camera_.angle_axis.x()), T(camera_.angle_axis.y()),
                             T(camera_.angle_axis.z())};
    ceres::AngleAxisRotatePoint(angle_axis, point, rotated);
    const T depth = T(camera_.translation.z()) - rotated[2];
    const T projected_x =
        T(0.5) + T(camera_.focal_length) * (rotated[0] + T(camera_.translation.x())) / depth;
    const T projected_y = T(0.5) - T(camera_.focal_length * camera_.aspect_ratio) *
                                       (rotated[1] + T(camera_.translation.y())) / depth;
    residuals[0] = T(sqrt_weight_) * (projected_x - T(target_.x()));
    residuals[1] = T(sqrt_weight_) * (projected_y - T(target_.y()));
    return true;
  }

 private:
  ShapeSample sample_;
  Eigen::Vector2d target_;
  CameraParameters camera_;
  double sqrt_weight_;
};

class ShapePrior {
 public:
  ShapePrior(int count, double weight) : count_(count), sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    for (int index = 0; index < count_; ++index) {
      residuals[index] = T(sqrt_weight_) * parameters[0][index];
    }
    return true;
  }

 private:
  int count_;
  double sqrt_weight_;
};

void AddProjectionResidual(ceres::Problem* problem, const ShapeSample& sample,
                           const Eigen::Vector2d& target, const CameraParameters& camera,
                           double weight, double huber_delta, double* shape, int shape_count) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<ShapeProjectionResidual, 4>(
      new ShapeProjectionResidual(sample, target, camera, weight));
  cost->AddParameterBlock(shape_count);
  cost->SetNumResiduals(2);
  ceres::LossFunction* loss = nullptr;
  if (huber_delta > 0.0) {
    loss = new ceres::HuberLoss(huber_delta * std::sqrt(weight));
  }
  problem->AddResidualBlock(cost, loss, shape);
}

double SemanticRmse(const FittingResult& fitting, const Eigen::VectorXd& vertices) {
  double squared_error = 0.0;
  int count = 0;
  for (const auto& reprojection : fitting.reprojections) {
    if (reprojection.name.rfind("contour_", 0) == 0 || reprojection.bfm_vertex_id < 0 ||
        3 * reprojection.bfm_vertex_id + 2 >= vertices.size()) {
      continue;
    }
    squared_error +=
        (ProjectVertex(vertices.segment<3>(3 * reprojection.bfm_vertex_id), fitting.camera) -
         reprojection.observed)
            .squaredNorm();
    ++count;
  }
  return count > 0 ? std::sqrt(squared_error / count) : 1.0;
}

double SilhouetteIou(const cv::Mat& rendered, const cv::Mat& target) {
  cv::Mat intersection;
  cv::Mat union_mask;
  cv::bitwise_and(rendered, target, intersection);
  cv::bitwise_or(rendered, target, union_mask);
  const double union_count = cv::countNonZero(union_mask);
  return union_count > 0.0 ? cv::countNonZero(intersection) / union_count : 0.0;
}

double ChamferDistance(const std::vector<cv::Point>& first, const std::vector<cv::Point>& second,
                       int normalizer) {
  if (first.empty() || second.empty()) {
    return 1.0;
  }
  double squared_sum = 0.0;
  int count = 0;
  auto accumulate = [&](const std::vector<cv::Point>& source,
                        const std::vector<cv::Point>& target) {
    for (const cv::Point& point : source) {
      double squared_distance = 0.0;
      NearestPoint(point, target, &squared_distance);
      squared_sum += squared_distance;
      ++count;
    }
  };
  accumulate(first, second);
  accumulate(second, first);
  return std::sqrt(squared_sum / std::max(count, 1)) / std::max(normalizer, 1);
}

SilhouetteMetrics EvaluateMetrics(const BfmModel& model, const FittingResult& initial_fitting,
                                  const Eigen::VectorXd& vertices, const cv::Mat& target_mask,
                                  const cv::Mat& occluder,
                                  cv::Mat* rendered_mask_out = nullptr) {
  const RasterizationResult rasterization = RasterizeMesh(
      vertices, model.triangles(), initial_fitting.camera, target_mask.cols, target_mask.rows);
  cv::Mat rendered_mask = RasterMask(rasterization);
  const std::vector<cv::Point> target_contour =
      RemoveOccludedContourPoints(LargestContour(target_mask), occluder);
  const std::vector<cv::Point> rendered_contour =
      RemoveOccludedContourPoints(LargestContour(rendered_mask), occluder);
  SilhouetteMetrics metrics;
  metrics.semantic_rmse = SemanticRmse(initial_fitting, vertices);
  metrics.silhouette_chamfer = ChamferDistance(target_contour, rendered_contour,
                                               std::max(target_mask.cols, target_mask.rows));
  if (occluder.empty()) {
    metrics.silhouette_iou = SilhouetteIou(rendered_mask, target_mask);
  } else {
    // The mask shapes disagree behind occluders by construction; measure
    // the overlap only where the face is actually observable.
    cv::Mat visible_rendered = rendered_mask.clone();
    cv::Mat visible_target = target_mask.clone();
    visible_rendered.setTo(0, occluder);
    visible_target.setTo(0, occluder);
    metrics.silhouette_iou = SilhouetteIou(visible_rendered, visible_target);
  }
  if (rendered_mask_out != nullptr) {
    *rendered_mask_out = std::move(rendered_mask);
  }
  return metrics;
}

void SaveDiagnostics(const std::string& output_directory, const cv::Mat& target,
                     const cv::Mat& initial, const cv::Mat& refined) {
  const std::filesystem::path directory(output_directory);
  std::filesystem::create_directories(directory);
  cv::imwrite((directory / "silhouette_target.png").string(), target);
  cv::imwrite((directory / "silhouette_initial.png").string(), initial);
  cv::imwrite((directory / "silhouette_refined.png").string(), refined);

  cv::Mat overlay(target.rows, target.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<std::vector<cv::Point>> target_contours = {LargestContour(target)};
  std::vector<std::vector<cv::Point>> initial_contours = {LargestContour(initial)};
  std::vector<std::vector<cv::Point>> refined_contours = {LargestContour(refined)};
  cv::drawContours(overlay, target_contours, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  cv::drawContours(overlay, initial_contours, -1, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
  cv::drawContours(overlay, refined_contours, -1, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
  cv::imwrite((directory / "silhouette_overlay.png").string(), overlay);
}

}  // namespace

SilhouetteFittingResult RefineGeometryFromSilhouette(const std::string& silhouette_mask_path,
                                                     const BfmModel& model,
                                                     const FittingResult& initial_fitting,
                                                     const SilhouetteFittingOptions& options,
                                                     const std::string& output_directory) {
  if (initial_fitting.shape_coefficients.size() == 0) {
    throw std::runtime_error("Silhouette fitting requires shape coefficients");
  }
  const cv::Mat full_target = LoadBinaryMask(silhouette_mask_path);
  const cv::Mat target = ResizeMask(full_target, options.resolution);
  const cv::Mat occluder = LoadOccluderMask(options.occluder_mask_path, target.size());
  const std::vector<cv::Point> full_target_contour = LargestContour(target);
  if (full_target_contour.size() < 20) {
    throw std::runtime_error("Silhouette mask has no usable outer contour");
  }
  const std::vector<cv::Point> target_contour =
      RemoveOccludedContourPoints(full_target_contour, occluder);

  SilhouetteFittingResult result;
  result.shape_coefficients = initial_fitting.shape_coefficients;
  result.vertices = initial_fitting.vertices;
  cv::Mat initial_mask;
  result.initial = EvaluateMetrics(model, initial_fitting, result.vertices, target, occluder,
                                   &initial_mask);
  SilhouetteMetrics current_metrics = result.initial;
  const double maximum_semantic_rmse =
      std::max(result.initial.semantic_rmse * options.semantic_degradation_limit,
               result.initial.semantic_rmse + 0.002);

  std::ostringstream summaries;
  if (target_contour.size() < 20) {
    summaries << "Occluders cover almost the entire face boundary ("
              << target_contour.size()
              << " visible contour points); keeping the semantic-only geometry\n";
  }
  for (int outer = 0; target_contour.size() >= 20 && outer < options.outer_iterations; ++outer) {
    const RasterizationResult rasterization = RasterizeMesh(
        result.vertices, model.triangles(), initial_fitting.camera, target.cols, target.rows);
    const cv::Mat rendered_mask = RasterMask(rasterization);
    const std::vector<cv::Point> rendered_contour =
        RemoveOccludedContourPoints(LargestContour(rendered_mask), occluder);
    const std::vector<RenderedContourSample> rendered_samples =
        BuildRenderedSamples(model, rasterization, rendered_contour, result.vertices,
                             result.shape_coefficients, options.sample_stride);
    const std::vector<SilhouetteMatch> silhouette_matches =
        BuildSilhouetteMatches(target_contour, rendered_samples, target.cols, target.rows,
                               options.sample_stride, options.maximum_correspondence_distance);
    if (silhouette_matches.size() < 20) {
      summaries << "Outer " << outer + 1 << ": insufficient silhouette correspondences ("
                << silhouette_matches.size() << ")\n";
      break;
    }
    result.silhouette_correspondences = static_cast<int>(silhouette_matches.size());

    ceres::Problem problem;
    const int shape_count = static_cast<int>(result.shape_coefficients.size());
    for (const auto& reprojection : initial_fitting.reprojections) {
      if (reprojection.name.rfind("contour_", 0) == 0) {
        continue;
      }
      const ShapeSample sample = VertexSample(model, result.vertices, result.shape_coefficients,
                                              reprojection.bfm_vertex_id);
      AddProjectionResidual(&problem, sample, reprojection.observed, initial_fitting.camera,
                            options.semantic_weight * reprojection.weight_multiplier,
                            options.huber_delta,
                            result.shape_coefficients.data(), shape_count);
    }
    for (const SilhouetteMatch& match : silhouette_matches) {
      AddProjectionResidual(&problem, match.surface, match.target, initial_fitting.camera,
                            options.silhouette_weight, options.huber_delta,
                            result.shape_coefficients.data(), shape_count);
    }
    auto* prior = new ceres::DynamicAutoDiffCostFunction<ShapePrior, 4>(
        new ShapePrior(shape_count, options.shape_regularization));
    prior->AddParameterBlock(shape_count);
    prior->SetNumResiduals(shape_count);
    problem.AddResidualBlock(prior, nullptr, result.shape_coefficients.data());
    for (int index = 0; index < shape_count; ++index) {
      problem.SetParameterLowerBound(result.shape_coefficients.data(), index, -3.0);
      problem.SetParameterUpperBound(result.shape_coefficients.data(), index, 3.0);
    }

    const Eigen::VectorXd previous_shape = result.shape_coefficients;
    const Eigen::VectorXd previous_vertices = result.vertices;
    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options.solver_iterations;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    const Eigen::VectorXd candidate_vertices =
        GenerateVertices(model, result.shape_coefficients, initial_fitting.expression_coefficients);
    const SilhouetteMetrics candidate_metrics =
        EvaluateMetrics(model, initial_fitting, candidate_vertices, target, occluder);
    const bool safe =
        summary.IsSolutionUsable() && candidate_metrics.semantic_rmse <= maximum_semantic_rmse &&
        candidate_metrics.silhouette_chamfer + 1.0e-6 < current_metrics.silhouette_chamfer;
    summaries << "Outer " << outer + 1 << ": " << summary.BriefReport() << ", chamfer "
              << current_metrics.silhouette_chamfer << " -> "
              << candidate_metrics.silhouette_chamfer << ", semantic "
              << current_metrics.semantic_rmse << " -> " << candidate_metrics.semantic_rmse
              << (safe ? " (accepted)" : " (rejected)") << "\n";
    if (!safe) {
      result.shape_coefficients = previous_shape;
      result.vertices = previous_vertices;
      break;
    }
    result.vertices = candidate_vertices;
    current_metrics = candidate_metrics;
    ++result.completed_outer_iterations;
  }

  cv::Mat final_mask;
  result.final = EvaluateMetrics(model, initial_fitting, result.vertices, target, occluder,
                                 &final_mask);
  result.usable = result.completed_outer_iterations > 0 &&
                  result.final.semantic_rmse <= maximum_semantic_rmse &&
                  result.final.silhouette_chamfer <= result.initial.silhouette_chamfer;
  result.solver_summary = summaries.str();
  if (options.save_diagnostics) {
    SaveDiagnostics(output_directory, target, initial_mask, final_mask);
  }
  return result;
}

void ApplySilhouetteResult(const SilhouetteFittingResult& refinement, FittingResult* fitting) {
  if (fitting == nullptr || !refinement.usable) {
    return;
  }
  fitting->shape_coefficients = refinement.shape_coefficients;
  fitting->vertices = refinement.vertices;
  double squared_error = 0.0;
  int count = 0;
  for (auto& reprojection : fitting->reprojections) {
    if (reprojection.name.rfind("contour_", 0) == 0 || reprojection.bfm_vertex_id < 0 ||
        3 * reprojection.bfm_vertex_id + 2 >= fitting->vertices.size()) {
      continue;
    }
    reprojection.projected = ProjectVertex(
        fitting->vertices.segment<3>(3 * reprojection.bfm_vertex_id), fitting->camera);
    squared_error += (reprojection.projected - reprojection.observed).squaredNorm();
    ++count;
  }
  if (count > 0) {
    fitting->final_rmse = std::sqrt(squared_error / count);
  }
  fitting->solver_summary += "\nSilhouette stage:\n" + refinement.solver_summary;
}

bool SaveSilhouetteFittingReport(const SilhouetteFittingResult& result,
                                 const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    return false;
  }
  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "completed_outer_iterations: " << result.completed_outer_iterations << "\n";
  output << "silhouette_correspondences: " << result.silhouette_correspondences << "\n";
  auto write_metrics = [&](const char* prefix, const SilhouetteMetrics& metrics) {
    output << prefix << "_semantic_rmse: " << metrics.semantic_rmse << "\n";
    output << prefix << "_silhouette_chamfer: " << metrics.silhouette_chamfer << "\n";
    output << prefix << "_silhouette_iou: " << metrics.silhouette_iou << "\n";
  };
  write_metrics("initial", result.initial);
  write_metrics("final", result.final);
  output << "\n" << result.solver_summary;
  return true;
}

}  // namespace face_recon
