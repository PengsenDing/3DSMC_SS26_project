#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace face_recon {
namespace {

struct MatchedLandmark {
  std::string name;
  int mediapipe_index = 0;
  int bfm_vertex_id = 0;
  Eigen::Vector2d target = Eigen::Vector2d::Zero();
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::MatrixXd shape_basis;
  Eigen::MatrixXd expression_basis;
  double weight = 1.0;
};

Eigen::MatrixXd ScaledBasisForVertex(const PcaComponent& component,
                                     int vertex_id,
                                     int count) {
  Eigen::MatrixXd basis = component.pca_basis.block(3 * vertex_id, 0, 3, count);
  for (int index = 0; index < count; ++index) {
    basis.col(index) *=
        std::sqrt(std::max(component.pca_variance[index], 1.0e-12));
  }
  return basis;
}

class LandmarkResidual {
 public:
  LandmarkResidual(MatchedLandmark landmark, double aspect_ratio)
      : landmark_(std::move(landmark)),
        aspect_ratio_(aspect_ratio),
        sqrt_weight_(std::sqrt(landmark_.weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* camera = parameters[0];
    const T* shape = parameters[1];
    const T* expression = parameters[2];

    T point[3] = {
        T(landmark_.mean.x()),
        T(landmark_.mean.y()),
        T(landmark_.mean.z()),
    };
    for (int column = 0; column < landmark_.shape_basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] += T(landmark_.shape_basis(axis, column)) * shape[column];
      }
    }
    for (int column = 0; column < landmark_.expression_basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] +=
            T(landmark_.expression_basis(axis, column)) * expression[column];
      }
    }

    T rotated[3];
    ceres::AngleAxisRotatePoint(camera, point, rotated);
    using std::exp;
    const T translation_z = exp(camera[5]);
    const T focal_length = exp(camera[6]);
    const T depth = translation_z - rotated[2];
    const T projected_x =
        T(0.5) + focal_length * (rotated[0] + camera[3]) / depth;
    const T projected_y =
        T(0.5) - focal_length * T(aspect_ratio_) *
                     (rotated[1] + camera[4]) / depth;
    residuals[0] =
        T(sqrt_weight_) * (projected_x - T(landmark_.target.x()));
    residuals[1] =
        T(sqrt_weight_) * (projected_y - T(landmark_.target.y()));
    return true;
  }

 private:
  MatchedLandmark landmark_;
  double aspect_ratio_;
  double sqrt_weight_;
};

class EyelidGapResidual {
 public:
  EyelidGapResidual(MatchedLandmark upper, MatchedLandmark lower,
                    double aspect_ratio, double weight)
      : upper_(std::move(upper)),
        lower_(std::move(lower)),
        aspect_ratio_(aspect_ratio),
        sqrt_weight_(std::sqrt(weight)),
        observed_gap_(lower_.target - upper_.target) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    T upper[2];
    T lower[2];
    Project(upper_, parameters, upper);
    Project(lower_, parameters, lower);
    residuals[0] = T(sqrt_weight_) *
                   ((lower[0] - upper[0]) - T(observed_gap_.x()));
    residuals[1] = T(sqrt_weight_) *
                   ((lower[1] - upper[1]) - T(observed_gap_.y()));
    return true;
  }

 private:
  template <typename T>
  void Project(const MatchedLandmark& landmark,
               T const* const* parameters,
               T* projected) const {
    const T* camera = parameters[0];
    const T* shape = parameters[1];
    const T* expression = parameters[2];
    T point[3] = {T(landmark.mean.x()), T(landmark.mean.y()),
                  T(landmark.mean.z())};
    for (int column = 0; column < landmark.shape_basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] += T(landmark.shape_basis(axis, column)) * shape[column];
      }
    }
    for (int column = 0; column < landmark.expression_basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] +=
            T(landmark.expression_basis(axis, column)) * expression[column];
      }
    }
    T rotated[3];
    ceres::AngleAxisRotatePoint(camera, point, rotated);
    using std::exp;
    const T depth = exp(camera[5]) - rotated[2];
    const T focal_length = exp(camera[6]);
    projected[0] =
        T(0.5) + focal_length * (rotated[0] + camera[3]) / depth;
    projected[1] =
        T(0.5) - focal_length * T(aspect_ratio_) *
                     (rotated[1] + camera[4]) / depth;
  }

  MatchedLandmark upper_;
  MatchedLandmark lower_;
  double aspect_ratio_;
  double sqrt_weight_;
  Eigen::Vector2d observed_gap_;
};

class NormalizedPcaPrior {
 public:
  NormalizedPcaPrior(int count, double weight)
      : count_(count), sqrt_weight_(std::sqrt(weight)) {}

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

// Quadratic pull of a parameter block toward a fixed reference vector; used
// for the temporal (previous frame) and acceleration (constant-velocity
// extrapolation) priors during sequence tracking.
class ReferencePrior {
 public:
  ReferencePrior(Eigen::VectorXd reference, double weight)
      : reference_(std::move(reference)), sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    for (int index = 0; index < reference_.size(); ++index) {
      residuals[index] =
          T(sqrt_weight_) * (parameters[0][index] - T(reference_[index]));
    }
    return true;
  }

 private:
  Eigen::VectorXd reference_;
  double sqrt_weight_;
};

void AddReferencePrior(ceres::Problem& problem,
                       double* parameters,
                       int count,
                       const Eigen::VectorXd& reference,
                       double weight) {
  if (count <= 0 || weight <= 0.0 || reference.size() != count) {
    return;
  }
  auto* cost = new ceres::DynamicAutoDiffCostFunction<ReferencePrior, 4>(
      new ReferencePrior(reference, weight));
  cost->AddParameterBlock(count);
  cost->SetNumResiduals(count);
  problem.AddResidualBlock(cost, nullptr, parameters);
}

class FocalLengthPrior {
 public:
  FocalLengthPrior(double target, double weight)
      : target_(std::log(target)), sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    residuals[0] =
        T(sqrt_weight_) * (parameters[0][6] - T(target_));
    return true;
  }

 private:
  double target_;
  double sqrt_weight_;
};

Eigen::Vector2d ProjectPoint(const Eigen::Vector3d& point,
                             const std::array<double, 7>& camera,
                             double aspect_ratio) {
  double rotated[3];
  ceres::AngleAxisRotatePoint(camera.data(), point.data(), rotated);
  const double translation_z = std::exp(camera[5]);
  const double focal_length = std::exp(camera[6]);
  const double depth = translation_z - rotated[2];
  return Eigen::Vector2d(
      0.5 + focal_length * (rotated[0] + camera[3]) / depth,
      0.5 - focal_length * aspect_ratio *
                (rotated[1] + camera[4]) / depth);
}

double InferImageAspectRatio(
    const std::vector<face_reconstruction::Landmark2D>& landmarks) {
  std::vector<double> widths;
  std::vector<double> heights;
  for (const auto& landmark : landmarks) {
    if (std::abs(landmark.x_norm) > 1.0e-6 && landmark.u > 0.0f) {
      widths.push_back(landmark.u / landmark.x_norm);
    }
    if (std::abs(landmark.y_norm) > 1.0e-6 && landmark.v > 0.0f) {
      heights.push_back(landmark.v / landmark.y_norm);
    }
  }
  if (widths.empty() || heights.empty()) {
    return 1.0;
  }
  const auto median = [](std::vector<double>* values) {
    const auto middle = values->begin() + values->size() / 2;
    std::nth_element(values->begin(), middle, values->end());
    return *middle;
  };
  const double width = median(&widths);
  const double height = median(&heights);
  return width > 0.0 && height > 0.0 ? width / height : 1.0;
}

std::pair<int, int> InferImageDimensions(
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    double aspect_ratio) {
  std::vector<double> widths;
  std::vector<double> heights;
  for (const auto& landmark : landmarks) {
    if (std::abs(landmark.x_norm) > 1.0e-6 && landmark.u > 0.0f) {
      widths.push_back(landmark.u / landmark.x_norm);
    }
    if (std::abs(landmark.y_norm) > 1.0e-6 && landmark.v > 0.0f) {
      heights.push_back(landmark.v / landmark.y_norm);
    }
  }
  const auto median = [](std::vector<double>* values) {
    const auto middle = values->begin() + values->size() / 2;
    std::nth_element(values->begin(), middle, values->end());
    return *middle;
  };
  if (!widths.empty() && !heights.empty()) {
    return {std::max(1, static_cast<int>(std::lround(median(&widths)))),
            std::max(1, static_cast<int>(std::lround(median(&heights))))};
  }

  constexpr int kFallbackWidth = 1024;
  const int fallback_height = std::max(
      1, static_cast<int>(std::lround(kFallbackWidth /
                                      std::max(aspect_ratio, 1.0e-6))));
  return {kFallbackWidth, fallback_height};
}

std::vector<MatchedLandmark> BuildMatchedLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    int num_shape,
    int num_expression,
    double weight) {
  std::unordered_map<int, const face_reconstruction::Landmark2D*> landmark_by_index;
  for (const auto& landmark : landmarks) {
    landmark_by_index[landmark.index] = &landmark;
  }

  const int vertex_count = static_cast<int>(model.shape().mean.size() / 3);
  std::vector<MatchedLandmark> matches;
  for (const auto& correspondence : correspondences) {
    const auto observation = landmark_by_index.find(correspondence.mediapipe_index);
    if (observation == landmark_by_index.end()) {
      continue;
    }
    if (correspondence.bfm_vertex_id < 0 ||
        correspondence.bfm_vertex_id >= vertex_count) {
      throw std::runtime_error("Correspondence references invalid BFM vertex " +
                               std::to_string(correspondence.bfm_vertex_id));
    }

    MatchedLandmark match;
    match.name = correspondence.bfm_landmark_name;
    match.mediapipe_index = correspondence.mediapipe_index;
    match.bfm_vertex_id = correspondence.bfm_vertex_id;
    match.target =
        Eigen::Vector2d(observation->second->x_norm, observation->second->y_norm);
    match.mean = model.GetMeanVertex(match.bfm_vertex_id);
    match.shape_basis =
        ScaledBasisForVertex(model.shape(), match.bfm_vertex_id, num_shape);
    match.expression_basis =
        ScaledBasisForVertex(model.expression(), match.bfm_vertex_id, num_expression);
    match.weight = weight;
    matches.push_back(std::move(match));
  }

  if (matches.size() < 6) {
    throw std::runtime_error(
        "At least 6 matched landmarks are required for fitting; found " +
        std::to_string(matches.size()));
  }
  return matches;
}

std::array<double, 7> InitializeCamera(
    const std::vector<MatchedLandmark>& matches,
    double aspect_ratio) {
  Eigen::Vector2d target_min =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d target_max =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());
  Eigen::Vector2d model_min =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d model_max =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());
  double mean_z = 0.0;

  for (const auto& match : matches) {
    target_min = target_min.cwiseMin(match.target);
    target_max = target_max.cwiseMax(match.target);
    const Eigen::Vector2d model_xy(match.mean.x(), match.mean.y());
    model_min = model_min.cwiseMin(model_xy);
    model_max = model_max.cwiseMax(model_xy);
    mean_z += match.mean.z();
  }

  const Eigen::Vector2d target_extent = target_max - target_min;
  const Eigen::Vector2d model_extent = model_max - model_min;
  const double scale_x =
      target_extent.x() / std::max(model_extent.x(), 1.0e-9);
  const double scale_y =
      target_extent.y() /
      std::max(aspect_ratio * model_extent.y(), 1.0e-9);
  const double weak_scale =
      std::max(0.5 * (scale_x + scale_y), 1.0e-9);
  const Eigen::Vector2d target_center = 0.5 * (target_min + target_max);
  const Eigen::Vector2d model_center = 0.5 * (model_min + model_max);
  constexpr double kInitialFocalLength = 1.2;
  const double optical_depth = kInitialFocalLength / weak_scale;
  const double translation_x =
      (target_center.x() - 0.5) / weak_scale - model_center.x();
  const double translation_y =
      -(target_center.y() - 0.5) /
          (weak_scale * aspect_ratio) -
      model_center.y();
  mean_z /= static_cast<double>(matches.size());
  const double translation_z =
      std::max(optical_depth + mean_z, mean_z + 10.0);
  return {0.0, 0.0, 0.0,
          translation_x, translation_y,
          std::log(translation_z), std::log(kInitialFocalLength)};
}

double ComputeRmse(const std::vector<MatchedLandmark>& matches,
                   const std::array<double, 7>& camera,
                   double aspect_ratio,
                   const Eigen::VectorXd& shape,
                   const Eigen::VectorXd& expression) {
  double squared_error = 0.0;
  for (const auto& match : matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    squared_error +=
        (ProjectPoint(point, camera, aspect_ratio) - match.target)
            .squaredNorm();
  }
  return std::sqrt(squared_error / static_cast<double>(matches.size()));
}

std::vector<MatchedLandmark> FilterOutliers(
    const std::vector<MatchedLandmark>& matches,
    const std::array<double, 7>& camera,
    double aspect_ratio,
    const Eigen::VectorXd& shape,
    const Eigen::VectorXd& expression,
    double absolute_threshold,
    int* rejected_count) {
  std::vector<double> errors;
  for (const auto& match : matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    errors.push_back(
        (ProjectPoint(point, camera, aspect_ratio) - match.target).norm());
  }

  std::vector<double> sorted_errors = errors;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  const double median = sorted_errors[sorted_errors.size() / 2];
  std::vector<double> deviations;
  for (const double error : errors) {
    deviations.push_back(std::abs(error - median));
  }
  std::sort(deviations.begin(), deviations.end());
  const double mad = deviations[deviations.size() / 2];
  const double threshold =
      std::max(absolute_threshold, median + 3.0 * 1.4826 * mad);

  std::vector<MatchedLandmark> filtered;
  for (int index = 0; index < static_cast<int>(matches.size()); ++index) {
    if (errors[index] <= threshold) {
      filtered.push_back(matches[index]);
    }
  }
  if (filtered.size() < 12) {
    filtered = matches;
  }
  *rejected_count = static_cast<int>(matches.size() - filtered.size());
  return filtered;
}

void AddLandmarkResiduals(ceres::Problem& problem,
                          const std::vector<MatchedLandmark>& matches,
                          const FittingOptions& options,
                          double* camera,
                          double* shape,
                          double* expression,
                          int num_shape,
                          int num_expression,
                          double aspect_ratio) {
  for (const auto& match : matches) {
    auto* cost = new ceres::DynamicAutoDiffCostFunction<LandmarkResidual, 4>(
        new LandmarkResidual(match, aspect_ratio));
    cost->AddParameterBlock(7);
    cost->AddParameterBlock(num_shape);
    cost->AddParameterBlock(num_expression);
    cost->SetNumResiduals(2);

    ceres::LossFunction* loss = nullptr;
    if (options.huber_delta > 0.0) {
      loss = new ceres::HuberLoss(
          options.huber_delta * std::sqrt(match.weight));
    }
    problem.AddResidualBlock(cost, loss, camera, shape, expression);
  }
}

bool IsEyelidLandmark(const std::string& name) {
  return name.find(".eye.top") != std::string::npos ||
         name.find(".eye.bottom") != std::string::npos;
}

void ApplyEyelidWeights(std::vector<MatchedLandmark>* matches,
                        double multiplier) {
  if (multiplier <= 1.0) return;
  for (auto& match : *matches) {
    if (IsEyelidLandmark(match.name)) match.weight *= multiplier;
  }
}

enum class LandmarkRegion { kLeftEye, kRightEye, kMouth, kOther };

LandmarkRegion RegionOf(const std::string& name) {
  // The trailing dot keeps "left.eyebrow.*" out of the eye region.
  if (name.rfind("left.eye.", 0) == 0) return LandmarkRegion::kLeftEye;
  if (name.rfind("right.eye.", 0) == 0) return LandmarkRegion::kRightEye;
  if (name.find("lips") != std::string::npos ||
      name.find("chin") != std::string::npos) {
    return LandmarkRegion::kMouth;
  }
  return LandmarkRegion::kOther;
}

int CountRegion(const std::vector<MatchedLandmark>& matches,
                LandmarkRegion region) {
  return static_cast<int>(
      std::count_if(matches.begin(), matches.end(),
                    [region](const MatchedLandmark& match) {
                      return RegionOf(match.name) == region;
                    }));
}

double RegionRmse(const std::vector<MatchedLandmark>& matches,
                  LandmarkRegion region,
                  const std::array<double, 7>& camera,
                  double aspect_ratio,
                  const Eigen::VectorXd& shape,
                  const Eigen::VectorXd& expression) {
  double squared_error = 0.0;
  int count = 0;
  for (const auto& match : matches) {
    if (RegionOf(match.name) != region) continue;
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    squared_error +=
        (ProjectPoint(point, camera, aspect_ratio) - match.target)
            .squaredNorm();
    ++count;
  }
  return count > 0 ? std::sqrt(squared_error / count)
                   : std::numeric_limits<double>::quiet_NaN();
}

double RmseConfidence(double rmse, double good, double bad) {
  if (!std::isfinite(rmse)) return 0.0;
  if (bad <= good) return rmse <= good ? 1.0 : 0.0;
  return std::clamp((bad - rmse) / (bad - good), 0.0, 1.0);
}

double RegionSupportConfidence(const std::vector<MatchedLandmark>& surviving,
                               LandmarkRegion region,
                               int total,
                               const std::array<double, 7>& camera,
                               double aspect_ratio,
                               const Eigen::VectorXd& shape,
                               const Eigen::VectorXd& expression,
                               const TrackingOptions& options) {
  if (total <= 0) return 0.0;
  const int kept = CountRegion(surviving, region);
  if (kept == 0) return 0.0;
  const double support = static_cast<double>(kept) / total;
  const double rmse =
      RegionRmse(surviving, region, camera, aspect_ratio, shape, expression);
  return support * RmseConfidence(rmse, options.confidence_rmse_good,
                                  options.confidence_rmse_bad);
}

int AddEyelidGapResiduals(ceres::Problem& problem,
                          const std::vector<MatchedLandmark>& matches,
                          double* camera,
                          double* shape,
                          double* expression,
                          int num_shape,
                          int num_expression,
                          double aspect_ratio,
                          double weight,
                          double huber_delta,
                          double left_weight_scale = 1.0,
                          double right_weight_scale = 1.0) {
  if (weight <= 0.0) return 0;
  static const std::array<std::pair<const char*, const char*>, 6> kPairs = {{
      {"left.eye.top.outer_mid", "left.eye.bottom.outer_mid"},
      {"left.eye.top", "left.eye.bottom"},
      {"left.eye.top.inner_mid", "left.eye.bottom.inner_mid"},
      {"right.eye.top.outer_mid", "right.eye.bottom.outer_mid"},
      {"right.eye.top", "right.eye.bottom"},
      {"right.eye.top.inner_mid", "right.eye.bottom.inner_mid"},
  }};
  const auto find = [&matches](const char* name) -> const MatchedLandmark* {
    const auto it = std::find_if(
        matches.begin(), matches.end(),
        [name](const MatchedLandmark& match) { return match.name == name; });
    return it == matches.end() ? nullptr : &*it;
  };
  int pair_count = 0;
  for (const auto& [upper_name, lower_name] : kPairs) {
    const MatchedLandmark* upper = find(upper_name);
    const MatchedLandmark* lower = find(lower_name);
    // If either lid is hidden by yaw/Z-buffer filtering, the pair carries no
    // reliable aperture observation and must not enter the optimization.
    if (upper == nullptr || lower == nullptr) continue;
    const double side_scale =
        RegionOf(upper_name) == LandmarkRegion::kLeftEye ? left_weight_scale
                                                         : right_weight_scale;
    const double pair_weight = weight * side_scale;
    if (pair_weight <= 0.0) continue;
    auto* cost = new ceres::DynamicAutoDiffCostFunction<EyelidGapResidual, 4>(
        new EyelidGapResidual(*upper, *lower, aspect_ratio, pair_weight));
    cost->AddParameterBlock(7);
    cost->AddParameterBlock(num_shape);
    cost->AddParameterBlock(num_expression);
    cost->SetNumResiduals(2);
    ceres::LossFunction* loss = nullptr;
    if (huber_delta > 0.0) {
      loss = new ceres::HuberLoss(huber_delta * std::sqrt(pair_weight));
    }
    problem.AddResidualBlock(cost, loss, camera, shape, expression);
    ++pair_count;
  }
  return pair_count;
}

void AddRegularization(ceres::Problem& problem,
                       double* coefficients,
                       int count,
                       double weight) {
  if (count <= 0 || weight <= 0.0) {
    return;
  }
  auto* cost = new ceres::DynamicAutoDiffCostFunction<NormalizedPcaPrior, 4>(
      new NormalizedPcaPrior(count, weight));
  cost->AddParameterBlock(count);
  cost->SetNumResiduals(count);
  problem.AddResidualBlock(cost, nullptr, coefficients);
  for (int index = 0; index < count; ++index) {
    problem.SetParameterLowerBound(coefficients, index, -3.0);
    problem.SetParameterUpperBound(coefficients, index, 3.0);
  }
}

void AddFocalRegularization(ceres::Problem& problem,
                            double* camera,
                            double weight) {
  if (weight <= 0.0) {
    return;
  }
  auto* cost =
      new ceres::DynamicAutoDiffCostFunction<FocalLengthPrior, 4>(
          new FocalLengthPrior(1.2, weight));
  cost->AddParameterBlock(7);
  cost->SetNumResiduals(1);
  problem.AddResidualBlock(cost, nullptr, camera);
}

void ConfigureCameraBounds(ceres::Problem& problem,
                           double* camera,
                           double minimum_translation_z) {
  problem.SetParameterLowerBound(camera, 3, -500.0);
  problem.SetParameterUpperBound(camera, 3, 500.0);
  problem.SetParameterLowerBound(camera, 4, -500.0);
  problem.SetParameterUpperBound(camera, 4, 500.0);
  problem.SetParameterLowerBound(
      camera, 5, std::log(std::max(minimum_translation_z, 1.0)));
  problem.SetParameterUpperBound(camera, 5, std::log(5000.0));
  problem.SetParameterLowerBound(camera, 6, std::log(0.35));
  problem.SetParameterUpperBound(camera, 6, std::log(3.0));
}

ceres::Solver::Summary SolveProblem(ceres::Problem& problem,
                                    int iterations,
                                    bool verbose) {
  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = iterations;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  solver_options.minimizer_progress_to_stdout = verbose;
  solver_options.num_threads = 1;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  return summary;
}

std::array<double, 7> CameraArray(const CameraParameters& camera) {
  return {camera.angle_axis.x(), camera.angle_axis.y(), camera.angle_axis.z(),
          camera.translation.x(), camera.translation.y(),
          std::log(camera.translation.z()), std::log(camera.focal_length)};
}

CameraParameters CameraParametersFromArray(
    const std::array<double, 7>& camera,
    double aspect_ratio) {
  CameraParameters result;
  result.angle_axis = Eigen::Vector3d(camera[0], camera[1], camera[2]);
  result.translation =
      Eigen::Vector3d(camera[3], camera[4], std::exp(camera[5]));
  result.focal_length = std::exp(camera[6]);
  result.aspect_ratio = aspect_ratio;
  return result;
}

std::vector<MatchedLandmark> FilterOccludedLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<MatchedLandmark>& matches,
    const std::array<double, 7>& camera,
    double aspect_ratio,
    const Eigen::VectorXd& shape,
    const Eigen::VectorXd& expression,
    float depth_tolerance,
    int* occluded_count,
    bool* filter_applied) {
  *occluded_count = 0;
  *filter_applied = false;
  if (model.triangles().size() == 0) {
    return matches;
  }

  const auto [width, height] = InferImageDimensions(landmarks, aspect_ratio);
  const Eigen::VectorXd vertices = GenerateVertices(model, shape, expression);
  const RasterizationResult rasterization = RasterizeMesh(
      vertices, model.triangles(),
      CameraParametersFromArray(camera, aspect_ratio), width, height);
  const std::vector<bool> visible_vertices =
      ComputeVisibleVertices(rasterization, depth_tolerance);

  std::vector<MatchedLandmark> visible_matches;
  visible_matches.reserve(matches.size());
  for (const auto& match : matches) {
    if (match.bfm_vertex_id >= 0 &&
        match.bfm_vertex_id < static_cast<int>(visible_vertices.size()) &&
        visible_vertices[match.bfm_vertex_id]) {
      visible_matches.push_back(match);
    }
  }

  // Six 2D/3D correspondences are the minimum accepted by this fitter. A bad
  // initialization must not be allowed to discard every constraint.
  if (visible_matches.size() < 6) {
    return matches;
  }
  *occluded_count =
      static_cast<int>(matches.size() - visible_matches.size());
  *filter_applied = true;
  return visible_matches;
}

std::vector<MatchedLandmark> FilterLandmarksByFaceMask(
    const std::vector<MatchedLandmark>& matches,
    const std::string& mask_path,
    double normalized_tolerance,
    int* rejected_count,
    int* corrected_count) {
  *rejected_count = 0;
  *corrected_count = 0;
  if (mask_path.empty()) {
    return matches;
  }
  cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) {
    throw std::runtime_error("Could not read landmark face mask: " + mask_path);
  }
  cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
  cv::Mat outside;
  cv::bitwise_not(mask, outside);
  cv::Mat distance;
  cv::distanceTransform(outside, distance, cv::DIST_L2, 3);
  const double tolerance_pixels =
      normalized_tolerance * std::max(mask.cols, mask.rows);
  const double maximum_correction = 0.08 * std::max(mask.cols, mask.rows);
  std::vector<cv::Point> face_pixels;
  cv::findNonZero(mask, face_pixels);
  const auto is_boundary_landmark = [](const std::string& name) {
    return name == "center.nose.tip" || name == "center.chin.tip" ||
           name.find("lips.corner") != std::string::npos ||
           name.find("nose.wing.outer") != std::string::npos;
  };

  std::vector<MatchedLandmark> filtered;
  filtered.reserve(matches.size());
  for (const auto& match : matches) {
    const int x = static_cast<int>(std::lround(match.target.x() * (mask.cols - 1)));
    const int y = static_cast<int>(std::lround(match.target.y() * (mask.rows - 1)));
    if (x < 0 || x >= mask.cols || y < 0 || y >= mask.rows) {
      continue;
    }
    const double mask_distance = distance.at<float>(y, x);
    if (mask_distance <= tolerance_pixels) {
      filtered.push_back(match);
      continue;
    }
    if (is_boundary_landmark(match.name) &&
        mask_distance <= maximum_correction && !face_pixels.empty()) {
      const auto nearest = std::min_element(
          face_pixels.begin(), face_pixels.end(),
          [x, y](const cv::Point& lhs, const cv::Point& rhs) {
            const int lhs_dx = lhs.x - x;
            const int lhs_dy = lhs.y - y;
            const int rhs_dx = rhs.x - x;
            const int rhs_dy = rhs.y - y;
            return lhs_dx * lhs_dx + lhs_dy * lhs_dy <
                   rhs_dx * rhs_dx + rhs_dy * rhs_dy;
          });
      MatchedLandmark corrected = match;
      corrected.target = Eigen::Vector2d(
          static_cast<double>(nearest->x) / std::max(mask.cols - 1, 1),
          static_cast<double>(nearest->y) / std::max(mask.rows - 1, 1));
      // A corrected boundary point carries direct silhouette evidence. Give
      // it enough influence to survive the robust loss and competing sparse
      // landmarks, while leaving ordinary detector observations unchanged.
      corrected.weight *= 9.0;
      filtered.push_back(std::move(corrected));
      ++*corrected_count;
    }
  }
  if (filtered.size() < 6) {
    return matches;
  }
  *rejected_count = static_cast<int>(matches.size() - filtered.size());
  return filtered;
}

double CameraYaw(const std::array<double, 7>& camera) {
  double rotation_values[9];
  ceres::AngleAxisToRotationMatrix(camera.data(), rotation_values);
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> rotation(
      rotation_values);
  return std::atan2(rotation(0, 2), rotation(2, 2));
}

std::vector<MatchedLandmark> FilterLandmarksByPose(
    const std::vector<MatchedLandmark>& matches,
    double yaw,
    double yaw_threshold,
    int* rejected_count) {
  *rejected_count = 0;
  std::vector<MatchedLandmark> filtered;
  filtered.reserve(matches.size());
  for (const auto& match : matches) {
    if (!IsFarSideLandmark(match.name, yaw, yaw_threshold)) {
      filtered.push_back(match);
    }
  }
  if (filtered.size() < 6) {
    return matches;
  }
  *rejected_count = static_cast<int>(matches.size() - filtered.size());
  return filtered;
}

}  // namespace

bool IsFarSideLandmark(const std::string& name, double yaw,
                       double yaw_threshold) {
  if (std::abs(yaw) < yaw_threshold) {
    return false;
  }
  const std::string far_side = yaw > 0.0 ? "right." : "left.";
  return name.rfind(far_side, 0) == 0;
}

double FarSideEyeConfidence(bool left_eye, double yaw, double soft_yaw,
                            double full_yaw) {
  const bool is_far_side = left_eye ? yaw < 0.0 : yaw > 0.0;
  if (!is_far_side) return 1.0;
  const double magnitude = std::abs(yaw);
  if (full_yaw <= soft_yaw) return magnitude < soft_yaw ? 1.0 : 0.0;
  return std::clamp((full_yaw - magnitude) / (full_yaw - soft_yaw), 0.0, 1.0);
}

Eigen::VectorXd GenerateVertices(const BfmModel& model,
                                 const Eigen::VectorXd& shape,
                                 const Eigen::VectorXd& expression) {
  Eigen::VectorXd scaled_shape = shape;
  Eigen::VectorXd scaled_expression = expression;
  for (int index = 0; index < scaled_shape.size(); ++index) {
    scaled_shape[index] *=
        std::sqrt(std::max(model.shape().pca_variance[index], 1.0e-12));
  }
  for (int index = 0; index < scaled_expression.size(); ++index) {
    scaled_expression[index] *=
        std::sqrt(std::max(model.expression().pca_variance[index], 1.0e-12));
  }
  return model.shape().mean + model.expression().mean +
         model.shape().pca_basis.leftCols(shape.size()) * scaled_shape +
         model.expression().pca_basis.leftCols(expression.size()) *
             scaled_expression;
}

FittingResult FitBfmToLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    const FittingOptions& options) {
  const int num_shape = std::clamp(
      options.num_shape_coefficients, 1,
      static_cast<int>(std::min(model.shape().pca_basis.cols(),
                                model.shape().pca_variance.size())));
  const int num_expression = std::clamp(
      options.num_expression_coefficients, 1,
      static_cast<int>(std::min(model.expression().pca_basis.cols(),
                                model.expression().pca_variance.size())));

  std::vector<MatchedLandmark> semantic_matches =
      BuildMatchedLandmarks(model, landmarks, correspondences, num_shape,
                            num_expression, options.landmark_weight);
  FittingResult result;
  semantic_matches = FilterLandmarksByFaceMask(
      semantic_matches, options.landmark_mask_path,
      options.landmark_mask_tolerance,
      &result.mask_rejected_landmark_count,
      &result.mask_corrected_landmark_count);
  const double aspect_ratio = InferImageAspectRatio(landmarks);
  std::array<double, 7> camera =
      InitializeCamera(semantic_matches, aspect_ratio);
  double maximum_mean_z = std::numeric_limits<double>::lowest();
  for (int vertex = 0; vertex < model.shape().mean.size() / 3; ++vertex) {
    maximum_mean_z =
        std::max(maximum_mean_z,
                 model.shape().mean[3 * vertex + 2] +
                     model.expression().mean[3 * vertex + 2]);
  }
  const double minimum_translation_z = maximum_mean_z + 10.0;
  Eigen::VectorXd shape = Eigen::VectorXd::Zero(num_shape);
  Eigen::VectorXd expression = Eigen::VectorXd::Zero(num_expression);

  result.initial_rmse =
      ComputeRmse(semantic_matches, camera, aspect_ratio, shape, expression);

  ceres::Problem camera_problem;
  AddLandmarkResiduals(camera_problem, semantic_matches, options, camera.data(),
                       shape.data(), expression.data(), num_shape,
                       num_expression, aspect_ratio);
  ConfigureCameraBounds(camera_problem, camera.data(),
                        minimum_translation_z);
  AddFocalRegularization(camera_problem, camera.data(),
                         options.focal_regularization);
  camera_problem.SetParameterBlockConstant(shape.data());
  camera_problem.SetParameterBlockConstant(expression.data());
  const ceres::Solver::Summary camera_summary =
      SolveProblem(camera_problem, options.camera_iterations, options.verbose);

  result.estimated_yaw = CameraYaw(camera);
  if (options.filter_landmarks_by_pose) {
    semantic_matches = FilterLandmarksByPose(
        semantic_matches, result.estimated_yaw, options.pose_yaw_threshold,
        &result.pose_rejected_landmark_count);
  }

  if (options.filter_occluded_landmarks) {
    semantic_matches = FilterOccludedLandmarks(
        model, landmarks, semantic_matches, camera, aspect_ratio, shape,
        expression, options.landmark_visibility_depth_tolerance,
        &result.occluded_landmark_count,
        &result.visibility_filter_applied);
  }

  semantic_matches =
      FilterOutliers(semantic_matches, camera, aspect_ratio, shape,
                     expression, options.outlier_threshold,
                     &result.rejected_landmark_count);
  ApplyEyelidWeights(&semantic_matches, options.eye_weight_multiplier);

  // Stage 2 jointly fits camera, identity, and expression using semantic
  // anchors only. The image silhouette is handled separately with dynamic,
  // view-dependent correspondences after this stable initialization.
  ceres::Problem joint_problem;
  AddLandmarkResiduals(joint_problem, semantic_matches, options, camera.data(),
                       shape.data(), expression.data(), num_shape,
                       num_expression, aspect_ratio);
  result.eyelid_gap_pair_count = AddEyelidGapResiduals(
      joint_problem, semantic_matches, camera.data(), shape.data(),
      expression.data(), num_shape, num_expression, aspect_ratio,
      options.eyelid_gap_weight, options.huber_delta);
  ConfigureCameraBounds(joint_problem, camera.data(),
                        minimum_translation_z);
  AddRegularization(joint_problem, shape.data(), num_shape,
                    options.shape_regularization);
  AddRegularization(joint_problem, expression.data(), num_expression,
                    options.expression_regularization);
  AddFocalRegularization(joint_problem, camera.data(),
                         options.focal_regularization);
  const ceres::Solver::Summary joint_summary =
      SolveProblem(joint_problem, options.joint_iterations, options.verbose);

  result.final_rmse =
      ComputeRmse(semantic_matches, camera, aspect_ratio, shape, expression);
  result.usable = camera_summary.IsSolutionUsable() &&
                  joint_summary.IsSolutionUsable();
  result.camera.angle_axis =
      Eigen::Vector3d(camera[0], camera[1], camera[2]);
  result.camera.translation =
      Eigen::Vector3d(camera[3], camera[4], std::exp(camera[5]));
  result.camera.focal_length = std::exp(camera[6]);
  result.camera.aspect_ratio = aspect_ratio;
  result.shape_coefficients = shape;
  result.expression_coefficients = expression;
  result.vertices = GenerateVertices(model, shape, expression);
  result.semantic_landmark_count = static_cast<int>(semantic_matches.size());
  result.contour_landmark_count = 0;
  result.solver_summary =
      "Mask plausibility: " +
      std::to_string(result.mask_rejected_landmark_count) +
      " landmark(s) outside the face mask removed, " +
      std::to_string(result.mask_corrected_landmark_count) +
      " boundary landmark(s) snapped to the mask" +
      "\nCamera stage: " + camera_summary.BriefReport() +
      "\nPose filtering: yaw=" + std::to_string(result.estimated_yaw) +
      ", " + std::to_string(result.pose_rejected_landmark_count) +
      " far-side landmark(s) removed" +
      "\nLandmark visibility: " +
      (result.visibility_filter_applied
           ? (std::to_string(result.occluded_landmark_count) +
              " self-occluded landmark(s) removed")
           : "filter not applied (disabled, no topology, or fewer than 6 visible landmarks)") +
      "\nEyelid gaps: " + std::to_string(result.eyelid_gap_pair_count) +
      " pair(s)" +
      "\nJoint stage: " + joint_summary.BriefReport();

  for (const auto& match : semantic_matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    LandmarkReprojection reprojection;
    reprojection.name = match.name;
    reprojection.mediapipe_index = match.mediapipe_index;
    reprojection.bfm_vertex_id = match.bfm_vertex_id;
    reprojection.observed = match.target;
    reprojection.projected =
        ProjectPoint(point, camera, aspect_ratio);
    reprojection.weight_multiplier =
        match.weight / std::max(options.landmark_weight, 1.0e-12);
    result.reprojections.push_back(std::move(reprojection));
  }
  return result;
}

FittingResult TrackBfmFrame(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    const FittingResult& identity,
    const FittingResult& previous,
    const FittingResult* previous_previous,
    const TrackingOptions& options) {
  const int num_shape = static_cast<int>(identity.shape_coefficients.size());
  const int num_expression =
      static_cast<int>(identity.expression_coefficients.size());
  const double aspect_ratio = identity.camera.aspect_ratio;

  FittingResult result;
  std::vector<MatchedLandmark> matches =
      BuildMatchedLandmarks(model, landmarks, correspondences, num_shape,
                            num_expression, options.landmark_weight);
  if (options.mouth_weight_multiplier > 1.0) {
    for (auto& match : matches) {
      if (match.name.find("lips") != std::string::npos ||
          match.name.find("chin") != std::string::npos) {
        match.weight *= options.mouth_weight_multiplier;
      }
    }
  }
  ApplyEyelidWeights(&matches, options.eye_weight_multiplier);
  // Region support denominators are counted before any filtering, so a
  // region whose landmarks get rejected loses confidence instead of silently
  // keeping a high score over the few survivors.
  const int total_landmark_count = static_cast<int>(matches.size());
  const int total_left_eye_count =
      CountRegion(matches, LandmarkRegion::kLeftEye);
  const int total_right_eye_count =
      CountRegion(matches, LandmarkRegion::kRightEye);
  const int total_mouth_count = CountRegion(matches, LandmarkRegion::kMouth);

  // Warm-start pose and expression from the previous frame; identity and
  // focal length stay frozen for the entire sequence.
  std::array<double, 7> camera = CameraArray(previous.camera);
  camera[6] = std::log(identity.camera.focal_length);
  Eigen::VectorXd shape = identity.shape_coefficients;
  Eigen::VectorXd expression = previous.expression_coefficients;
  if (expression.size() != num_expression) {
    expression = Eigen::VectorXd::Zero(num_expression);
    const int shared = std::min<int>(
        num_expression, previous.expression_coefficients.size());
    expression.head(shared) =
        previous.expression_coefficients.head(shared);
  }

  result.initial_rmse =
      ComputeRmse(matches, camera, aspect_ratio, shape, expression);
  result.estimated_yaw = CameraYaw(camera);
  // Before the binary far-side cut at 0.55 engages, fade the far-side eye's
  // landmark and gap evidence in continuously so growing yaw hands the eye
  // over to the temporal prior instead of flipping its weight at one frame.
  double left_eye_weight_scale = 1.0;
  double right_eye_weight_scale = 1.0;
  if (options.region_confidence_control) {
    left_eye_weight_scale = FarSideEyeConfidence(
        true, result.estimated_yaw, options.eye_confidence_soft_yaw,
        options.eye_confidence_full_yaw);
    right_eye_weight_scale = FarSideEyeConfidence(
        false, result.estimated_yaw, options.eye_confidence_soft_yaw,
        options.eye_confidence_full_yaw);
    for (auto& match : matches) {
      const LandmarkRegion region = RegionOf(match.name);
      if (region == LandmarkRegion::kLeftEye) {
        match.weight *= left_eye_weight_scale;
      } else if (region == LandmarkRegion::kRightEye) {
        match.weight *= right_eye_weight_scale;
      }
    }
  }
  matches = FilterLandmarksByPose(matches, result.estimated_yaw, 0.55,
                                  &result.pose_rejected_landmark_count);
  if (options.filter_occluded_landmarks) {
    matches = FilterOccludedLandmarks(
        model, landmarks, matches, camera, aspect_ratio, shape, expression,
        options.landmark_visibility_depth_tolerance,
        &result.occluded_landmark_count, &result.visibility_filter_applied);
  }
  matches = FilterOutliers(matches, camera, aspect_ratio, shape, expression,
                           options.outlier_threshold,
                           &result.rejected_landmark_count);

  double maximum_mean_z = std::numeric_limits<double>::lowest();
  for (int vertex = 0; vertex < model.shape().mean.size() / 3; ++vertex) {
    maximum_mean_z =
        std::max(maximum_mean_z,
                 model.shape().mean[3 * vertex + 2] +
                     model.expression().mean[3 * vertex + 2]);
  }

  FittingOptions residual_options;
  residual_options.huber_delta = options.huber_delta;
  ceres::Problem problem;
  AddLandmarkResiduals(problem, matches, residual_options, camera.data(),
                       shape.data(), expression.data(), num_shape,
                       num_expression, aspect_ratio);
  result.eyelid_gap_pair_count = AddEyelidGapResiduals(
      problem, matches, camera.data(), shape.data(), expression.data(),
      num_shape, num_expression, aspect_ratio, options.eyelid_gap_weight,
      options.huber_delta, left_eye_weight_scale, right_eye_weight_scale);
  ConfigureCameraBounds(problem, camera.data(), maximum_mean_z + 10.0);
  problem.SetParameterBlockConstant(shape.data());
  problem.SetManifold(camera.data(), new ceres::SubsetManifold(7, {6}));
  AddRegularization(problem, expression.data(), num_expression,
                    options.expression_regularization);

  // First-order temporal priors damp the solution toward the previous frame;
  // with two history frames the same prior instead pulls toward the
  // constant-velocity extrapolation, penalizing acceleration.
  const std::array<double, 7> previous_camera = CameraArray(previous.camera);
  AddReferencePrior(problem, camera.data(), 7,
                    Eigen::Map<const Eigen::VectorXd>(previous_camera.data(), 7),
                    options.pose_temporal_weight);
  AddReferencePrior(problem, expression.data(), num_expression,
                    previous.expression_coefficients,
                    options.expression_temporal_weight);
  if (previous_previous != nullptr &&
      previous_previous->expression_coefficients.size() ==
          previous.expression_coefficients.size() &&
      previous.expression_coefficients.size() == num_expression) {
    const std::array<double, 7> older_camera =
        CameraArray(previous_previous->camera);
    Eigen::VectorXd extrapolated_camera(7);
    for (int index = 0; index < 7; ++index) {
      extrapolated_camera[index] =
          2.0 * previous_camera[index] - older_camera[index];
    }
    AddReferencePrior(problem, camera.data(), 7, extrapolated_camera,
                      options.acceleration_weight);
    AddReferencePrior(problem, expression.data(), num_expression,
                      2.0 * previous.expression_coefficients -
                          previous_previous->expression_coefficients,
                      options.acceleration_weight);
  }

  const ceres::Solver::Summary summary =
      SolveProblem(problem, options.iterations, options.verbose);

  result.final_rmse =
      ComputeRmse(matches, camera, aspect_ratio, shape, expression);

  // Region confidences at the solution. Always computed and exported so the
  // downstream reenactment and completion stages can weigh the per-region
  // driving signal; the control flag only decides whether they also gate the
  // in-solve weights above and the usability verdict below.
  const double solved_yaw = CameraYaw(camera);
  result.confidences.left_eye =
      RegionSupportConfidence(matches, LandmarkRegion::kLeftEye,
                              total_left_eye_count, camera, aspect_ratio,
                              shape, expression, options) *
      FarSideEyeConfidence(true, solved_yaw, options.eye_confidence_soft_yaw,
                           options.eye_confidence_full_yaw);
  result.confidences.right_eye =
      RegionSupportConfidence(matches, LandmarkRegion::kRightEye,
                              total_right_eye_count, camera, aspect_ratio,
                              shape, expression, options) *
      FarSideEyeConfidence(false, solved_yaw, options.eye_confidence_soft_yaw,
                           options.eye_confidence_full_yaw);
  result.confidences.mouth = RegionSupportConfidence(
      matches, LandmarkRegion::kMouth, total_mouth_count, camera,
      aspect_ratio, shape, expression, options);
  const double pose_support =
      total_landmark_count > 0
          ? static_cast<double>(matches.size()) / total_landmark_count
          : 0.0;
  result.confidences.pose =
      pose_support * RmseConfidence(result.final_rmse,
                                    options.confidence_rmse_good,
                                    options.confidence_rmse_bad);

  result.usable = summary.IsSolutionUsable() &&
                  std::isfinite(result.final_rmse) &&
                  result.final_rmse <= options.failure_rmse &&
                  (!options.region_confidence_control ||
                   result.confidences.pose >= options.min_pose_confidence);
  result.camera = CameraParametersFromArray(camera, aspect_ratio);
  result.shape_coefficients = shape;
  result.expression_coefficients = expression;
  result.vertices = GenerateVertices(model, shape, expression);
  result.semantic_landmark_count = static_cast<int>(matches.size());
  result.solver_summary =
      "Eyelid gaps: " + std::to_string(result.eyelid_gap_pair_count) +
      " pair(s)\n" + summary.BriefReport();

  for (const auto& match : matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    LandmarkReprojection reprojection;
    reprojection.name = match.name;
    reprojection.mediapipe_index = match.mediapipe_index;
    reprojection.bfm_vertex_id = match.bfm_vertex_id;
    reprojection.observed = match.target;
    reprojection.projected = ProjectPoint(point, camera, aspect_ratio);
    reprojection.weight_multiplier =
        match.weight / std::max(options.landmark_weight, 1.0e-12);
    result.reprojections.push_back(std::move(reprojection));
  }
  return result;
}

Eigen::Vector2d ProjectVertex(const Eigen::Vector3d& vertex,
                              const CameraParameters& camera) {
  return ProjectPoint(vertex, CameraArray(camera), camera.aspect_ratio);
}

CameraParameters TransferRelativeCameraPose(
    const CameraParameters& target_reference,
    const CameraParameters& source_reference,
    const CameraParameters& source_current,
    double scale) {
  if (!std::isfinite(scale) || scale < 0.0) {
    throw std::invalid_argument("Pose transfer scale must be finite and non-negative");
  }
  const auto rotation = [](const Eigen::Vector3d& angle_axis)
      -> Eigen::Matrix3d {
    const double angle = angle_axis.norm();
    if (angle <= 1.0e-12) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(angle, angle_axis / angle).toRotationMatrix();
  };
  const Eigen::Matrix3d source_delta =
      rotation(source_current.angle_axis) *
      rotation(source_reference.angle_axis).transpose();
  const Eigen::AngleAxisd delta_angle_axis(source_delta);
  const Eigen::Matrix3d scaled_delta =
      Eigen::AngleAxisd(scale * delta_angle_axis.angle(),
                        delta_angle_axis.axis())
          .toRotationMatrix();
  const Eigen::Matrix3d target_rotation =
      scaled_delta * rotation(target_reference.angle_axis);
  const Eigen::AngleAxisd target_angle_axis(target_rotation);

  CameraParameters result = target_reference;
  result.angle_axis =
      target_angle_axis.axis() * target_angle_axis.angle();
  const double source_depth =
      std::max(source_reference.translation.z(), 1.0e-6);
  const double translation_scale =
      target_reference.translation.z() / source_depth;
  result.translation.x() +=
      scale * translation_scale *
      (source_current.translation.x() - source_reference.translation.x());
  result.translation.y() +=
      scale * translation_scale *
      (source_current.translation.y() - source_reference.translation.y());
  const double depth_ratio =
      std::max(source_current.translation.z(), 1.0e-6) / source_depth;
  result.translation.z() *= std::pow(depth_ratio, scale);
  return result;
}

Eigen::VectorXd ApplyCameraRotation(const Eigen::VectorXd& vertices,
                                    const CameraParameters& camera) {
  Eigen::VectorXd rotated(vertices.size());
  for (int vertex = 0; vertex < vertices.size() / 3; ++vertex) {
    ceres::AngleAxisRotatePoint(camera.angle_axis.data(),
                                vertices.data() + 3 * vertex,
                                rotated.data() + 3 * vertex);
  }
  return rotated;
}

}  // namespace face_recon
