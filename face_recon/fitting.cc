#include "face_recon/fitting.h"

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

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

}  // namespace

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

  FittingResult result;
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

  semantic_matches =
      FilterOutliers(semantic_matches, camera, aspect_ratio, shape,
                     expression, options.outlier_threshold,
                     &result.rejected_landmark_count);

  // Stage 2 jointly fits camera, identity, and expression using semantic
  // anchors only. The image silhouette is handled separately with dynamic,
  // view-dependent correspondences after this stable initialization.
  ceres::Problem joint_problem;
  AddLandmarkResiduals(joint_problem, semantic_matches, options, camera.data(),
                       shape.data(), expression.data(), num_shape,
                       num_expression, aspect_ratio);
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
      "Camera stage: " + camera_summary.BriefReport() +
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
    result.reprojections.push_back(std::move(reprojection));
  }
  return result;
}

Eigen::Vector2d ProjectVertex(const Eigen::Vector3d& vertex,
                              const CameraParameters& camera) {
  return ProjectPoint(vertex, CameraArray(camera), camera.aspect_ratio);
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
