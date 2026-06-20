#include "face_recon/fitting.h"

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
  explicit LandmarkResidual(MatchedLandmark landmark)
      : landmark_(std::move(landmark)),
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
    const T scale = exp(camera[3]);
    const T projected_x = scale * rotated[0] + camera[4];
    const T projected_y = -scale * rotated[1] + camera[5];
    residuals[0] =
        T(sqrt_weight_) * (projected_x - T(landmark_.target.x()));
    residuals[1] =
        T(sqrt_weight_) * (projected_y - T(landmark_.target.y()));
    return true;
  }

 private:
  MatchedLandmark landmark_;
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

Eigen::Vector2d ProjectPoint(const Eigen::Vector3d& point,
                             const std::array<double, 6>& camera) {
  double rotated[3];
  ceres::AngleAxisRotatePoint(camera.data(), point.data(), rotated);
  const double scale = std::exp(camera[3]);
  return Eigen::Vector2d(scale * rotated[0] + camera[4],
                         -scale * rotated[1] + camera[5]);
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

std::vector<int> BoundaryVertices(const Eigen::MatrixXi& triangles) {
  std::unordered_map<std::uint64_t, int> edge_counts;
  const bool rows_are_faces = triangles.cols() == 3;
  const int face_count = rows_are_faces ? triangles.rows() : triangles.cols();

  for (int face = 0; face < face_count; ++face) {
    int indices[3];
    for (int corner = 0; corner < 3; ++corner) {
      indices[corner] =
          rows_are_faces ? triangles(face, corner) : triangles(corner, face);
    }
    for (int edge = 0; edge < 3; ++edge) {
      const std::uint32_t first = static_cast<std::uint32_t>(
          std::min(indices[edge], indices[(edge + 1) % 3]));
      const std::uint32_t second = static_cast<std::uint32_t>(
          std::max(indices[edge], indices[(edge + 1) % 3]));
      const std::uint64_t key =
          (static_cast<std::uint64_t>(first) << 32U) | second;
      ++edge_counts[key];
    }
  }

  std::vector<int> boundary;
  for (const auto& [key, count] : edge_counts) {
    if (count == 1) {
      boundary.push_back(static_cast<int>(key >> 32U));
      boundary.push_back(static_cast<int>(key & 0xffffffffU));
    }
  }
  std::sort(boundary.begin(), boundary.end());
  boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());
  return boundary;
}

std::vector<MatchedLandmark> BuildContourLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::array<double, 6>& camera,
    const Eigen::VectorXd& shape,
    const Eigen::VectorXd& expression,
    int num_shape,
    int num_expression,
    double weight) {
  static constexpr std::array<int, 21> kJawIndices = {
      234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
      377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
  };

  std::unordered_map<int, const face_reconstruction::Landmark2D*> observations;
  for (const auto& landmark : landmarks) {
    observations[landmark.index] = &landmark;
  }

  const std::vector<int> boundary = BoundaryVertices(model.triangles());
  std::vector<Eigen::Vector2d> projected_boundary;
  projected_boundary.reserve(boundary.size());
  for (const int vertex_id : boundary) {
    const Eigen::Vector3d point =
        model.GetMeanVertex(vertex_id) +
        ScaledBasisForVertex(model.shape(), vertex_id, num_shape) * shape +
        ScaledBasisForVertex(model.expression(), vertex_id, num_expression) *
            expression;
    projected_boundary.push_back(ProjectPoint(point, camera));
  }

  std::vector<bool> used(boundary.size(), false);
  std::vector<MatchedLandmark> matches;
  for (const int mediapipe_index : kJawIndices) {
    const auto observation = observations.find(mediapipe_index);
    if (observation == observations.end()) {
      continue;
    }
    const Eigen::Vector2d target(observation->second->x_norm,
                                 observation->second->y_norm);

    int best = -1;
    double best_distance = std::numeric_limits<double>::max();
    for (int index = 0; index < static_cast<int>(boundary.size()); ++index) {
      if (used[index]) {
        continue;
      }
      const double distance = (projected_boundary[index] - target).squaredNorm();
      if (distance < best_distance) {
        best_distance = distance;
        best = index;
      }
    }
    if (best < 0 || std::sqrt(best_distance) > 0.08) {
      continue;
    }

    used[best] = true;
    MatchedLandmark match;
    match.name = "contour_" + std::to_string(mediapipe_index);
    match.mediapipe_index = mediapipe_index;
    match.bfm_vertex_id = boundary[best];
    match.target = target;
    match.mean = model.GetMeanVertex(match.bfm_vertex_id);
    match.shape_basis =
        ScaledBasisForVertex(model.shape(), match.bfm_vertex_id, num_shape);
    match.expression_basis =
        ScaledBasisForVertex(model.expression(), match.bfm_vertex_id, num_expression);
    match.weight = weight;
    matches.push_back(std::move(match));
  }
  return matches;
}

std::array<double, 6> InitializeCamera(
    const std::vector<MatchedLandmark>& matches) {
  Eigen::Vector2d target_min =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d target_max =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());
  Eigen::Vector2d model_min =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d model_max =
      Eigen::Vector2d::Constant(std::numeric_limits<double>::lowest());

  for (const auto& match : matches) {
    target_min = target_min.cwiseMin(match.target);
    target_max = target_max.cwiseMax(match.target);
    const Eigen::Vector2d model_xy(match.mean.x(), -match.mean.y());
    model_min = model_min.cwiseMin(model_xy);
    model_max = model_max.cwiseMax(model_xy);
  }

  const Eigen::Vector2d target_extent = target_max - target_min;
  const Eigen::Vector2d model_extent = model_max - model_min;
  const double scale_x =
      target_extent.x() / std::max(model_extent.x(), 1.0e-9);
  const double scale_y =
      target_extent.y() / std::max(model_extent.y(), 1.0e-9);
  const double scale = std::max(0.5 * (scale_x + scale_y), 1.0e-9);
  const Eigen::Vector2d target_center = 0.5 * (target_min + target_max);
  const Eigen::Vector2d model_center = 0.5 * (model_min + model_max);
  const Eigen::Vector2d translation = target_center - scale * model_center;
  return {0.0, 0.0, 0.0, std::log(scale), translation.x(), translation.y()};
}

double ComputeRmse(const std::vector<MatchedLandmark>& matches,
                   const std::array<double, 6>& camera,
                   const Eigen::VectorXd& shape,
                   const Eigen::VectorXd& expression) {
  double squared_error = 0.0;
  for (const auto& match : matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    squared_error += (ProjectPoint(point, camera) - match.target).squaredNorm();
  }
  return std::sqrt(squared_error / static_cast<double>(matches.size()));
}

std::vector<MatchedLandmark> FilterOutliers(
    const std::vector<MatchedLandmark>& matches,
    const std::array<double, 6>& camera,
    const Eigen::VectorXd& shape,
    const Eigen::VectorXd& expression,
    double absolute_threshold,
    int* rejected_count) {
  std::vector<double> errors;
  for (const auto& match : matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    errors.push_back((ProjectPoint(point, camera) - match.target).norm());
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
                          int num_expression) {
  for (const auto& match : matches) {
    auto* cost = new ceres::DynamicAutoDiffCostFunction<LandmarkResidual, 4>(
        new LandmarkResidual(match));
    cost->AddParameterBlock(6);
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

void ConfigureCameraBounds(ceres::Problem& problem, double* camera) {
  problem.SetParameterLowerBound(camera, 3, -20.0);
  problem.SetParameterUpperBound(camera, 3, 5.0);
  problem.SetParameterLowerBound(camera, 4, -2.0);
  problem.SetParameterUpperBound(camera, 4, 3.0);
  problem.SetParameterLowerBound(camera, 5, -2.0);
  problem.SetParameterUpperBound(camera, 5, 3.0);
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

std::array<double, 6> CameraArray(const CameraParameters& camera) {
  return {camera.angle_axis.x(), camera.angle_axis.y(), camera.angle_axis.z(),
          std::log(camera.scale), camera.translation_x, camera.translation_y};
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
  std::array<double, 6> camera = InitializeCamera(semantic_matches);
  Eigen::VectorXd shape = Eigen::VectorXd::Zero(num_shape);
  Eigen::VectorXd expression = Eigen::VectorXd::Zero(num_expression);

  FittingResult result;
  result.initial_rmse =
      ComputeRmse(semantic_matches, camera, shape, expression);

  ceres::Problem camera_problem;
  AddLandmarkResiduals(camera_problem, semantic_matches, options, camera.data(),
                       shape.data(), expression.data(), num_shape, num_expression);
  ConfigureCameraBounds(camera_problem, camera.data());
  camera_problem.SetParameterBlockConstant(shape.data());
  camera_problem.SetParameterBlockConstant(expression.data());
  const ceres::Solver::Summary camera_summary =
      SolveProblem(camera_problem, options.camera_iterations, options.verbose);

  semantic_matches =
      FilterOutliers(semantic_matches, camera, shape, expression,
                     options.outlier_threshold, &result.rejected_landmark_count);

  ceres::Problem shape_problem;
  AddLandmarkResiduals(shape_problem, semantic_matches, options, camera.data(),
                       shape.data(), expression.data(), num_shape, num_expression);
  ConfigureCameraBounds(shape_problem, camera.data());
  AddRegularization(shape_problem, shape.data(), num_shape,
                    options.shape_regularization);
  shape_problem.SetParameterBlockConstant(expression.data());
  const ceres::Solver::Summary shape_summary =
      SolveProblem(shape_problem, options.shape_iterations, options.verbose);

  ceres::Solver::Summary joint_summary;
  std::vector<MatchedLandmark> contour_matches;
  const int refinement_steps = std::max(options.contour_refinement_steps, 1);
  for (int refinement = 0; refinement < refinement_steps; ++refinement) {
    contour_matches =
        BuildContourLandmarks(model, landmarks, camera, shape, expression,
                              num_shape, num_expression, options.contour_weight);
    std::vector<MatchedLandmark> all_matches = semantic_matches;
    all_matches.insert(all_matches.end(), contour_matches.begin(),
                       contour_matches.end());

    ceres::Problem joint_problem;
    AddLandmarkResiduals(joint_problem, all_matches, options, camera.data(),
                         shape.data(), expression.data(), num_shape,
                         num_expression);
    ConfigureCameraBounds(joint_problem, camera.data());
    AddRegularization(joint_problem, shape.data(), num_shape,
                      options.shape_regularization);
    AddRegularization(joint_problem, expression.data(), num_expression,
                      options.expression_regularization);
    joint_summary =
        SolveProblem(joint_problem, options.joint_iterations, options.verbose);
  }

  result.final_rmse =
      ComputeRmse(semantic_matches, camera, shape, expression);
  result.usable = camera_summary.IsSolutionUsable() &&
                  shape_summary.IsSolutionUsable() &&
                  joint_summary.IsSolutionUsable();
  result.camera.angle_axis =
      Eigen::Vector3d(camera[0], camera[1], camera[2]);
  result.camera.scale = std::exp(camera[3]);
  result.camera.translation_x = camera[4];
  result.camera.translation_y = camera[5];
  result.shape_coefficients = shape;
  result.expression_coefficients = expression;
  result.vertices = GenerateVertices(model, shape, expression);
  result.semantic_landmark_count = static_cast<int>(semantic_matches.size());
  result.contour_landmark_count = static_cast<int>(contour_matches.size());
  result.solver_summary =
      "Camera stage: " + camera_summary.BriefReport() +
      "\nIdentity stage: " + shape_summary.BriefReport() +
      "\nJoint stage: " + joint_summary.BriefReport();

  std::vector<MatchedLandmark> final_matches = semantic_matches;
  final_matches.insert(final_matches.end(), contour_matches.begin(),
                       contour_matches.end());
  for (const auto& match : final_matches) {
    const Eigen::Vector3d point =
        match.mean + match.shape_basis * shape +
        match.expression_basis * expression;
    LandmarkReprojection reprojection;
    reprojection.name = match.name;
    reprojection.mediapipe_index = match.mediapipe_index;
    reprojection.bfm_vertex_id = match.bfm_vertex_id;
    reprojection.observed = match.target;
    reprojection.projected = ProjectPoint(point, camera);
    result.reprojections.push_back(std::move(reprojection));
  }
  return result;
}

Eigen::Vector2d ProjectVertex(const Eigen::Vector3d& vertex,
                              const CameraParameters& camera) {
  return ProjectPoint(vertex, CameraArray(camera));
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
