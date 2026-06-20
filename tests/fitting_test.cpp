#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_reconstruction/landmarks.hpp"

#include <ceres/rotation.h>

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

Eigen::Vector2d Project(const Eigen::Vector3d& point,
                        const Eigen::Vector3d& angle_axis,
                        double scale,
                        double translation_x,
                        double translation_y) {
  double rotated[3];
  ceres::AngleAxisRotatePoint(angle_axis.data(), point.data(), rotated);
  return Eigen::Vector2d(scale * rotated[0] + translation_x,
                         -scale * rotated[1] + translation_y);
}

}  // namespace

int main() {
  constexpr int kVertexCount = 8;
  face_recon::BfmModel model;

  face_recon::PcaComponent shape;
  shape.mean.resize(3 * kVertexCount);
  shape.mean <<
      -1.0, 1.0, 0.0,
       1.0, 1.0, 0.0,
      -1.1, 0.0, 0.1,
       1.1, 0.0, 0.1,
      -0.8, -1.0, 0.0,
       0.8, -1.0, 0.0,
       0.0, 0.3, 0.8,
       0.0, -0.6, 0.3;
  shape.pca_basis = Eigen::MatrixXd::Zero(3 * kVertexCount, 2);
  for (int vertex = 0; vertex < kVertexCount; ++vertex) {
    shape.pca_basis(3 * vertex, 0) =
        shape.mean[3 * vertex] < 0.0 ? -0.2 :
        shape.mean[3 * vertex] > 0.0 ? 0.2 : 0.0;
    shape.pca_basis(3 * vertex + 1, 1) =
        shape.mean[3 * vertex + 1] * 0.15;
  }
  shape.pca_variance = Eigen::Vector2d(4.0, 9.0);

  face_recon::PcaComponent expression;
  expression.mean = Eigen::VectorXd::Zero(3 * kVertexCount);
  expression.pca_basis = Eigen::MatrixXd::Zero(3 * kVertexCount, 1);
  expression.pca_basis(3 * 4 + 1, 0) = -0.15;
  expression.pca_basis(3 * 5 + 1, 0) = -0.15;
  expression.pca_basis(3 * 7 + 1, 0) = -0.25;
  expression.pca_variance = Eigen::VectorXd::Constant(1, 2.25);

  model.set_shape(shape);
  model.set_expression(expression);

  const Eigen::Vector2d true_shape(0.35, -0.2);
  Eigen::VectorXd true_expression(1);
  true_expression << 0.4;
  const Eigen::Vector3d true_rotation(0.04, -0.08, 0.03);
  constexpr double kTrueScale = 0.18;
  constexpr double kTrueTranslationX = 0.51;
  constexpr double kTrueTranslationY = 0.49;

  std::vector<face_reconstruction::Landmark2D> landmarks;
  std::vector<face_reconstruction::BfmMediaPipeCorrespondence> correspondences;
  for (int vertex = 0; vertex < kVertexCount; ++vertex) {
    const Eigen::Vector3d point =
        shape.mean.segment<3>(3 * vertex) +
        shape.pca_basis.block(3 * vertex, 0, 3, 2) *
            shape.pca_variance.cwiseSqrt().asDiagonal() * true_shape +
        expression.pca_basis.block(3 * vertex, 0, 3, 1) *
            expression.pca_variance.cwiseSqrt().asDiagonal() *
            true_expression;
    const Eigen::Vector2d target =
        Project(point, true_rotation, kTrueScale,
                kTrueTranslationX, kTrueTranslationY);

    face_reconstruction::Landmark2D landmark;
    landmark.index = vertex;
    landmark.x_norm = static_cast<float>(target.x());
    landmark.y_norm = static_cast<float>(target.y());
    landmarks.push_back(landmark);

    face_reconstruction::BfmMediaPipeCorrespondence correspondence;
    correspondence.bfm_landmark_name = "synthetic_" + std::to_string(vertex);
    correspondence.bfm_vertex_id = vertex;
    correspondence.mediapipe_index = vertex;
    correspondences.push_back(correspondence);
  }

  face_recon::FittingOptions options;
  options.num_shape_coefficients = 2;
  options.num_expression_coefficients = 1;
  options.landmark_weight = 1000.0;
  options.shape_regularization = 1.0e-8;
  options.expression_regularization = 1.0e-8;
  options.huber_delta = 0.0;

  const face_recon::FittingResult result =
      face_recon::FitBfmToLandmarks(model, landmarks, correspondences, options);
  std::cout << "Synthetic fitting RMSE: " << result.initial_rmse
            << " -> " << result.final_rmse << "\n";

  if (!result.usable || result.final_rmse >= 1.0e-4 ||
      result.final_rmse >= result.initial_rmse) {
    std::cerr << result.solver_summary << "\n";
    return 1;
  }
  return 0;
}
