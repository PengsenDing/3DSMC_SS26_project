#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_reconstruction/landmarks.hpp"

#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

Eigen::Matrix3d Rotation(const Eigen::Vector3d& angle_axis) {
  const double angle = angle_axis.norm();
  if (angle <= 1.0e-12) return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, angle_axis / angle).toRotationMatrix();
}

Eigen::Vector2d Project(const Eigen::Vector3d& point,
                        const Eigen::Vector3d& angle_axis,
                        const Eigen::Vector3d& translation,
                        double focal_length,
                        double aspect_ratio) {
  double rotated[3];
  ceres::AngleAxisRotatePoint(angle_axis.data(), point.data(), rotated);
  const double depth = translation.z() - rotated[2];
  return Eigen::Vector2d(
      0.5 + focal_length * (rotated[0] + translation.x()) / depth,
      0.5 - focal_length * aspect_ratio *
                (rotated[1] + translation.y()) / depth);
}

}  // namespace

int main() {
  if (!face_recon::IsFarSideLandmark("right.eye.corner_outer", 0.8, 0.55) ||
      face_recon::IsFarSideLandmark("left.eye.corner_outer", 0.8, 0.55) ||
      !face_recon::IsFarSideLandmark("left.lips.corner", -0.8, 0.55) ||
      face_recon::IsFarSideLandmark("center.nose.tip", 0.8, 0.55) ||
      face_recon::IsFarSideLandmark("right.eye.corner_outer", 0.3, 0.55)) {
    std::cerr << "Pose-aware landmark classification failed\n";
    return 1;
  }

  constexpr double kSoftYaw = 0.35;
  constexpr double kFullYaw = 0.70;
  if (face_recon::FarSideEyeConfidence(true, 0.5, kSoftYaw, kFullYaw) != 1.0 ||
      face_recon::FarSideEyeConfidence(false, -0.5, kSoftYaw, kFullYaw) !=
          1.0 ||
      face_recon::FarSideEyeConfidence(false, 0.2, kSoftYaw, kFullYaw) !=
          1.0 ||
      std::abs(face_recon::FarSideEyeConfidence(false, 0.525, kSoftYaw,
                                                kFullYaw) -
               0.5) > 1.0e-12 ||
      std::abs(face_recon::FarSideEyeConfidence(true, -0.525, kSoftYaw,
                                                kFullYaw) -
               0.5) > 1.0e-12 ||
      face_recon::FarSideEyeConfidence(true, -0.9, kSoftYaw, kFullYaw) !=
          0.0) {
    std::cerr << "Far-side eye confidence ramp failed\n";
    return 1;
  }

  face_recon::CameraParameters target_camera;
  target_camera.angle_axis = Eigen::Vector3d(0.1, 0.0, 0.0);
  target_camera.translation = Eigen::Vector3d(10.0, 20.0, 200.0);
  target_camera.focal_length = 1.4;
  target_camera.aspect_ratio = 0.75;
  face_recon::CameraParameters source_reference;
  source_reference.translation = Eigen::Vector3d(1.0, 2.0, 100.0);
  face_recon::CameraParameters source_current = source_reference;
  source_current.angle_axis = Eigen::Vector3d(0.0, 0.0, 0.2);
  source_current.translation = Eigen::Vector3d(3.0, 1.0, 110.0);
  const face_recon::CameraParameters transferred =
      face_recon::TransferRelativeCameraPose(
          target_camera, source_reference, source_current);
  const Eigen::Matrix3d expected_rotation =
      Rotation(source_current.angle_axis) * Rotation(target_camera.angle_axis);
  if (!Rotation(transferred.angle_axis).isApprox(expected_rotation, 1.0e-10) ||
      !transferred.translation.isApprox(Eigen::Vector3d(14.0, 18.0, 220.0),
                                        1.0e-10) ||
      transferred.focal_length != target_camera.focal_length ||
      transferred.aspect_ratio != target_camera.aspect_ratio) {
    std::cerr << "Relative camera-pose transfer failed\n";
    return 1;
  }
  const face_recon::CameraParameters zero_scale =
      face_recon::TransferRelativeCameraPose(
          target_camera, source_reference, source_current, 0.0);
  if (!Rotation(zero_scale.angle_axis)
           .isApprox(Rotation(target_camera.angle_axis), 1.0e-10) ||
      !zero_scale.translation.isApprox(target_camera.translation, 1.0e-10)) {
    std::cerr << "Zero pose scale did not preserve the target camera\n";
    return 1;
  }

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
  const Eigen::Vector3d true_translation(0.12, -0.08, 15.0);
  constexpr double kTrueFocalLength = 1.2;
  constexpr double kAspectRatio = 4.0 / 3.0;

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
        Project(point, true_rotation, true_translation,
                kTrueFocalLength, kAspectRatio);

    face_reconstruction::Landmark2D landmark;
    landmark.index = vertex;
    landmark.x_norm = static_cast<float>(target.x());
    landmark.y_norm = static_cast<float>(target.y());
    landmark.u = static_cast<float>(target.x() * 800.0);
    landmark.v = static_cast<float>(target.y() * 600.0);
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
      result.final_rmse >= result.initial_rmse ||
      result.contour_landmark_count != 0 ||
      result.visibility_filter_applied ||
      result.occluded_landmark_count != 0 ||
      result.pose_rejected_landmark_count != 0 ||
      result.mask_rejected_landmark_count != 0 ||
      result.mask_corrected_landmark_count != 0 ||
      result.reprojections.size() != correspondences.size()) {
    std::cerr << result.solver_summary << "\n";
    return 1;
  }

  // Tracking the same landmarks from the fitted identity must stay usable
  // with high pose confidence, while regions without any correspondences
  // (the synthetic names carry no eye/mouth landmarks) report zero.
  face_recon::TrackingOptions tracking;
  tracking.filter_occluded_landmarks = false;  // No triangles to rasterize.
  const face_recon::FittingResult tracked = face_recon::TrackBfmFrame(
      model, landmarks, correspondences, result, result, nullptr, tracking);
  if (!tracked.usable || tracked.confidences.pose < 0.9 ||
      tracked.confidences.left_eye != 0.0 ||
      tracked.confidences.right_eye != 0.0 ||
      tracked.confidences.mouth != 0.0) {
    std::cerr << "Region confidence tracking failed: pose="
              << tracked.confidences.pose
              << " left_eye=" << tracked.confidences.left_eye
              << " right_eye=" << tracked.confidences.right_eye
              << " mouth=" << tracked.confidences.mouth << "\n"
              << tracked.solver_summary << "\n";
    return 1;
  }
  return 0;
}
