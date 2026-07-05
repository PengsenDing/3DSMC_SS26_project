#ifndef FACE_RECON_FITTING_H_
#define FACE_RECON_FITTING_H_

#include "face_recon/bfm_model.h"
#include "face_reconstruction/landmarks.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace face_recon {

struct FittingOptions {
  // Defaults match the Basel Face Model's full PCA dimensionality (199
  // shape, 100 expression components for model2019_face12.h5).
  int num_shape_coefficients = 199;
  int num_expression_coefficients = 100;
  int camera_iterations = 100;
  int joint_iterations = 200;
  double landmark_weight = 100.0;
  // Raised proportionally to the increase in free PCA coefficients above,
  // so the optimizer does not use the newly available coefficients to
  // overfit a comparatively small set of 2D landmark observations.
  double shape_regularization = 0.1;
  double expression_regularization = 0.08;
  double focal_regularization = 0.25;
  double huber_delta = 0.01;
  double outlier_threshold = 0.035;
  // After the camera-only initialization, rasterize the mean BFM and remove
  // semantic landmarks hidden by another part of the face from the joint fit.
  bool filter_occluded_landmarks = true;
  // Rasterizer stores inverse optical depth; this is an absolute tolerance in
  // that same space when comparing a landmark vertex with the Z-buffer.
  float landmark_visibility_depth_tolerance = 1.0e-4f;
  bool verbose = false;
};

struct CameraParameters {
  Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
  Eigen::Vector3d translation = Eigen::Vector3d(0.0, 0.0, 400.0);
  // Horizontal focal length normalized by image width.
  double focal_length = 1.2;
  double aspect_ratio = 1.0;
};

struct LandmarkReprojection {
  std::string name;
  int mediapipe_index = 0;
  int bfm_vertex_id = 0;
  Eigen::Vector2d observed = Eigen::Vector2d::Zero();
  Eigen::Vector2d projected = Eigen::Vector2d::Zero();
};

struct FittingResult {
  bool usable = false;
  CameraParameters camera;
  Eigen::VectorXd shape_coefficients;
  Eigen::VectorXd expression_coefficients;
  Eigen::VectorXd vertices;
  std::vector<LandmarkReprojection> reprojections;
  int semantic_landmark_count = 0;
  int contour_landmark_count = 0;  // Always zero: contours use silhouette fitting.
  int occluded_landmark_count = 0;
  int rejected_landmark_count = 0;
  bool visibility_filter_applied = false;
  double initial_rmse = 0.0;
  double final_rmse = 0.0;
  std::string solver_summary;
};

FittingResult FitBfmToLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    const FittingOptions& options = {});

Eigen::VectorXd GenerateVertices(const BfmModel& model,
                                 const Eigen::VectorXd& shape,
                                 const Eigen::VectorXd& expression);
Eigen::Vector2d ProjectVertex(const Eigen::Vector3d& vertex,
                              const CameraParameters& camera);
Eigen::VectorXd ApplyCameraRotation(const Eigen::VectorXd& vertices,
                                    const CameraParameters& camera);

}  // namespace face_recon

#endif  // FACE_RECON_FITTING_H_
