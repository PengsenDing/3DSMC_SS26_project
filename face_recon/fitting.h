#ifndef FACE_RECON_FITTING_H_
#define FACE_RECON_FITTING_H_

#include "face_recon/bfm_model.h"
#include "face_reconstruction/landmarks.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace face_recon {

struct FittingOptions {
  int num_shape_coefficients = 80;
  int num_expression_coefficients = 30;
  int camera_iterations = 100;
  int shape_iterations = 150;
  int joint_iterations = 200;
  double landmark_weight = 100.0;
  double contour_weight = 35.0;
  double shape_regularization = 0.01;
  double expression_regularization = 0.01;
  double huber_delta = 0.01;
  double outlier_threshold = 0.035;
  int contour_refinement_steps = 2;
  bool verbose = false;
};

struct CameraParameters {
  Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
  double scale = 1.0;
  double translation_x = 0.5;
  double translation_y = 0.5;
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
  int contour_landmark_count = 0;
  int rejected_landmark_count = 0;
  double initial_rmse = 0.0;
  double final_rmse = 0.0;
  std::string solver_summary;
};

FittingResult FitBfmToLandmarks(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    const FittingOptions& options = {});

Eigen::Vector2d ProjectVertex(const Eigen::Vector3d& vertex,
                              const CameraParameters& camera);
Eigen::VectorXd ApplyCameraRotation(const Eigen::VectorXd& vertices,
                                    const CameraParameters& camera);

}  // namespace face_recon

#endif  // FACE_RECON_FITTING_H_
