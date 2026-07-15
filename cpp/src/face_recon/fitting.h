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
  // Additional joint-stage evidence for blinking. Individual eyelid points
  // localize the contour, while the gap residual directly observes the
  // upper-to-lower lid separation and is insensitive to image translation.
  double eye_weight_multiplier = 3.0;
  double eyelid_gap_weight = 12.0;
  double focal_regularization = 0.25;
  double huber_delta = 0.01;
  double outlier_threshold = 0.035;
  // After the camera-only initialization, rasterize the mean BFM and remove
  // semantic landmarks hidden by another part of the face from the joint fit.
  bool filter_occluded_landmarks = true;
  // Rasterizer stores inverse optical depth; this is an absolute tolerance in
  // that same space when comparing a landmark vertex with the Z-buffer.
  float landmark_visibility_depth_tolerance = 1.0e-4f;
  bool filter_landmarks_by_pose = true;
  double pose_yaw_threshold = 0.55;
  // Optional binary face-skin mask used to reject gross detections outside
  // the observed facial surface before camera initialization.
  std::string landmark_mask_path;
  double landmark_mask_tolerance = 0.015;
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
  double weight_multiplier = 1.0;
};

// Per-region tracking reliability in [0,1]. Each value combines the fraction
// of the region's landmark correspondences that survived visibility/outlier
// filtering with how well the surviving ones are explained at the solution;
// eye confidences additionally fall off continuously for the far-side eye at
// large yaw. 1.0 means fully trusted; regions with no observations get 0.
struct RegionConfidences {
  double left_eye = 1.0;
  double right_eye = 1.0;
  double mouth = 1.0;
  double pose = 1.0;
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
  int pose_rejected_landmark_count = 0;
  int mask_rejected_landmark_count = 0;
  int mask_corrected_landmark_count = 0;
  int rejected_landmark_count = 0;
  int eyelid_gap_pair_count = 0;
  bool visibility_filter_applied = false;
  RegionConfidences confidences;
  double estimated_yaw = 0.0;
  double initial_rmse = 0.0;
  double final_rmse = 0.0;
  std::string solver_summary;
};

// Options for tracking a known identity through a sequence. Unlike the full
// reconstruction, tracking keeps shape and focal length fixed and only
// updates rigid pose and expression from the previous frame.
struct TrackingOptions {
  int iterations = 80;
  double landmark_weight = 100.0;
  // Weaker than the single-image prior (0.08): only ~9 of the 40 semantic
  // landmarks observe the mouth, so at 0.08 the prior overpowers them and
  // talking expressions come out heavily damped (the mouth barely opens).
  double expression_regularization = 0.02;
  // Extra weight on lips/chin landmarks; during speech they carry nearly all
  // of the expression signal but are outnumbered by the static landmarks.
  double mouth_weight_multiplier = 6.0;
  double eye_weight_multiplier = 3.0;
  double eyelid_gap_weight = 12.0;
  double pose_temporal_weight = 2.0;
  double expression_temporal_weight = 1.0;
  double acceleration_weight = 0.25;
  // Apply the same Z-buffer visibility test used by single-image fitting to
  // every tracking frame before optimizing expression and pose.
  bool filter_occluded_landmarks = true;
  float landmark_visibility_depth_tolerance = 1.0e-4f;
  // Residuals are evaluated at the previous frame's warm start, so fast
  // motion (a mouth opening between frames) legitimately produces errors of
  // ~0.05 normalized units. The robust loss and the outlier gate must stay
  // clear of that range or the very landmarks carrying the expression change
  // get capped and rejected, freezing the expression at the previous frame.
  double huber_delta = 0.03;
  double outlier_threshold = 0.12;
  double failure_rmse = 0.06;
  // Region-aware uncertainty. When enabled, eyelid landmark and gap weights
  // of the far-side eye fade continuously between the two yaw angles instead
  // of only being cut binarily at the 0.55 pose filter, and a frame whose
  // pose confidence falls below the minimum is reported unusable so the
  // driver reinitializes instead of rendering an implausible state.
  bool region_confidence_control = true;
  double eye_confidence_soft_yaw = 0.35;
  double eye_confidence_full_yaw = 0.70;
  // Linear ramp mapping a region's landmark rmse (normalized image units) to
  // confidence: 1 at or below good, 0 at or beyond bad.
  double confidence_rmse_good = 0.02;
  double confidence_rmse_bad = 0.08;
  double min_pose_confidence = 0.2;
  bool verbose = false;
};

FittingResult TrackBfmFrame(
    const BfmModel& model,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const std::vector<face_reconstruction::BfmMediaPipeCorrespondence>& correspondences,
    const FittingResult& identity,
    const FittingResult& previous,
    const FittingResult* previous_previous = nullptr,
    const TrackingOptions& options = {});

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

// Applies a source trajectory's pose delta (relative to its first frame) to
// a target photograph's fitted baseline camera. Rotations are composed on
// SO(3); angle-axis vectors are never added component-wise.
CameraParameters TransferRelativeCameraPose(
    const CameraParameters& target_reference,
    const CameraParameters& source_reference,
    const CameraParameters& source_current,
    double scale = 1.0);
Eigen::VectorXd ApplyCameraRotation(const Eigen::VectorXd& vertices,
                                    const CameraParameters& camera);
bool IsFarSideLandmark(const std::string& name, double yaw,
                       double yaw_threshold);
// Continuous confidence in the given eye's landmarks under head yaw: 1 while
// the eye faces the camera, fading linearly to 0 as the far-side eye
// approaches self-occlusion. Positive yaw turns the right side away.
double FarSideEyeConfidence(bool left_eye, double yaw, double soft_yaw,
                            double full_yaw);

}  // namespace face_recon

#endif  // FACE_RECON_FITTING_H_
