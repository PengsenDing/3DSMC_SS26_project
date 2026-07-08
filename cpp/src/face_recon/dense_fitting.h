#ifndef FACE_RECON_DENSE_FITTING_H_
#define FACE_RECON_DENSE_FITTING_H_

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/photometric_fitting.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

struct DenseFittingOptions {
  bool enabled = true;
  // Expression coefficients driven by the dense term. Higher-order BFM
  // components displace vertices by less than image noise; they keep their
  // landmark-fit values.
  int num_expression_coefficients = 40;
  int pixel_stride = 4;
  int mask_erosion = 2;
  // Binary mask of external occluders (hair, hands, clothes, accessories);
  // pixels under it never contribute color observations. Empty disables it.
  std::string occluder_mask_path;
  // Visibility and shading are frozen per outer iteration; each iteration
  // re-rasterizes with the current parameters and re-linearizes.
  int outer_iterations = 3;
  // The first outer iterations match against a Gaussian-blurred photograph
  // to widen the photometric basin of convergence (coarse-to-fine).
  int blurred_iterations = 1;
  double blur_sigma = 2.0;
  int solver_iterations = 12;
  // Semantic landmarks act as anchors inside the dense problem so the
  // photometric term cannot drag the face off the detected features.
  double landmark_weight = 4000.0;
  double expression_regularization = 0.5;
  double huber_delta = 0.05;
  bool optimize_pose = true;
  bool save_diagnostics = false;
  bool verbose = false;
};

struct DenseFittingResult {
  bool usable = false;
  int sample_count = 0;
  double initial_photometric_rmse = 0.0;
  double final_photometric_rmse = 0.0;
  double initial_landmark_rmse = 0.0;
  double final_landmark_rmse = 0.0;
  // Full-length coefficient vector; only the leading block was optimized.
  Eigen::VectorXd expression_coefficients;
  CameraParameters camera;
  Eigen::VectorXd vertices;
  std::string summary;
};

// Analysis-by-synthesis refinement: minimizes the dense color difference
// between the shaded model and the photograph over expression (and
// optionally pose), with the semantic landmarks and the expression PCA
// prior as safeguards. Identity, albedo, illumination, and focal length
// stay fixed.
DenseFittingResult RefineExpressionDense(
    const std::string& image_path,
    const BfmModel& model,
    const FittingResult& fitting_result,
    const PhotometricResult& photometric_result,
    const DenseFittingOptions& options,
    const std::string& output_directory);

bool SaveDenseFittingReport(const DenseFittingResult& result,
                            const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_DENSE_FITTING_H_
