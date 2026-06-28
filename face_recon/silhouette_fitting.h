#ifndef FACE_RECON_SILHOUETTE_FITTING_H_
#define FACE_RECON_SILHOUETTE_FITTING_H_

#include <Eigen/Core>
#include <string>

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"

namespace face_recon {

struct SilhouetteFittingOptions {
  bool enabled = true;
  bool save_diagnostics = false;
  int resolution = 192;
  int outer_iterations = 3;
  int solver_iterations = 60;
  int sample_stride = 2;
  double semantic_weight = 100.0;
  double silhouette_weight = 10.0;
  double shape_regularization = 0.1;
  double huber_delta = 0.015;
  double maximum_correspondence_distance = 0.08;
  double semantic_degradation_limit = 1.15;
};

struct SilhouetteMetrics {
  double semantic_rmse = 0.0;
  double silhouette_chamfer = 0.0;
  double silhouette_iou = 0.0;
};

struct SilhouetteFittingResult {
  bool usable = false;
  int completed_outer_iterations = 0;
  int silhouette_correspondences = 0;
  SilhouetteMetrics initial;
  SilhouetteMetrics final;
  Eigen::VectorXd shape_coefficients;
  Eigen::VectorXd vertices;
  std::string solver_summary;
};

SilhouetteFittingResult RefineGeometryFromSilhouette(const std::string& silhouette_mask_path,
                                                     const BfmModel& model,
                                                     const FittingResult& initial_fitting,
                                                     const SilhouetteFittingOptions& options,
                                                     const std::string& output_directory);

void ApplySilhouetteResult(const SilhouetteFittingResult& refinement, FittingResult* fitting);

bool SaveSilhouetteFittingReport(const SilhouetteFittingResult& result,
                                 const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_SILHOUETTE_FITTING_H_
