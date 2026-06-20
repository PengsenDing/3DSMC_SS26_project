#ifndef FACE_RECON_DENSE_REFINEMENT_H_
#define FACE_RECON_DENSE_REFINEMENT_H_

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/photometric_fitting.h"
#include "face_reconstruction/landmarks.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace face_recon {

struct DenseRefinementOptions {
  bool enabled = true;
  int resolution = 192;
  int shape_coefficients = 6;
  int expression_coefficients = 6;
  int passes = 1;
  double coefficient_step = 0.08;
  double minimum_step = 0.02;
  double landmark_weight = 2.0;
  double silhouette_weight = 0.05;
  double photometric_weight = 2.0;
  double shape_prior_weight = 0.002;
  double expression_prior_weight = 0.002;
  double landmark_degradation_limit = 1.05;
};

struct DenseObjective {
  double landmark_rmse = 0.0;
  double silhouette_loss = 0.0;
  double photometric_rmse = 0.0;
  double shape_prior = 0.0;
  double expression_prior = 0.0;
  double total = 0.0;
};

struct DenseRefinementResult {
  bool usable = false;
  int accepted_updates = 0;
  DenseObjective initial;
  DenseObjective final;
  Eigen::VectorXd shape_coefficients;
  Eigen::VectorXd expression_coefficients;
  Eigen::VectorXd vertices;
};

DenseRefinementResult RefineGeometryDense(
    const std::string& image_path,
    const BfmModel& model,
    const FittingResult& initial_fitting,
    const PhotometricResult& appearance,
    const std::vector<face_reconstruction::Landmark2D>& landmarks,
    const DenseRefinementOptions& options,
    const std::string& output_directory);

bool SaveDenseRefinementReport(const DenseRefinementResult& result,
                               const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_DENSE_REFINEMENT_H_
