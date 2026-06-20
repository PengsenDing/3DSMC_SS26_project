#ifndef FACE_RECON_PHOTOMETRIC_FITTING_H_
#define FACE_RECON_PHOTOMETRIC_FITTING_H_

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

struct PhotometricOptions {
  bool save_diagnostics = false;
  int num_albedo_coefficients = 30;
  int pixel_stride = 2;
  int mask_erosion = 1;
  int illumination_iterations = 5;
  int joint_iterations = 4;
  double albedo_regularization = 0.1;
  double illumination_regularization = 1.0e-4;
  double huber_delta = 0.08;
};

struct PhotometricResult {
  bool usable = false;
  int sample_count = 0;
  double initial_rmse = 0.0;
  double illumination_rmse = 0.0;
  double final_rmse = 0.0;
  Eigen::VectorXd albedo_coefficients;
  // One row per RGB channel; columns are [1, nx, ny, nz].
  Eigen::Matrix<double, 3, 4> illumination =
      Eigen::Matrix<double, 3, 4>::Zero();
  // Intrinsic BFM albedo after applying the optimized PCA coefficients.
  Eigen::VectorXd fitted_vertex_albedo;
};

Eigen::VectorXd ComputeVertexNormals(const Eigen::VectorXd& vertices,
                                     const Eigen::MatrixXi& triangles);

PhotometricResult FitPhotometricAppearance(
    const std::string& image_path,
    const BfmModel& model,
    const Eigen::VectorXd& vertices,
    const CameraParameters& camera,
    const RasterizationResult& rasterization,
    const PhotometricOptions& options,
    const std::string& output_directory);

bool SavePhotometricReport(const PhotometricResult& result,
                           const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_PHOTOMETRIC_FITTING_H_
