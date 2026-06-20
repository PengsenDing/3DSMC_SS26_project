#ifndef FACE_RECON_TEXTURE_FITTING_H_
#define FACE_RECON_TEXTURE_FITTING_H_

#include "face_recon/rasterizer.h"
#include "face_reconstruction/landmarks.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace face_recon {

struct TextureFittingOptions {
  int pixel_stride = 1;
  int mask_erosion = 1;
  int solver_iterations = 300;
  double prior_weight = 0.02;
  double smoothness_weight = 0.01;
  double solver_tolerance = 1.0e-6;
};

struct TextureFittingResult {
  bool usable = false;
  int sample_count = 0;
  int visible_vertex_count = 0;
  double initial_rmse = 0.0;
  double final_rmse = 0.0;
  Eigen::VectorXd vertex_colors;
};

TextureFittingResult FitDenseVertexColors(
    const std::string& image_path,
    const Eigen::MatrixXi& triangles,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& initial_vertex_colors,
    const std::vector<face_reconstruction::Landmark2D>* landmarks,
    const TextureFittingOptions& options,
    const std::string& output_directory);

bool SaveTextureFittingReport(const TextureFittingResult& result,
                              const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_TEXTURE_FITTING_H_
