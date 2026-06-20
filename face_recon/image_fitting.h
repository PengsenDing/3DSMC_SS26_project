#ifndef FACE_RECON_IMAGE_FITTING_H_
#define FACE_RECON_IMAGE_FITTING_H_

#include "face_recon/fitting.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

Eigen::VectorXd SampleVertexColorsFromImage(
    const std::string& image_path,
    const Eigen::VectorXd& vertices,
    const CameraParameters& camera,
    const Eigen::VectorXd& fallback_colors);

bool SaveReprojectionOverlay(const std::string& image_path,
                             const FittingResult& result,
                             const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_IMAGE_FITTING_H_
