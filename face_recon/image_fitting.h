#ifndef FACE_RECON_IMAGE_FITTING_H_
#define FACE_RECON_IMAGE_FITTING_H_

#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

Eigen::Vector2i ReadImageSize(const std::string& image_path);

Eigen::VectorXd SampleVisibleVertexColorsFromImage(
    const std::string& image_path,
    const Eigen::VectorXd& vertices,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& fallback_colors);

bool SaveRasterDepthImage(const RasterizationResult& rasterization,
                          const std::string& output_path);
bool SaveVisibilityImage(const RasterizationResult& rasterization,
                         const std::string& output_path);

bool SaveReprojectionOverlay(const std::string& image_path,
                             const FittingResult& result,
                             const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_IMAGE_FITTING_H_
