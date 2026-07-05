#ifndef FACE_RECON_IMAGE_FITTING_H_
#define FACE_RECON_IMAGE_FITTING_H_

#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

Eigen::Vector2i ReadImageSize(const std::string& image_path);

bool SaveRasterDepthImage(const RasterizationResult& rasterization,
                          const std::string& output_path);
bool SaveVisibilityImage(const RasterizationResult& rasterization,
                         const std::string& output_path);
bool SaveBfmSurfaceOverlay(const std::string& image_path,
                           const Eigen::MatrixXi& triangles,
                           const Eigen::VectorXd& camera_normals,
                           const RasterizationResult& rasterization,
                           const std::string& output_path,
                           double alpha = 0.45);

bool SaveReprojectionOverlay(const std::string& image_path,
                             const FittingResult& result,
                             const std::string& output_path);

}  // namespace face_recon

#endif  // FACE_RECON_IMAGE_FITTING_H_
