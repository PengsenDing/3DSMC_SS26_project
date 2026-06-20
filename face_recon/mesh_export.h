#ifndef FACE_RECON_MESH_EXPORT_H_
#define FACE_RECON_MESH_EXPORT_H_

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"

#include <Eigen/Core>

#include <string>

namespace face_recon {

bool SaveMeshToOff(const BfmModel& model,
                   const Eigen::VectorXd& vertices,
                   const std::string& filename);
bool SaveMeshToPly(const BfmModel& model,
                   const Eigen::VectorXd& vertices,
                   const std::string& filename,
                   const Eigen::VectorXd* vertex_colors = nullptr);
bool SaveFittingReport(const FittingResult& result, const std::string& filename);
bool SaveReprojectionsToCsv(const FittingResult& result, const std::string& filename);

}  // namespace face_recon

#endif  // FACE_RECON_MESH_EXPORT_H_
