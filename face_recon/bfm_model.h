// Defines the core asset loader and semantic manager for the Basel Face Model.

#ifndef SRC_BFM_MODEL_H_
#define SRC_BFM_MODEL_H_

#include <Eigen/Core>
#include <string>
#include <map>

namespace face_recon {

// Represents a low-dimensional PCA subspace layout from the Basel Face Model.
struct PcaComponent {
  Eigen::VectorXd mean;
  Eigen::MatrixXd pca_basis;
  Eigen::VectorXd pca_variance;
  double noise_variance;
  int major_version;
  int minor_version;
  std::string build_time;

  // Writes a summary of the PCA component's dimensions and metadata to standard console logging.
  void PrintStats(const std::string& name) const;
};

// Manages the statistical data structures of a 3D Morphable Model (3DMM).
// This class loads shape, expression, and color PCA spaces from an HDF5 container,
// maintains topological mappings for facial landmarks, and provides export utilities
// for visualization in external mesh viewing tools (e.g., MeshLab).
//
// This class is thread-compatible but not thread-safe.
class BfmModel {
 public:
  BfmModel() = default;
  ~BfmModel() = default;

  // Disallow copying to prevent heavy matrices from leaking performance.
  BfmModel(const BfmModel&) = delete;
  BfmModel& operator=(const BfmModel&) = delete;

  // Loads the full 3DMM subspace matrices from a specified HDF5 asset file.
  // Parses internal geometry, albedo, and metadata arrays.
  // Returns true on successful configuration, false if file reading fails.
  bool LoadFromH5(const std::string& filepath);

  // Writes the statistical mean mesh to disk as an ASCII PLY file.
  // The exported mesh includes coordinates and raw vertex albedo RGB colors.
  // Returns true if the file was written successfully.
  bool SaveMeanMeshToPly(const std::string& filename) const;

  // Exports the mean face mesh combined with 3D octagonal glyphs that visually
  // highlight the exact spatial locations of the loaded landmark points.
  // Returns true if the file was written successfully.
  bool SaveMeanMeshWithLandmarksToPly(const std::string& filename) const;

  // Exports a simple text file listing landmark names and their corresponding vertex indices.
  bool SaveLandmarksToTxt(const std::string& filename) const;

  // Dumps model layer shapes and asset metadata to standard console logging.
  void PrintModelSummary() const;

  // Read-only accessors.
  const PcaComponent& shape() const { return shape_; }
  const PcaComponent& expression() const { return expression_; }
  const PcaComponent& color() const { return color_; }
  const std::string& color_space() const { return color_space_; }
  const Eigen::MatrixXi& triangles() const { return triangles_; }
  const Eigen::MatrixXd& points() const { return points_; }
  const std::map<std::string, int>& landmarks() const { return landmarks_; }

 private:
  // Resolves semantic landmark string keys to exact mesh vertex indices by
  // matching JSON coordinate targets against the reference template points.
  bool ParseLandmarksFromJson(std::string raw_json, const Eigen::MatrixXd& template_points);

  // Private Member Variables

  PcaComponent shape_;
  PcaComponent expression_;
  PcaComponent color_;
  std::string color_space_;
  Eigen::MatrixXi triangles_;         // Mesh topology (vertex indices).
  Eigen::MatrixXd points_;            // Template face Vertex coordinates.
  // Landmark tracking dictionary mapping names to 3D mesh vertex IDs
  std::map<std::string, int> landmarks_;

  // Metadata fields from the HDF5 container.
  int root_major_version_;
  int root_minor_version_;
  std::string catalog_path_;
  std::string catalog_type_;
  std::string color_catalog_path_;  
  std::string color_catalog_type_;
  std::string expression_catalog_path_;
  std::string expression_catalog_type_;
  std::string shape_catalog_path_;
  std::string shape_catalog_type_;
};

}  // namespace face_recon

#endif  // SRC_BFM_MODEL_H_