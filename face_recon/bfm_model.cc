#include "face_recon/bfm_model.h"

#define HIGHFIVE_USE_EIGEN
#include <highfive/H5Easy.hpp>
#include <nlohmann/json.hpp>
#include <glog/logging.h>
#include <Eigen/Dense>

#include <fstream>   
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <limits>
#include <algorithm>

namespace face_recon {

void PcaComponent::PrintStats(const std::string& name) const {
  std::cout << "  [" << name << " Subspace] (v" << major_version << "." << minor_version << "):\n";
  std::cout << "    - Mean Vector Size:   " << mean.size() << "\n";
  std::cout << "    - PCA Basis Matrix:  " << pca_basis.rows() << " x " << pca_basis.cols() << "\n";
  std::cout << "    - Variance Vector:   " << pca_variance.size() << "\n";
  std::cout << "    - Noise Variance:    " << noise_variance << "\n\n";
  std::cout << "    - Build Time:        " << build_time << "\n";
}

bool BfmModel::LoadFromH5(const std::string& filepath) {
  try {
    HighFive::File file(filepath, HighFive::File::ReadOnly);

    // Extract Global Root Metadata.
    root_major_version_ = H5Easy::load<int>(file, "/version/majorVersion");
    root_minor_version_ = H5Easy::load<int>(file, "/version/minorVersion");
    catalog_path_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel/modelPath");
    catalog_type_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel/modelType");
    color_catalog_path_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.color/modelPath");
    color_catalog_type_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.color/modelType");
    expression_catalog_path_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.expression/modelPath");
    expression_catalog_type_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.expression/modelType");
    shape_catalog_path_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.shape/modelPath");
    shape_catalog_type_ = H5Easy::load<std::string>(file, "/catalog/MorphableModel.shape/modelType");
    color_space_ = H5Easy::load<std::string>(file, "/color/representer/colorspace");

    // Extract Shape Subspace.
    shape_.mean = H5Easy::load<Eigen::VectorXd>(file, "/shape/model/mean");
    shape_.pca_basis = H5Easy::load<Eigen::MatrixXd>(file, "/shape/model/pcaBasis");
    shape_.pca_variance = H5Easy::load<Eigen::VectorXd>(file, "/shape/model/pcaVariance");
    shape_.noise_variance = H5Easy::load<double>(file, "/shape/model/noiseVariance");
    shape_.major_version = H5Easy::load<int>(file, "/shape/version/majorVersion");
    shape_.minor_version = H5Easy::load<int>(file, "/shape/version/minorVersion");
    shape_.build_time = H5Easy::load<std::string>(file, "/shape/modelinfo/build-time");

    // Extract Expression Subspace.
    expression_.mean = H5Easy::load<Eigen::VectorXd>(file, "/expression/model/mean");
    expression_.pca_basis = H5Easy::load<Eigen::MatrixXd>(file, "/expression/model/pcaBasis");
    expression_.pca_variance = H5Easy::load<Eigen::VectorXd>(file, "/expression/model/pcaVariance");
    expression_.noise_variance = H5Easy::load<double>(file, "/expression/model/noiseVariance");
    expression_.major_version = H5Easy::load<int>(file, "/expression/version/majorVersion");
    expression_.minor_version = H5Easy::load<int>(file, "/expression/version/minorVersion");
    expression_.build_time = H5Easy::load<std::string>(file, "/expression/modelinfo/build-time");

    // Extract Color/Albedo Subspace.
    color_.mean = H5Easy::load<Eigen::VectorXd>(file, "/color/model/mean");
    color_.pca_basis = H5Easy::load<Eigen::MatrixXd>(file, "/color/model/pcaBasis");
    color_.pca_variance = H5Easy::load<Eigen::VectorXd>(file, "/color/model/pcaVariance");
    color_.noise_variance = H5Easy::load<double>(file, "/color/model/noiseVariance");
    color_.major_version = H5Easy::load<int>(file, "/color/version/majorVersion");
    color_.minor_version = H5Easy::load<int>(file, "/color/version/minorVersion");
    color_.build_time = H5Easy::load<std::string>(file, "/color/modelinfo/build-time");

    // Extract Topology and Face Template from Shape Subspace (all subspaces are based on the same topology).
    triangles_ = H5Easy::load<Eigen::MatrixXi>(file, "/shape/representer/cells");
    points_ = H5Easy::load<Eigen::MatrixXd>(file, "/shape/representer/points");

    // Parse Landmarks from JSON Metadata
    Eigen::MatrixXd template_points = points_;
    std::string raw_json;
    std::cout << "[INFO] Successfully loaded model data. Now parsing landmarks...\n";
    if (file.exist("/metadata/landmarks/json")) {
      raw_json = H5Easy::load<std::string>(file, "/metadata/landmarks/json");
    } else {
      std::cerr << "[ERROR] Landmark path missing inside HDF5 container.\n";
      return false;
    }
    
    return ParseLandmarksFromJson(raw_json, template_points);
  }catch (const HighFive::Exception& e) {
    std::cerr << "[ERROR] HighFive HDF5 parsing failed: " << e.what()
              << std::endl;
    return false;
  }
}

bool BfmModel::ParseLandmarksFromJson(std::string raw_json, const Eigen::MatrixXd& template_points) {
  try {
    // Parse the JSON string into a structured format.
    nlohmann::json json_array = nlohmann::json::parse(raw_json);
    landmarks_.clear();

    // Calculate total vertices directly from the 1D mean vector (3 values per vertex).
    const int total_vertices = template_points.cols();
    std::cout << "[INFO] Comparing " << total_vertices << " template vertices...\n";

    // Get closest vertex for each landmark and store the mapping.
    for (const auto& item : json_array) {
      if (!item.contains("id") || !item.contains("coordinates")) continue;

      std::string landmark_id = item["id"].get<std::string>();
      auto coords = item["coordinates"].get<std::vector<double>>();
      if (coords.size() < 3) continue;

      Eigen::Vector3d landmark_pos(coords[0], coords[1], coords[2]);

      int closest_vertex_idx = -1;
      double min_squared_distance = std::numeric_limits<double>::max();

      for (int i = 0; i < total_vertices; ++i) {
        // Unpack the sequential [x, y, z] coordinates from the template_points matrix.
        Eigen::Vector3d vertex_pos(template_points(0, i), 
                                   template_points(1, i), 
                                   template_points(2, i));
        
        double current_sq_dist = (vertex_pos - landmark_pos).squaredNorm();
        if (current_sq_dist < min_squared_distance) {
          min_squared_distance = current_sq_dist;
          closest_vertex_idx = i;
        }
      }

      if (closest_vertex_idx != -1) {
        landmarks_[landmark_id] = closest_vertex_idx;
      }
    }

    return true;
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << "[FATAL] Landmark JSON parsing error: " << e.what() << "\n";
    return false;
  }
}

void BfmModel::PrintModelSummary() const {
  std::cout << "==================================================\n";
  std::cout << "       BASEL FACE MODEL 2019 COMPREHENSIVE DUMP   \n";
  std::cout << "==================================================\n";
  std::cout << "File Specifications:\n";
  std::cout << "  - Statismo Spec Version:  v" << root_major_version_ << "." << root_minor_version_ << "\n";
  std::cout << "  - Catalog Model Type:     " << catalog_type_ << "\n";
  std::cout << "  - Catalog Intended Path:  " << catalog_path_ << "\n\n";
  std::cout << "Component Catalogs:\n";
  std::cout << "  - Shape Catalog Type:     " << shape_catalog_type_ << "\n";
  std::cout << "  - Shape Catalog Path:     " << shape_catalog_path_ << "\n";
  std::cout << "  - Expression Catalog Type: " << expression_catalog_type_ << "\n";
  std::cout << "  - Expression Catalog Path: " << expression_catalog_path_ << "\n";
  std::cout << "  - Color Catalog Type:     " << color_catalog_type_ << "\n";
  std::cout << "  - Color Catalog Path:     " << color_catalog_path_ << "\n\n";

  std::cout << " - Color Space:              " << color_space_ << "\n\n";

  std::cout << "Topology:\n";
  std::cout << "  - Total Triangles:      " << triangles_.cols() << "\n";
  std::cout << "  - Total Template Points: " << points_.cols() << "\n";

  // System Core Metrics.
  shape_.PrintStats("Shape (Identity)");
  expression_.PrintStats("Expression");
  color_.PrintStats("Color (Albedo)");

  std::cout << "Semantic Landmarks:\n";
  std::cout << "  - Total Anchors Discovered: " << landmarks_.size() << "\n";
  
  // Dump all the landmark associations to verify text key extraction.
  int count = 0;
  for (const auto& [name, index] : landmarks_) {
    std::cout << "    * Anchor Name: \"" << name << "\" maps to Vertex ID: " << index << "\n";
    count++;
  }
  std::cout << "==================================================\n";
}

bool BfmModel::SaveMeanMeshToPly(const std::string& filename, const bool include_landmarks) const {
  if (shape_.mean.size() == 0 || color_.mean.size() == 0) {
    std::cerr << "[ERROR] Mesh data arrays are uninitialized.\n";
    return false;
  }

  std::ofstream ply_file(filename);
  if (!ply_file.is_open()) {
    std::cerr << "[ERROR] Failed to create PLY container: " << filename << "\n";
    return false;
  }

  // Calculate base mesh dimensions.
  const int base_vertices = shape_.mean.size() / 3;
  int base_faces = (triangles_.cols() == 3) ? triangles_.rows() : triangles_.cols();
  const bool is_row_per_face = (triangles_.cols() == 3);

  // Account for our 3D Landmark Glyphs (Octahedrons) if needed.
  // Each tracking anchor adds 6 vertices and 8 triangular faces.
  const int num_landmarks = include_landmarks ? static_cast<int>(landmarks_.size()) : 0;
  const int total_vertices = base_vertices + (num_landmarks * 6);
  const int total_faces = base_faces + (num_landmarks * 8);

  // Adjustable landmark size in millimeters (change this to scale the glyphs).
  const double glyph_size = 2.0; 

  // Write PLY Header.
  ply_file << "ply\n";
  ply_file << "format ascii 1.0\n";
  ply_file << "element vertex " << total_vertices << "\n";
  ply_file << "property float x\n";
  ply_file << "property float y\n";
  ply_file << "property float z\n";
  ply_file << "property uchar red\n";
  ply_file << "property uchar green\n";
  ply_file << "property uchar blue\n";
  ply_file << "element face " << total_faces << "\n";
  ply_file << "property list uchar int vertex_indices\n";
  ply_file << "end_header\n";

  // Determine color scaling.
  double max_color_val = color_.mean.maxCoeff();
  double color_scale = (max_color_val <= 1.05) ? 255.0 : 1.0;

  // Stream Mean Face Vertices (with Mean Albedo Texturing).
  Eigen::VectorXd base_face_mean = shape_.mean + expression_.mean;
  for (int i = 0; i < base_vertices; ++i) {
    double x = base_face_mean(3 * i + 0);
    double y = base_face_mean(3 * i + 1);
    double z = base_face_mean(3 * i + 2);

    int r = std::clamp(static_cast<int>(color_.mean(3 * i + 0) * color_scale), 0, 255);
    int g = std::clamp(static_cast<int>(color_.mean(3 * i + 1) * color_scale), 0, 255);
    int b = std::clamp(static_cast<int>(color_.mean(3 * i + 2) * color_scale), 0, 255);

    ply_file << x << " " << y << " " << z << " " << r << " " << g << " " << b << "\n";
  }

  // Stream Landmark Glyph Vertices if needed (Pure Red: 255, 0, 0).
  if (include_landmarks) {
    for (const auto& [name, vertex_idx] : landmarks_) {
      // Extract the exact 3D location of this anchor point on the mean face.
      double cx = base_face_mean(3 * vertex_idx + 0);
      double cy = base_face_mean(3 * vertex_idx + 1);
      double cz = base_face_mean(3 * vertex_idx + 2);

      // Generate 6 cross tips floating around the center point.
      ply_file << (cx + glyph_size) << " " << cy << " " << cz << " 255 0 0\n"; // Right
      ply_file << (cx - glyph_size) << " " << cy << " " << cz << " 255 0 0\n"; // Left
      ply_file << cx << " " << (cy + glyph_size) << " " << cz << " 255 0 0\n"; // Up
      ply_file << cx << " " << (cy - glyph_size) << " " << cz << " 255 0 0\n"; // Down
      ply_file << cx << " " << cy << " " << (cz + glyph_size) << " 255 0 0\n"; // Forward
      ply_file << cx << " " << cy << " " << (cz - glyph_size) << " 255 0 0\n"; // Backward
    }
  }

  // Stream Original Base Mesh Triangles.
  for (int i = 0; i < base_faces; ++i) {
    int idx0 = is_row_per_face ? triangles_(i, 0) : triangles_(0, i);
    int idx1 = is_row_per_face ? triangles_(i, 1) : triangles_(1, i);
    int idx2 = is_row_per_face ? triangles_(i, 2) : triangles_(2, i);
    ply_file << "3 " << idx0 << " " << idx1 << " " << idx2 << "\n";
  }

  // Stream Landmark Glyph Triangles if needed (stitch the diamonds together).
  if (include_landmarks) {
    int landmark_count = 0;
    for (const auto& [name, vertex_idx] : landmarks_) {
      // Calculate where this specific landmark's 6 vertices start in the global PLY block.
      int base_idx = base_vertices + (landmark_count * 6);

      // Local vertex aliases for clarity.
      int xp = base_idx + 0; // +X
      int xn = base_idx + 1; // -X
      int yp = base_idx + 2; // +Y
      int yn = base_idx + 3; // -Y
      int zp = base_idx + 4; // +Z
      int zn = base_idx + 5; // -Z

      // Top Pyramid Faces
      ply_file << "3 " << zp << " " << xp << " " << yp << "\n";
      ply_file << "3 " << zp << " " << yp << " " << xn << "\n";
      ply_file << "3 " << zp << " " << xn << " " << yn << "\n";
      ply_file << "3 " << zp << " " << yn << " " << xp << "\n";

      // Bottom Pyramid Faces
      ply_file << "3 " << zn << " " << yp << " " << xp << "\n";
      ply_file << "3 " << zn << " " << xn << " " << yp << "\n";
      ply_file << "3 " << zn << " " << yn << " " << xn << "\n";
      ply_file << "3 " << zn << " " << xp << " " << yn << "\n";

      landmark_count++;
    }
  }

  ply_file.close();
  std::cout << "[SUCCESS] Exported mesh populated with 3D landmark anchors to: " << filename << "\n";
  return true;
}

bool BfmModel::SaveLandmarksToTxt(const std::string& filename) const {
  std::ofstream txt_file(filename);
  if (!txt_file.is_open()) return false;

  txt_file << "# Landmark Name | Vertex ID | X | Y | Z\n";
  for (const auto& [name, vertex_idx] : landmarks_) {
    double cx = shape_.mean(3 * vertex_idx + 0);
    double cy = shape_.mean(3 * vertex_idx + 1);
    double cz = shape_.mean(3 * vertex_idx + 2);

    txt_file << name << " " << vertex_idx << " " << cx << " " << cy << " " << cz << "\n";
  }
  txt_file.close();
  return true;
}

int BfmModel::GetLandmarkVertexId(const std::string& name) const {
  auto it = landmarks_.find(name);
  if (it == landmarks_.end()) {
    throw std::runtime_error("BFM landmark not found: " + name);
  }
  return it->second;
}

Eigen::Vector3d BfmModel::GetMeanVertex(int vertex_id) const {
  Eigen::VectorXd mean = shape_.mean + expression_.mean;

  return Eigen::Vector3d(
      mean(3 * vertex_id + 0),
      mean(3 * vertex_id + 1),
      mean(3 * vertex_id + 2));
}

Eigen::MatrixXd BfmModel::GetShapeBasisForVertex(int vertex_id, int num_shape) const {
  return shape_.pca_basis.block(3 * vertex_id, 0, 3, num_shape);
}

Eigen::MatrixXd BfmModel::GetExpressionBasisForVertex(int vertex_id, int num_expr) const {
  return expression_.pca_basis.block(3 * vertex_id, 0, 3, num_expr);
}

const Eigen::VectorXd& BfmModel::ShapeVariance() const {
  return shape_.pca_variance;
}

const Eigen::VectorXd& BfmModel::ExpressionVariance() const {
  return expression_.pca_variance;
}

}  // namespace face_recon