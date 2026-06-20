#include "face_recon/mesh_export.h"

#include <Eigen/Core>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace face_recon {
namespace {

int TriangleCount(const Eigen::MatrixXi& triangles) {
  return triangles.cols() == 3 ? triangles.rows() : triangles.cols();
}

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

}  // namespace

bool SaveMeshToOff(const BfmModel& model,
                   const Eigen::VectorXd& vertices,
                   const std::string& filename) {
  if (vertices.size() == 0 || vertices.size() % 3 != 0) {
    std::cerr << "[ERROR] Cannot export invalid fitted vertex vector.\n";
    return false;
  }

  std::ofstream output(filename);
  if (!output) {
    std::cerr << "[ERROR] Could not create OFF mesh: " << filename << "\n";
    return false;
  }

  const int vertex_count = static_cast<int>(vertices.size() / 3);
  const int triangle_count = TriangleCount(model.triangles());
  output << "OFF\n";
  output << vertex_count << " " << triangle_count << " 0\n";
  output << std::setprecision(10);

  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    output << vertices[3 * vertex] << " "
           << vertices[3 * vertex + 1] << " "
           << vertices[3 * vertex + 2] << "\n";
  }

  for (int triangle = 0; triangle < triangle_count; ++triangle) {
    output << "3 "
           << TriangleIndex(model.triangles(), triangle, 0) << " "
           << TriangleIndex(model.triangles(), triangle, 1) << " "
           << TriangleIndex(model.triangles(), triangle, 2) << "\n";
  }

  std::cout << "[SUCCESS] Saved fitted OFF mesh: " << filename << "\n";
  return true;
}

bool SaveMeshToPly(const BfmModel& model,
                   const Eigen::VectorXd& vertices,
                   const std::string& filename,
                   const Eigen::VectorXd* vertex_colors) {
  if (vertices.size() == 0 || vertices.size() % 3 != 0) {
    std::cerr << "[ERROR] Cannot export invalid fitted vertex vector.\n";
    return false;
  }

  std::ofstream output(filename);
  if (!output) {
    std::cerr << "[ERROR] Could not create PLY mesh: " << filename << "\n";
    return false;
  }

  const int vertex_count = static_cast<int>(vertices.size() / 3);
  const int triangle_count = TriangleCount(model.triangles());
  const Eigen::VectorXd& colors =
      vertex_colors != nullptr ? *vertex_colors : model.color().mean;
  const bool has_albedo = colors.size() == vertices.size();
  const double color_scale =
      has_albedo && colors.maxCoeff() <= 1.05 ? 255.0 : 1.0;

  output << "ply\n";
  output << "format ascii 1.0\n";
  output << "element vertex " << vertex_count << "\n";
  output << "property float x\n";
  output << "property float y\n";
  output << "property float z\n";
  output << "property uchar red\n";
  output << "property uchar green\n";
  output << "property uchar blue\n";
  output << "element face " << triangle_count << "\n";
  output << "property list uchar int vertex_indices\n";
  output << "end_header\n";
  output << std::setprecision(10);

  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    int red = 184;
    int green = 140;
    int blue = 117;
    if (has_albedo) {
      red = std::clamp(
          static_cast<int>(colors[3 * vertex] * color_scale), 0, 255);
      green = std::clamp(
          static_cast<int>(colors[3 * vertex + 1] * color_scale), 0, 255);
      blue = std::clamp(
          static_cast<int>(colors[3 * vertex + 2] * color_scale), 0, 255);
    }
    output << vertices[3 * vertex] << " "
           << vertices[3 * vertex + 1] << " "
           << vertices[3 * vertex + 2] << " "
           << red << " " << green << " " << blue << "\n";
  }

  for (int triangle = 0; triangle < triangle_count; ++triangle) {
    output << "3 "
           << TriangleIndex(model.triangles(), triangle, 0) << " "
           << TriangleIndex(model.triangles(), triangle, 1) << " "
           << TriangleIndex(model.triangles(), triangle, 2) << "\n";
  }

  std::cout << "[SUCCESS] Saved fitted colored PLY mesh: " << filename << "\n";
  return true;
}

bool SaveFittingReport(const FittingResult& result, const std::string& filename) {
  std::ofstream output(filename);
  if (!output) {
    std::cerr << "[ERROR] Could not create fitting report: " << filename << "\n";
    return false;
  }

  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "matched_landmarks: " << result.reprojections.size() << "\n";
  output << "semantic_landmarks: " << result.semantic_landmark_count << "\n";
  output << "contour_landmarks: " << result.contour_landmark_count << "\n";
  output << "rejected_semantic_landmarks: "
         << result.rejected_landmark_count << "\n";
  output << "initial_rmse_normalized: " << result.initial_rmse << "\n";
  output << "final_rmse_normalized: " << result.final_rmse << "\n";
  output << "camera_angle_axis: "
         << result.camera.angle_axis.x() << " "
         << result.camera.angle_axis.y() << " "
         << result.camera.angle_axis.z() << "\n";
  output << "camera_scale: " << result.camera.scale << "\n";
  output << "camera_translation: "
         << result.camera.translation_x << " "
         << result.camera.translation_y << "\n";
  output << "shape_coefficients: " << result.shape_coefficients.size() << "\n";
  for (int index = 0; index < result.shape_coefficients.size(); ++index) {
    output << "shape[" << index << "]: " << result.shape_coefficients[index] << "\n";
  }
  output << "expression_coefficients: " << result.expression_coefficients.size() << "\n";
  for (int index = 0; index < result.expression_coefficients.size(); ++index) {
    output << "expression[" << index << "]: "
           << result.expression_coefficients[index] << "\n";
  }
  output << "\n" << result.solver_summary << "\n";
  return true;
}

bool SaveReprojectionsToCsv(const FittingResult& result, const std::string& filename) {
  std::ofstream output(filename);
  if (!output) {
    std::cerr << "[ERROR] Could not create reprojection CSV: " << filename << "\n";
    return false;
  }

  output << "name,mediapipe_index,bfm_vertex_id,observed_x_norm,observed_y_norm,"
            "projected_x_norm,projected_y_norm,error_norm\n";
  output << std::setprecision(10);
  for (const auto& reprojection : result.reprojections) {
    const double error =
        (reprojection.projected - reprojection.observed).norm();
    output << reprojection.name << ","
           << reprojection.mediapipe_index << ","
           << reprojection.bfm_vertex_id << ","
           << reprojection.observed.x() << ","
           << reprojection.observed.y() << ","
           << reprojection.projected.x() << ","
           << reprojection.projected.y() << ","
           << error << "\n";
  }
  return true;
}

}  // namespace face_recon
