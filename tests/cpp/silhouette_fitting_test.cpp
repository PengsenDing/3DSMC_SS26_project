#include "face_recon/silhouette_fitting.h"

#include <Eigen/Core>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

namespace {

cv::Mat RasterMask(const face_recon::RasterizationResult& rasterization) {
  cv::Mat mask(rasterization.height, rasterization.width, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  return mask;
}

}  // namespace

int main() {
  constexpr int kVertexCount = 9;
  face_recon::BfmModel model;
  face_recon::PcaComponent shape;
  shape.mean.resize(3 * kVertexCount);
  shape.mean << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.7, 0.7, 0.0, 1.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0,
      -1.0, 0.0, -0.7, -0.7, 0.0, -1.0, 0.0, 0.0, -0.7, 0.7, 0.0;
  shape.pca_basis = Eigen::MatrixXd::Zero(3 * kVertexCount, 1);
  for (int vertex = 1; vertex < kVertexCount; ++vertex) {
    shape.pca_basis(3 * vertex, 0) = 0.35 * shape.mean[3 * vertex];
  }
  shape.pca_variance = Eigen::VectorXd::Ones(1);
  model.set_shape(shape);

  face_recon::PcaComponent expression;
  expression.mean = Eigen::VectorXd::Zero(3 * kVertexCount);
  expression.pca_basis = Eigen::MatrixXd::Zero(3 * kVertexCount, 1);
  expression.pca_variance = Eigen::VectorXd::Ones(1);
  model.set_expression(expression);

  Eigen::MatrixXi triangles(8, 3);
  for (int triangle = 0; triangle < 8; ++triangle) {
    triangles(triangle, 0) = 0;
    triangles(triangle, 1) = triangle + 1;
    triangles(triangle, 2) = triangle == 7 ? 1 : triangle + 2;
  }
  model.set_triangles(triangles);

  face_recon::FittingResult initial;
  initial.usable = true;
  initial.camera.translation = Eigen::Vector3d(0.0, 0.0, 5.0);
  initial.camera.focal_length = 1.5;
  initial.camera.aspect_ratio = 1.0;
  initial.shape_coefficients = Eigen::VectorXd::Zero(1);
  initial.expression_coefficients = Eigen::VectorXd::Zero(1);
  initial.vertices = face_recon::GenerateVertices(model, initial.shape_coefficients,
                                                  initial.expression_coefficients);

  // Center, top, and bottom anchors preserve pose/height without directly
  // prescribing the horizontal silhouette deformation under test.
  for (const int vertex_id : {0, 1, 5}) {
    face_recon::LandmarkReprojection landmark;
    landmark.name = "semantic_" + std::to_string(vertex_id);
    landmark.bfm_vertex_id = vertex_id;
    landmark.observed =
        face_recon::ProjectVertex(initial.vertices.segment<3>(3 * vertex_id), initial.camera);
    landmark.projected = landmark.observed;
    initial.reprojections.push_back(landmark);
  }

  Eigen::VectorXd target_shape(1);
  target_shape << 0.8;
  const Eigen::VectorXd target_vertices =
      face_recon::GenerateVertices(model, target_shape, initial.expression_coefficients);
  const cv::Mat target_mask =
      RasterMask(face_recon::RasterizeMesh(target_vertices, triangles, initial.camera, 128, 128));
  const std::filesystem::path mask_path =
      std::filesystem::temp_directory_path() / "face_reconstruction_silhouette_test.png";
  if (!cv::imwrite(mask_path.string(), target_mask)) {
    std::cerr << "Could not write synthetic silhouette mask\n";
    return 1;
  }

  face_recon::SilhouetteFittingOptions options;
  options.resolution = 128;
  options.outer_iterations = 3;
  options.solver_iterations = 30;
  options.sample_stride = 2;
  options.silhouette_weight = 100.0;
  options.shape_regularization = 1.0e-4;
  options.maximum_correspondence_distance = 0.25;
  const face_recon::SilhouetteFittingResult result = face_recon::RefineGeometryFromSilhouette(
      mask_path.string(), model, initial, options, std::filesystem::temp_directory_path().string());
  std::filesystem::remove(mask_path);

  std::cout << "Synthetic silhouette Chamfer: " << result.initial.silhouette_chamfer << " -> "
            << result.final.silhouette_chamfer << ", IoU: " << result.initial.silhouette_iou
            << " -> " << result.final.silhouette_iou << "\n";
  if (!result.usable || result.final.silhouette_chamfer >= result.initial.silhouette_chamfer ||
      result.final.silhouette_iou <= result.initial.silhouette_iou ||
      result.shape_coefficients[0] <= 0.2) {
    std::cerr << result.solver_summary;
    return 1;
  }
  return 0;
}
