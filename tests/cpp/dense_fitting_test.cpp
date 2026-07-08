// Regression test for the dense photometric expression refinement.
//
// A textured quad has one expression component that shifts it sideways. The
// target image is rendered at a known coefficient; the fit starts at zero and
// must recover the shift from pixel colors alone (no landmark anchors).

#include "face_recon/dense_fitting.h"
#include "face_recon/fitting.h"
#include "face_recon/rasterizer.h"

#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

constexpr int kImageSize = 240;
constexpr double kTrueDelta = 0.8;

void BuildQuadModel(face_recon::BfmModel* model) {
  face_recon::PcaComponent shape;
  shape.mean = Eigen::VectorXd(12);
  shape.mean << -50.0, -50.0, 0.0,
                 50.0, -50.0, 0.0,
                 50.0,  50.0, 0.0,
                -50.0,  50.0, 0.0;
  shape.pca_basis = Eigen::MatrixXd::Zero(12, 1);
  shape.pca_variance = Eigen::VectorXd::Ones(1);

  face_recon::PcaComponent expression;
  expression.mean = Eigen::VectorXd::Zero(12);
  expression.pca_basis = Eigen::MatrixXd::Zero(12, 1);
  for (int vertex = 0; vertex < 4; ++vertex) {
    expression.pca_basis(3 * vertex, 0) = 20.0;  // +x shift per unit delta
  }
  expression.pca_variance = Eigen::VectorXd::Ones(1);

  face_recon::PcaComponent color;
  color.mean = Eigen::VectorXd::Constant(12, 0.5);
  color.pca_basis = Eigen::MatrixXd::Zero(12, 1);
  color.pca_variance = Eigen::VectorXd::Ones(1);

  Eigen::MatrixXi triangles(2, 3);
  triangles << 0, 1, 2,
               0, 2, 3;

  model->set_shape(shape);
  model->set_expression(expression);
  model->set_color(color);
  model->set_triangles(triangles);
}

// Smooth color ramps across the quad so the photometric term has a usable
// gradient at every pixel.
Eigen::VectorXd BuildVertexAlbedo() {
  Eigen::VectorXd albedo(12);
  albedo << 0.2, 0.5, 0.3,
            0.8, 0.5, 0.3,
            0.8, 0.5, 0.7,
            0.2, 0.5, 0.7;
  return albedo;
}

cv::Mat RenderTarget(const face_recon::BfmModel& model,
                     const face_recon::CameraParameters& camera,
                     const Eigen::VectorXd& vertex_albedo,
                     double delta) {
  Eigen::VectorXd expression(1);
  expression << delta;
  const Eigen::VectorXd vertices = face_recon::GenerateVertices(
      model, Eigen::VectorXd::Zero(1), expression);
  const face_recon::RasterizationResult rasterization =
      face_recon::RasterizeMesh(vertices, model.triangles(), camera,
                                kImageSize, kImageSize);

  cv::Mat image(kImageSize, kImageSize, CV_8UC3, cv::Scalar(128, 128, 128));
  for (int y = 0; y < kImageSize; ++y) {
    for (int x = 0; x < kImageSize; ++x) {
      const face_recon::RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = model.triangles()(pixel.triangle_id, 0);
      const int i1 = model.triangles()(pixel.triangle_id, 1);
      const int i2 = model.triangles()(pixel.triangle_id, 2);
      const Eigen::Vector3d rgb =
          static_cast<double>(pixel.barycentric[0]) *
              vertex_albedo.segment<3>(3 * i0) +
          static_cast<double>(pixel.barycentric[1]) *
              vertex_albedo.segment<3>(3 * i1) +
          static_cast<double>(pixel.barycentric[2]) *
              vertex_albedo.segment<3>(3 * i2);
      image.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<unsigned char>(std::lround(rgb[2] * 255.0)),
                    static_cast<unsigned char>(std::lround(rgb[1] * 255.0)),
                    static_cast<unsigned char>(std::lround(rgb[0] * 255.0)));
    }
  }
  return image;
}

}  // namespace

int main() {
  face_recon::BfmModel model;
  BuildQuadModel(&model);
  const Eigen::VectorXd vertex_albedo = BuildVertexAlbedo();

  face_recon::CameraParameters camera;
  camera.aspect_ratio = 1.0;  // defaults: identity rotation, z=400, f=1.2

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "dense_fitting_test";
  std::filesystem::create_directories(directory);
  const std::filesystem::path image_path = directory / "target.png";
  if (!cv::imwrite(image_path.string(),
                   RenderTarget(model, camera, vertex_albedo, kTrueDelta))) {
    std::cerr << "Could not write the synthetic target image\n";
    return 1;
  }

  face_recon::FittingResult fitting_result;
  fitting_result.usable = true;
  fitting_result.camera = camera;
  fitting_result.shape_coefficients = Eigen::VectorXd::Zero(1);
  fitting_result.expression_coefficients = Eigen::VectorXd::Zero(1);
  fitting_result.vertices = face_recon::GenerateVertices(
      model, fitting_result.shape_coefficients,
      fitting_result.expression_coefficients);

  face_recon::PhotometricResult photometric_result;
  photometric_result.usable = true;
  photometric_result.fitted_vertex_albedo = vertex_albedo;
  photometric_result.illumination.setZero();
  photometric_result.illumination.col(0).setOnes();  // ambient light only

  face_recon::DenseFittingOptions options;
  options.num_expression_coefficients = 1;
  options.pixel_stride = 2;
  options.mask_erosion = 1;
  options.outer_iterations = 6;
  options.blurred_iterations = 3;
  options.blur_sigma = 3.0;
  options.solver_iterations = 20;
  options.landmark_weight = 0.0;  // no anchors: colors must do all the work
  options.expression_regularization = 1.0e-3;
  options.huber_delta = 0.5;
  // A rigid x-shift is indistinguishable from camera translation; hold the
  // pose so the expression coefficient has to explain the motion.
  options.optimize_pose = false;

  const face_recon::DenseFittingResult result =
      face_recon::RefineExpressionDense(image_path.string(), model,
                                        fitting_result, photometric_result,
                                        options, directory.string());

  std::cout << "samples: " << result.sample_count
            << ", photometric RMSE: " << result.initial_photometric_rmse
            << " -> " << result.final_photometric_rmse
            << ", delta: " << result.expression_coefficients[0]
            << " (expected " << kTrueDelta << ")\n";

  if (!result.usable) {
    std::cerr << "Dense refinement result was not usable\n"
              << result.summary << "\n";
    return 1;
  }
  if (result.final_photometric_rmse >= 0.5 * result.initial_photometric_rmse) {
    std::cerr << "Photometric error did not drop enough\n";
    return 1;
  }
  if (std::abs(result.expression_coefficients[0] - kTrueDelta) > 0.15) {
    std::cerr << "Recovered expression coefficient is off: "
              << result.expression_coefficients[0] << "\n";
    return 1;
  }
  std::cout << "Dense fitting test passed\n";
  return 0;
}
