#include "face_recon/photometric_fitting.h"

#include <ceres/rotation.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace face_recon {
namespace {

struct PixelSample {
  int x = 0;
  int y = 0;
  Eigen::Vector3d observed = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d mean_albedo = Eigen::Vector3d::Zero();
  Eigen::MatrixXd albedo_basis;
};

int TriangleCount(const Eigen::MatrixXi& triangles) {
  return triangles.cols() == 3 ? triangles.rows() : triangles.cols();
}

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

double ColorScale(const PcaComponent& color) {
  return color.mean.size() > 0 && color.mean.maxCoeff() > 1.05 ? 1.0 / 255.0
                                                               : 1.0;
}

Eigen::Vector4d ShBasis(const Eigen::Vector3d& normal) {
  return Eigen::Vector4d(1.0, normal.x(), normal.y(), normal.z());
}

Eigen::Vector3d EvaluateLighting(
    const Eigen::Vector3d& normal,
    const Eigen::Matrix<double, 3, 4>& illumination) {
  return illumination * ShBasis(normal);
}

double HuberWeight(double residual_norm, double delta) {
  if (delta <= 0.0 || residual_norm <= delta) {
    return 1.0;
  }
  return delta / std::max(residual_norm, 1.0e-12);
}

Eigen::Vector3d InterpolateVector(const Eigen::VectorXd& values,
                                  int i0,
                                  int i1,
                                  int i2,
                                  const Eigen::Vector3f& barycentric,
                                  double scale = 1.0) {
  return scale *
         (static_cast<double>(barycentric[0]) * values.segment<3>(3 * i0) +
          static_cast<double>(barycentric[1]) * values.segment<3>(3 * i1) +
          static_cast<double>(barycentric[2]) * values.segment<3>(3 * i2));
}

Eigen::MatrixXd InterpolateBasis(const Eigen::MatrixXd& basis,
                                 int i0,
                                 int i1,
                                 int i2,
                                 const Eigen::Vector3f& barycentric,
                                 int count,
                                 double scale) {
  return scale *
         (static_cast<double>(barycentric[0]) *
              basis.block(3 * i0, 0, 3, count) +
          static_cast<double>(barycentric[1]) *
              basis.block(3 * i1, 0, 3, count) +
          static_cast<double>(barycentric[2]) *
              basis.block(3 * i2, 0, 3, count));
}

// Removes externally occluded pixels (hair, hands, clothes, accessories)
// from a sampling mask. The occluder mask is dilated a little because
// segmentation boundaries are imprecise and mixed border pixels would
// otherwise still leak occluder colors into the fit.
void RemoveOccludedPixels(cv::Mat* mask, const std::string& occluder_mask_path) {
  if (occluder_mask_path.empty()) {
    return;
  }
  cv::Mat occluder = cv::imread(occluder_mask_path, cv::IMREAD_GRAYSCALE);
  if (occluder.empty()) {
    throw std::runtime_error("Could not read occluder mask: " +
                             occluder_mask_path);
  }
  if (occluder.size() != mask->size()) {
    cv::resize(occluder, occluder, mask->size(), 0.0, 0.0, cv::INTER_NEAREST);
  }
  cv::threshold(occluder, occluder, 127, 255, cv::THRESH_BINARY);
  const int radius = std::max(
      2, static_cast<int>(std::lround(0.01 * std::max(mask->cols, mask->rows))));
  cv::dilate(occluder, occluder,
             cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                       cv::Size(2 * radius + 1, 2 * radius + 1)));
  mask->setTo(0, occluder);
}

cv::Mat BuildMask(const RasterizationResult& rasterization, int erosion,
                  const std::string& occluder_mask_path) {
  cv::Mat mask(rasterization.height, rasterization.width, CV_8UC1,
               cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (rasterization.at(x, y).visible()) {
        mask.at<unsigned char>(y, x) = 255;
      }
    }
  }
  RemoveOccludedPixels(&mask, occluder_mask_path);
  if (erosion > 0) {
    const int size = 2 * erosion + 1;
    cv::erode(mask, mask,
              cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(size, size)));
  }
  return mask;
}

std::vector<PixelSample> BuildSamples(
    const cv::Mat& image,
    const cv::Mat& mask,
    const BfmModel& model,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& camera_normals,
    int num_albedo,
    int pixel_stride) {
  const double color_scale = ColorScale(model.color());
  Eigen::MatrixXd scaled_basis =
      model.color().pca_basis.leftCols(num_albedo);
  for (int index = 0; index < num_albedo; ++index) {
    scaled_basis.col(index) *=
        std::sqrt(std::max(model.color().pca_variance[index], 1.0e-12));
  }

  std::vector<PixelSample> samples;
  const int stride = std::max(pixel_stride, 1);
  samples.reserve(static_cast<std::size_t>(
      rasterization.width * rasterization.height / (stride * stride * 4)));
  for (int y = 0; y < rasterization.height; y += stride) {
    for (int x = 0; x < rasterization.width; x += stride) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = TriangleIndex(model.triangles(), pixel.triangle_id, 0);
      const int i1 = TriangleIndex(model.triangles(), pixel.triangle_id, 1);
      const int i2 = TriangleIndex(model.triangles(), pixel.triangle_id, 2);

      PixelSample sample;
      sample.x = x;
      sample.y = y;
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      sample.observed =
          Eigen::Vector3d(bgr[2], bgr[1], bgr[0]) / 255.0;
      sample.normal =
          InterpolateVector(camera_normals, i0, i1, i2, pixel.barycentric);
      const double normal_length = sample.normal.norm();
      if (normal_length <= 1.0e-9) {
        continue;
      }
      sample.normal /= normal_length;
      sample.mean_albedo =
          InterpolateVector(model.color().mean, i0, i1, i2,
                            pixel.barycentric, color_scale);
      sample.albedo_basis =
          InterpolateBasis(scaled_basis, i0, i1, i2, pixel.barycentric,
                           num_albedo, color_scale);
      samples.push_back(std::move(sample));
    }
  }
  return samples;
}

Eigen::Vector3d SampleAlbedo(const PixelSample& sample,
                             const Eigen::VectorXd& coefficients) {
  return sample.mean_albedo + sample.albedo_basis * coefficients;
}

Eigen::Vector3d RenderSample(
    const PixelSample& sample,
    const Eigen::VectorXd& coefficients,
    const Eigen::Matrix<double, 3, 4>& illumination) {
  return SampleAlbedo(sample, coefficients)
      .cwiseProduct(EvaluateLighting(sample.normal, illumination));
}

std::vector<double> RobustWeights(
    const std::vector<PixelSample>& samples,
    const Eigen::VectorXd& coefficients,
    const Eigen::Matrix<double, 3, 4>& illumination,
    double huber_delta) {
  std::vector<double> weights;
  weights.reserve(samples.size());
  for (const PixelSample& sample : samples) {
    weights.push_back(HuberWeight(
        (RenderSample(sample, coefficients, illumination) - sample.observed)
            .norm(),
        huber_delta));
  }
  return weights;
}

Eigen::Matrix<double, 3, 4> FitIllumination(
    const std::vector<PixelSample>& samples,
    const Eigen::VectorXd& coefficients,
    Eigen::Matrix<double, 3, 4> illumination,
    const PhotometricOptions& options) {
  for (int iteration = 0;
       iteration < std::max(options.illumination_iterations, 1);
       ++iteration) {
    const std::vector<double> weights =
        RobustWeights(samples, coefficients, illumination, options.huber_delta);
    for (int channel = 0; channel < 3; ++channel) {
      Eigen::Matrix4d normal_matrix =
          options.illumination_regularization * Eigen::Matrix4d::Identity();
      Eigen::Vector4d right_hand_side = Eigen::Vector4d::Zero();
      for (int index = 0; index < static_cast<int>(samples.size()); ++index) {
        const PixelSample& sample = samples[index];
        const double albedo =
            std::clamp(SampleAlbedo(sample, coefficients)[channel], 0.02, 1.5);
        const Eigen::Vector4d row = albedo * ShBasis(sample.normal);
        normal_matrix.noalias() += weights[index] * row * row.transpose();
        right_hand_side.noalias() +=
            weights[index] * row * sample.observed[channel];
      }
      illumination.row(channel) =
          normal_matrix.ldlt().solve(right_hand_side).transpose();
    }
  }
  return illumination;
}

Eigen::VectorXd FitAlbedo(
    const std::vector<PixelSample>& samples,
    const Eigen::Matrix<double, 3, 4>& illumination,
    const Eigen::VectorXd& initial,
    const PhotometricOptions& options) {
  const int count = static_cast<int>(initial.size());
  Eigen::VectorXd coefficients = initial;
  const std::vector<double> weights =
      RobustWeights(samples, coefficients, illumination, options.huber_delta);
  Eigen::MatrixXd normal_matrix =
      options.albedo_regularization *
      Eigen::MatrixXd::Identity(count, count);
  Eigen::VectorXd right_hand_side = Eigen::VectorXd::Zero(count);

  for (int index = 0; index < static_cast<int>(samples.size()); ++index) {
    const PixelSample& sample = samples[index];
    const Eigen::Vector3d lighting =
        EvaluateLighting(sample.normal, illumination);
    for (int channel = 0; channel < 3; ++channel) {
      if (lighting[channel] <= 0.02) {
        continue;
      }
      const Eigen::VectorXd row =
          lighting[channel] * sample.albedo_basis.row(channel).transpose();
      const double target =
          sample.observed[channel] -
          lighting[channel] * sample.mean_albedo[channel];
      const double weight = weights[index];
      normal_matrix.noalias() += weight * row * row.transpose();
      right_hand_side.noalias() += weight * row * target;
    }
  }

  coefficients = normal_matrix.ldlt().solve(right_hand_side);
  for (int index = 0; index < count; ++index) {
    coefficients[index] = std::clamp(coefficients[index], -3.0, 3.0);
  }
  return coefficients;
}

double ComputeRmse(
    const std::vector<PixelSample>& samples,
    const Eigen::VectorXd& coefficients,
    const Eigen::Matrix<double, 3, 4>& illumination) {
  if (samples.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  double squared_error = 0.0;
  for (const PixelSample& sample : samples) {
    squared_error +=
        (RenderSample(sample, coefficients, illumination) - sample.observed)
            .squaredNorm();
  }
  return std::sqrt(squared_error /
                   (3.0 * static_cast<double>(samples.size())));
}

Eigen::VectorXd GenerateVertexAlbedo(const BfmModel& model,
                                     const Eigen::VectorXd& coefficients) {
  Eigen::VectorXd scaled = coefficients;
  for (int index = 0; index < scaled.size(); ++index) {
    scaled[index] *=
        std::sqrt(std::max(model.color().pca_variance[index], 1.0e-12));
  }
  Eigen::VectorXd albedo =
      model.color().mean +
      model.color().pca_basis.leftCols(coefficients.size()) * scaled;
  albedo *= ColorScale(model.color());
  return albedo.cwiseMax(0.0).cwiseMin(1.0);
}

Eigen::Vector3d PixelAlbedo(const BfmModel& model,
                            const RasterPixel& pixel,
                            const Eigen::VectorXd& vertex_albedo) {
  const int i0 = TriangleIndex(model.triangles(), pixel.triangle_id, 0);
  const int i1 = TriangleIndex(model.triangles(), pixel.triangle_id, 1);
  const int i2 = TriangleIndex(model.triangles(), pixel.triangle_id, 2);
  return InterpolateVector(vertex_albedo, i0, i1, i2, pixel.barycentric);
}

Eigen::Vector3d PixelNormal(const BfmModel& model,
                            const RasterPixel& pixel,
                            const Eigen::VectorXd& camera_normals) {
  const int i0 = TriangleIndex(model.triangles(), pixel.triangle_id, 0);
  const int i1 = TriangleIndex(model.triangles(), pixel.triangle_id, 1);
  const int i2 = TriangleIndex(model.triangles(), pixel.triangle_id, 2);
  Eigen::Vector3d normal =
      InterpolateVector(camera_normals, i0, i1, i2, pixel.barycentric);
  return normal.norm() > 1.0e-9 ? normal.normalized()
                                : Eigen::Vector3d::UnitZ();
}

cv::Vec3b ToBgr8(const Eigen::Vector3d& rgb) {
  const Eigen::Vector3d value = rgb.cwiseMax(0.0).cwiseMin(1.0) * 255.0;
  return cv::Vec3b(static_cast<unsigned char>(std::lround(value[2])),
                   static_cast<unsigned char>(std::lround(value[1])),
                   static_cast<unsigned char>(std::lround(value[0])));
}

bool SaveDiagnosticImages(
    const cv::Mat& input,
    const cv::Mat& mask,
    const BfmModel& model,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& camera_normals,
    const Eigen::VectorXd& fitted_vertex_albedo,
    const Eigen::Matrix<double, 3, 4>& initial_illumination,
    const Eigen::Matrix<double, 3, 4>& illumination_only,
    const Eigen::Matrix<double, 3, 4>& final_illumination,
    const std::string& output_directory) {
  cv::Mat mean_albedo(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat estimated_albedo = mean_albedo.clone();
  cv::Mat camera_normal = mean_albedo.clone();
  cv::Mat rendered_initial = mean_albedo.clone();
  cv::Mat rendered_illumination = mean_albedo.clone();
  cv::Mat rendered_final = mean_albedo.clone();
  cv::Mat rendered_overlay = input.clone();
  cv::Mat residual = mean_albedo.clone();
  const Eigen::VectorXd mean_vertex_albedo =
      (ColorScale(model.color()) * model.color().mean)
          .cwiseMax(0.0)
          .cwiseMin(1.0);

  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const Eigen::Vector3d normal =
          PixelNormal(model, pixel, camera_normals);
      const Eigen::Vector3d mean =
          PixelAlbedo(model, pixel, mean_vertex_albedo);
      const Eigen::Vector3d fitted =
          PixelAlbedo(model, pixel, fitted_vertex_albedo);
      const Eigen::Vector3d initial =
          mean.cwiseProduct(EvaluateLighting(normal, initial_illumination));
      const Eigen::Vector3d lit_mean =
          mean.cwiseProduct(EvaluateLighting(normal, illumination_only));
      const Eigen::Vector3d final =
          fitted.cwiseProduct(EvaluateLighting(normal, final_illumination));
      const cv::Vec3b observed = input.at<cv::Vec3b>(y, x);
      const Eigen::Vector3d observed_rgb(observed[2] / 255.0,
                                         observed[1] / 255.0,
                                         observed[0] / 255.0);

      mean_albedo.at<cv::Vec3b>(y, x) = ToBgr8(mean);
      estimated_albedo.at<cv::Vec3b>(y, x) = ToBgr8(fitted);
      camera_normal.at<cv::Vec3b>(y, x) =
          ToBgr8(0.5 * (normal + Eigen::Vector3d::Ones()));
      rendered_initial.at<cv::Vec3b>(y, x) = ToBgr8(initial);
      rendered_illumination.at<cv::Vec3b>(y, x) = ToBgr8(lit_mean);
      rendered_final.at<cv::Vec3b>(y, x) = ToBgr8(final);
      rendered_overlay.at<cv::Vec3b>(y, x) = ToBgr8(final);
      residual.at<cv::Vec3b>(y, x) =
          ToBgr8((final - observed_rgb).cwiseAbs());
    }
  }

  const std::filesystem::path directory(output_directory);
  std::filesystem::create_directories(directory);
  return cv::imwrite((directory / "photometric_mask.png").string(), mask) &&
         cv::imwrite((directory / "mean_albedo_render.png").string(),
                     mean_albedo) &&
         cv::imwrite((directory / "estimated_albedo.png").string(),
                     estimated_albedo) &&
         cv::imwrite((directory / "camera_normal.png").string(),
                     camera_normal) &&
         cv::imwrite((directory / "rendered_initial.png").string(),
                     rendered_initial) &&
         cv::imwrite((directory / "rendered_illumination.png").string(),
                     rendered_illumination) &&
         cv::imwrite((directory / "rendered_intrinsic.png").string(),
                     rendered_final) &&
         cv::imwrite((directory / "rendered_intrinsic_overlay.png").string(),
                     rendered_overlay) &&
         cv::imwrite((directory / "photometric_residual.png").string(),
                     residual);
}

}  // namespace

Eigen::VectorXd ComputeVertexNormals(const Eigen::VectorXd& vertices,
                                     const Eigen::MatrixXi& triangles) {
  if (vertices.size() == 0 || vertices.size() % 3 != 0) {
    throw std::runtime_error("Cannot compute normals for invalid vertices");
  }
  const int vertex_count = static_cast<int>(vertices.size() / 3);
  Eigen::VectorXd normals = Eigen::VectorXd::Zero(vertices.size());
  for (int triangle = 0; triangle < TriangleCount(triangles); ++triangle) {
    const int i0 = TriangleIndex(triangles, triangle, 0);
    const int i1 = TriangleIndex(triangles, triangle, 1);
    const int i2 = TriangleIndex(triangles, triangle, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
      throw std::runtime_error("Invalid triangle index while computing normals");
    }
    const Eigen::Vector3d v0 = vertices.segment<3>(3 * i0);
    const Eigen::Vector3d v1 = vertices.segment<3>(3 * i1);
    const Eigen::Vector3d v2 = vertices.segment<3>(3 * i2);
    const Eigen::Vector3d face_normal = (v1 - v0).cross(v2 - v0);
    normals.segment<3>(3 * i0) += face_normal;
    normals.segment<3>(3 * i1) += face_normal;
    normals.segment<3>(3 * i2) += face_normal;
  }
  double mean_z = 0.0;
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    Eigen::Vector3d normal = normals.segment<3>(3 * vertex);
    if (normal.norm() > 1.0e-12) {
      normal.normalize();
    } else {
      normal = Eigen::Vector3d::UnitZ();
    }
    normals.segment<3>(3 * vertex) = normal;
    mean_z += normal.z();
  }
  if (mean_z < 0.0) {
    normals = -normals;
  }
  return normals;
}

PhotometricResult FitPhotometricAppearance(
    const std::string& image_path,
    const BfmModel& model,
    const Eigen::VectorXd& vertices,
    const CameraParameters& camera,
    const RasterizationResult& rasterization,
    const PhotometricOptions& options,
    const std::string& output_directory) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read photometric input: " + image_path);
  }
  if (image.cols != rasterization.width ||
      image.rows != rasterization.height) {
    throw std::runtime_error(
        "Photometric input dimensions do not match rasterization");
  }
  const int num_albedo = std::clamp(
      options.num_albedo_coefficients, 1,
      static_cast<int>(std::min(model.color().pca_basis.cols(),
                                model.color().pca_variance.size())));

  Eigen::VectorXd camera_normals =
      ComputeVertexNormals(vertices, model.triangles());
  for (int vertex = 0; vertex < camera_normals.size() / 3; ++vertex) {
    double rotated[3];
    ceres::AngleAxisRotatePoint(camera.angle_axis.data(),
                                camera_normals.data() + 3 * vertex, rotated);
    Eigen::Vector3d normal(rotated[0], rotated[1], rotated[2]);
    camera_normals.segment<3>(3 * vertex) =
        normal.norm() > 1.0e-12 ? normal.normalized()
                                : Eigen::Vector3d::UnitZ();
  }

  const cv::Mat mask = BuildMask(rasterization, options.mask_erosion,
                                 options.occluder_mask_path);
  const std::vector<PixelSample> samples =
      BuildSamples(image, mask, model, rasterization, camera_normals,
                   num_albedo, options.pixel_stride);
  if (samples.size() < 100) {
    throw std::runtime_error(
        "Too few visible pixels for photometric fitting");
  }

  PhotometricResult result;
  result.sample_count = static_cast<int>(samples.size());
  result.albedo_coefficients = Eigen::VectorXd::Zero(num_albedo);
  Eigen::Matrix<double, 3, 4> initial_illumination =
      Eigen::Matrix<double, 3, 4>::Zero();
  initial_illumination.col(0).setOnes();
  result.initial_rmse =
      ComputeRmse(samples, result.albedo_coefficients, initial_illumination);

  Eigen::Matrix<double, 3, 4> illumination_only =
      FitIllumination(samples, result.albedo_coefficients,
                      initial_illumination, options);
  result.illumination_rmse =
      ComputeRmse(samples, result.albedo_coefficients, illumination_only);
  result.illumination = illumination_only;

  for (int iteration = 0; iteration < std::max(options.joint_iterations, 1);
       ++iteration) {
    result.albedo_coefficients =
        FitAlbedo(samples, result.illumination, result.albedo_coefficients,
                  options);
    result.illumination =
        FitIllumination(samples, result.albedo_coefficients,
                        result.illumination, options);
  }
  result.final_rmse =
      ComputeRmse(samples, result.albedo_coefficients, result.illumination);
  result.fitted_vertex_albedo =
      GenerateVertexAlbedo(model, result.albedo_coefficients);
  result.usable = std::isfinite(result.final_rmse) &&
                  result.final_rmse <= result.initial_rmse;

  if (options.save_diagnostics &&
      !SaveDiagnosticImages(
          image, mask, model, rasterization, camera_normals,
          result.fitted_vertex_albedo, initial_illumination,
          illumination_only, result.illumination, output_directory)) {
    throw std::runtime_error("Could not save photometric diagnostic images");
  }
  std::cout << "[RESULT] Photometric RMSE: " << result.initial_rmse
            << " -> " << result.illumination_rmse
            << " -> " << result.final_rmse << "\n";
  return result;
}

bool SavePhotometricReport(const PhotometricResult& result,
                           const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    return false;
  }
  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "sample_count: " << result.sample_count << "\n";
  output << "initial_rmse_rgb: " << result.initial_rmse << "\n";
  output << "illumination_rmse_rgb: " << result.illumination_rmse << "\n";
  output << "final_rmse_rgb: " << result.final_rmse << "\n";
  for (int channel = 0; channel < 3; ++channel) {
    output << "illumination[" << channel << "]:";
    for (int coefficient = 0; coefficient < 4; ++coefficient) {
      output << " " << result.illumination(channel, coefficient);
    }
    output << "\n";
  }
  output << "albedo_coefficients: "
         << result.albedo_coefficients.size() << "\n";
  for (int index = 0; index < result.albedo_coefficients.size(); ++index) {
    output << "albedo[" << index << "]: "
           << result.albedo_coefficients[index] << "\n";
  }
  return true;
}

}  // namespace face_recon
