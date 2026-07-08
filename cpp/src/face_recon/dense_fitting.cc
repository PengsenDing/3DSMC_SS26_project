#include "face_recon/dense_fitting.h"

#include "face_recon/rasterizer.h"

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace face_recon {
namespace {

using ImageGrid = ceres::Grid2D<double, 3>;
using ImageInterpolator = ceres::BiCubicInterpolator<ImageGrid>;

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

std::array<double, 7> CameraArray(const CameraParameters& camera) {
  return {camera.angle_axis.x(), camera.angle_axis.y(), camera.angle_axis.z(),
          camera.translation.x(), camera.translation.y(),
          std::log(camera.translation.z()), std::log(camera.focal_length)};
}

CameraParameters CameraParametersFromArray(const std::array<double, 7>& camera,
                                           double aspect_ratio) {
  CameraParameters result;
  result.angle_axis = Eigen::Vector3d(camera[0], camera[1], camera[2]);
  result.translation =
      Eigen::Vector3d(camera[3], camera[4], std::exp(camera[5]));
  result.focal_length = std::exp(camera[6]);
  result.aspect_ratio = aspect_ratio;
  return result;
}

Eigen::Vector4d ShBasis(const Eigen::Vector3d& normal) {
  return Eigen::Vector4d(1.0, normal.x(), normal.y(), normal.z());
}

// One color observation anchored to a fixed surface point (triangle +
// perspective-correct barycentric coordinates from the last rasterization).
struct DenseSample {
  // Model-space position with identity and the frozen high-order expression
  // coefficients baked in: position(delta) = base + basis * delta.
  Eigen::Vector3d base = Eigen::Vector3d::Zero();
  Eigen::MatrixXd basis;  // 3 x K
  // Shaded model color (albedo * SH lighting), frozen per outer iteration.
  Eigen::Vector3d shading = Eigen::Vector3d::Zero();
  Eigen::Vector3d observed = Eigen::Vector3d::Zero();
};

// Anchor from the sparse stage: keeps the dense term from dragging the face
// off the detected semantic features.
struct DenseAnchor {
  Eigen::Vector3d base = Eigen::Vector3d::Zero();
  Eigen::MatrixXd basis;  // 3 x K
  Eigen::Vector2d target = Eigen::Vector2d::Zero();
  double weight = 1.0;
};

template <typename T>
bool ProjectModelPoint(const T* camera, const T point[3], double aspect_ratio,
                       T* projected_x, T* projected_y) {
  T rotated[3];
  ceres::AngleAxisRotatePoint(camera, point, rotated);
  using std::exp;
  const T translation_z = exp(camera[5]);
  const T focal_length = exp(camera[6]);
  const T depth = translation_z - rotated[2];
  if (depth <= T(1.0e-6)) {
    return false;
  }
  *projected_x = T(0.5) + focal_length * (rotated[0] + camera[3]) / depth;
  *projected_y = T(0.5) - focal_length * T(aspect_ratio) *
                              (rotated[1] + camera[4]) / depth;
  return true;
}

class DensePhotometricResidual {
 public:
  DensePhotometricResidual(const DenseSample& sample,
                           const ImageInterpolator* image,
                           double aspect_ratio,
                           int width,
                           int height)
      : sample_(sample),
        image_(image),
        aspect_ratio_(aspect_ratio),
        width_(width),
        height_(height) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* camera = parameters[0];
    const T* expression = parameters[1];

    T point[3] = {T(sample_.base.x()), T(sample_.base.y()),
                  T(sample_.base.z())};
    for (int column = 0; column < sample_.basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] += T(sample_.basis(axis, column)) * expression[column];
      }
    }

    T projected_x;
    T projected_y;
    if (!ProjectModelPoint(camera, point, aspect_ratio_, &projected_x,
                           &projected_y)) {
      return false;
    }
    // Raster pixel (x, y) stores the color of the area whose center is
    // (x + 0.5, y + 0.5) in projected coordinates, hence the half-pixel
    // shift into image index space.
    const T column = projected_x * T(width_ - 1) - T(0.5);
    const T row = projected_y * T(height_ - 1) - T(0.5);
    T sampled[3];
    image_->Evaluate(row, column, sampled);
    for (int channel = 0; channel < 3; ++channel) {
      residuals[channel] = T(sample_.shading[channel]) - sampled[channel];
    }
    return true;
  }

 private:
  DenseSample sample_;
  const ImageInterpolator* image_;
  double aspect_ratio_;
  int width_;
  int height_;
};

class DenseAnchorResidual {
 public:
  DenseAnchorResidual(DenseAnchor anchor, double aspect_ratio)
      : anchor_(std::move(anchor)),
        aspect_ratio_(aspect_ratio),
        sqrt_weight_(std::sqrt(anchor_.weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* camera = parameters[0];
    const T* expression = parameters[1];

    T point[3] = {T(anchor_.base.x()), T(anchor_.base.y()),
                  T(anchor_.base.z())};
    for (int column = 0; column < anchor_.basis.cols(); ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        point[axis] += T(anchor_.basis(axis, column)) * expression[column];
      }
    }

    T projected_x;
    T projected_y;
    if (!ProjectModelPoint(camera, point, aspect_ratio_, &projected_x,
                           &projected_y)) {
      return false;
    }
    residuals[0] = T(sqrt_weight_) * (projected_x - T(anchor_.target.x()));
    residuals[1] = T(sqrt_weight_) * (projected_y - T(anchor_.target.y()));
    return true;
  }

 private:
  DenseAnchor anchor_;
  double aspect_ratio_;
  double sqrt_weight_;
};

class ExpressionPrior {
 public:
  ExpressionPrior(int count, double weight)
      : count_(count), sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    for (int index = 0; index < count_; ++index) {
      residuals[index] = T(sqrt_weight_) * parameters[0][index];
    }
    return true;
  }

 private:
  int count_;
  double sqrt_weight_;
};

// Same policy as the appearance stage: visible raster pixels, minus dilated
// external occluders, eroded a little so mixed border pixels do not leak
// background colors into the fit.
cv::Mat BuildSamplingMask(const RasterizationResult& rasterization,
                          int erosion,
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
  if (!occluder_mask_path.empty()) {
    cv::Mat occluder = cv::imread(occluder_mask_path, cv::IMREAD_GRAYSCALE);
    if (occluder.empty()) {
      throw std::runtime_error("Could not read occluder mask: " +
                               occluder_mask_path);
    }
    if (occluder.size() != mask.size()) {
      cv::resize(occluder, occluder, mask.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    cv::threshold(occluder, occluder, 127, 255, cv::THRESH_BINARY);
    const int radius = std::max(
        2,
        static_cast<int>(std::lround(0.01 * std::max(mask.cols, mask.rows))));
    cv::dilate(occluder, occluder,
               cv::getStructuringElement(
                   cv::MORPH_ELLIPSE,
                   cv::Size(2 * radius + 1, 2 * radius + 1)));
    mask.setTo(0, occluder);
  }
  if (erosion > 0) {
    const int size = 2 * erosion + 1;
    cv::erode(mask, mask,
              cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(size, size)));
  }
  return mask;
}

Eigen::Vector3d InterpolateVector3(const Eigen::VectorXd& values,
                                   int i0,
                                   int i1,
                                   int i2,
                                   const Eigen::Vector3f& barycentric) {
  return static_cast<double>(barycentric[0]) * values.segment<3>(3 * i0) +
         static_cast<double>(barycentric[1]) * values.segment<3>(3 * i1) +
         static_cast<double>(barycentric[2]) * values.segment<3>(3 * i2);
}

std::vector<DenseSample> BuildDenseSamples(
    const cv::Mat& image,
    const cv::Mat& mask,
    const Eigen::MatrixXi& triangles,
    const RasterizationResult& rasterization,
    const Eigen::VectorXd& base_vertices,
    const Eigen::MatrixXd& scaled_expression_basis,
    const Eigen::VectorXd& camera_normals,
    const Eigen::VectorXd& vertex_albedo,
    const Eigen::Matrix<double, 3, 4>& illumination,
    int pixel_stride) {
  const int num_expression =
      static_cast<int>(scaled_expression_basis.cols());
  std::vector<DenseSample> samples;
  const int stride = std::max(pixel_stride, 1);
  samples.reserve(static_cast<std::size_t>(rasterization.width) *
                  rasterization.height / (stride * stride * 4));
  for (int y = 0; y < rasterization.height; y += stride) {
    for (int x = 0; x < rasterization.width; x += stride) {
      if (mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = TriangleIndex(triangles, pixel.triangle_id, 0);
      const int i1 = TriangleIndex(triangles, pixel.triangle_id, 1);
      const int i2 = TriangleIndex(triangles, pixel.triangle_id, 2);

      Eigen::Vector3d normal =
          InterpolateVector3(camera_normals, i0, i1, i2, pixel.barycentric);
      const double normal_length = normal.norm();
      if (normal_length <= 1.0e-9) {
        continue;
      }
      normal /= normal_length;

      DenseSample sample;
      sample.base =
          InterpolateVector3(base_vertices, i0, i1, i2, pixel.barycentric);
      sample.basis =
          static_cast<double>(pixel.barycentric[0]) *
              scaled_expression_basis.block(3 * i0, 0, 3, num_expression) +
          static_cast<double>(pixel.barycentric[1]) *
              scaled_expression_basis.block(3 * i1, 0, 3, num_expression) +
          static_cast<double>(pixel.barycentric[2]) *
              scaled_expression_basis.block(3 * i2, 0, 3, num_expression);
      const Eigen::Vector3d albedo =
          InterpolateVector3(vertex_albedo, i0, i1, i2, pixel.barycentric)
              .cwiseMax(0.0)
              .cwiseMin(1.0);
      sample.shading = albedo.cwiseProduct(illumination * ShBasis(normal));
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      sample.observed = Eigen::Vector3d(bgr[2], bgr[1], bgr[0]) / 255.0;
      samples.push_back(std::move(sample));
    }
  }
  return samples;
}

double SampleRmse(const std::vector<DenseSample>& samples) {
  if (samples.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  double squared_error = 0.0;
  for (const DenseSample& sample : samples) {
    squared_error += (sample.shading - sample.observed).squaredNorm();
  }
  return std::sqrt(squared_error / (3.0 * static_cast<double>(samples.size())));
}

double AnchorRmse(const std::vector<DenseAnchor>& anchors,
                  const std::array<double, 7>& camera,
                  double aspect_ratio,
                  const Eigen::VectorXd& delta) {
  if (anchors.empty()) {
    return 0.0;
  }
  double squared_error = 0.0;
  for (const DenseAnchor& anchor : anchors) {
    const Eigen::Vector3d point = anchor.base + anchor.basis * delta;
    double projected_x = 0.0;
    double projected_y = 0.0;
    ProjectModelPoint(camera.data(), point.data(), aspect_ratio,
                      &projected_x, &projected_y);
    squared_error +=
        (Eigen::Vector2d(projected_x, projected_y) - anchor.target)
            .squaredNorm();
  }
  return std::sqrt(squared_error / static_cast<double>(anchors.size()));
}

std::vector<double> ImageToRgbGrid(const cv::Mat& image) {
  std::vector<double> data(static_cast<std::size_t>(image.rows) *
                           image.cols * 3);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      const std::size_t offset =
          3 * (static_cast<std::size_t>(y) * image.cols + x);
      data[offset + 0] = bgr[2] / 255.0;
      data[offset + 1] = bgr[1] / 255.0;
      data[offset + 2] = bgr[0] / 255.0;
    }
  }
  return data;
}

cv::Vec3b ToBgr8(const Eigen::Vector3d& rgb) {
  const Eigen::Vector3d value = rgb.cwiseMax(0.0).cwiseMin(1.0) * 255.0;
  return cv::Vec3b(static_cast<unsigned char>(std::lround(value[2])),
                   static_cast<unsigned char>(std::lround(value[1])),
                   static_cast<unsigned char>(std::lround(value[0])));
}

bool SaveShadingOverlay(const cv::Mat& image,
                        const Eigen::MatrixXi& triangles,
                        const RasterizationResult& rasterization,
                        const Eigen::VectorXd& camera_normals,
                        const Eigen::VectorXd& vertex_albedo,
                        const Eigen::Matrix<double, 3, 4>& illumination,
                        const std::string& output_path) {
  cv::Mat overlay = image.clone();
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) {
        continue;
      }
      const int i0 = TriangleIndex(triangles, pixel.triangle_id, 0);
      const int i1 = TriangleIndex(triangles, pixel.triangle_id, 1);
      const int i2 = TriangleIndex(triangles, pixel.triangle_id, 2);
      Eigen::Vector3d normal =
          InterpolateVector3(camera_normals, i0, i1, i2, pixel.barycentric);
      if (normal.norm() <= 1.0e-9) {
        continue;
      }
      normal.normalize();
      const Eigen::Vector3d albedo =
          InterpolateVector3(vertex_albedo, i0, i1, i2, pixel.barycentric)
              .cwiseMax(0.0)
              .cwiseMin(1.0);
      overlay.at<cv::Vec3b>(y, x) =
          ToBgr8(albedo.cwiseProduct(illumination * ShBasis(normal)));
    }
  }
  return cv::imwrite(output_path, overlay);
}

}  // namespace

DenseFittingResult RefineExpressionDense(
    const std::string& image_path,
    const BfmModel& model,
    const FittingResult& fitting_result,
    const PhotometricResult& photometric_result,
    const DenseFittingOptions& options,
    const std::string& output_directory) {
  const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Could not read dense-fitting input: " +
                             image_path);
  }
  const int width = image.cols;
  const int height = image.rows;
  const double aspect_ratio = fitting_result.camera.aspect_ratio;

  DenseFittingResult result;
  result.camera = fitting_result.camera;
  result.expression_coefficients = fitting_result.expression_coefficients;
  result.vertices = fitting_result.vertices;

  const int total_expression =
      static_cast<int>(fitting_result.expression_coefficients.size());
  const int num_expression =
      std::clamp(options.num_expression_coefficients, 1, total_expression);

  if (photometric_result.fitted_vertex_albedo.size() !=
      fitting_result.vertices.size()) {
    result.summary = "skipped: vertex albedo unavailable";
    return result;
  }

  // Everything the optimized coefficients do not control is baked into a
  // constant base: identity, both model means, and the frozen high-order
  // expression coefficients.
  Eigen::VectorXd frozen_expression = fitting_result.expression_coefficients;
  frozen_expression.head(num_expression).setZero();
  const Eigen::VectorXd base_vertices = GenerateVertices(
      model, fitting_result.shape_coefficients, frozen_expression);

  Eigen::MatrixXd scaled_expression_basis =
      model.expression().pca_basis.leftCols(num_expression);
  for (int index = 0; index < num_expression; ++index) {
    scaled_expression_basis.col(index) *=
        std::sqrt(std::max(model.expression().pca_variance[index], 1.0e-12));
  }

  std::vector<DenseAnchor> anchors;
  if (options.landmark_weight > 0.0) {
    anchors.reserve(fitting_result.reprojections.size());
    for (const LandmarkReprojection& reprojection :
         fitting_result.reprojections) {
      DenseAnchor anchor;
      anchor.base = base_vertices.segment<3>(3 * reprojection.bfm_vertex_id);
      anchor.basis = scaled_expression_basis.block(
          3 * reprojection.bfm_vertex_id, 0, 3, num_expression);
      anchor.target = reprojection.observed;
      anchor.weight =
          options.landmark_weight * reprojection.weight_multiplier;
      anchors.push_back(std::move(anchor));
    }
  }

  const std::vector<double> sharp_data = ImageToRgbGrid(image);
  const ImageGrid sharp_grid(sharp_data.data(), 0, height, 0, width);
  const ImageInterpolator sharp_interpolator(sharp_grid);
  cv::Mat blurred_image;
  std::vector<double> blurred_data;
  std::unique_ptr<ImageGrid> blurred_grid;
  std::unique_ptr<ImageInterpolator> blurred_interpolator;
  if (options.blurred_iterations > 0) {
    cv::GaussianBlur(image, blurred_image, cv::Size(0, 0),
                     std::max(options.blur_sigma, 0.1));
    blurred_data = ImageToRgbGrid(blurred_image);
    blurred_grid =
        std::make_unique<ImageGrid>(blurred_data.data(), 0, height, 0, width);
    blurred_interpolator =
        std::make_unique<ImageInterpolator>(*blurred_grid);
  }

  std::array<double, 7> camera = CameraArray(fitting_result.camera);
  Eigen::VectorXd delta =
      fitting_result.expression_coefficients.head(num_expression);

  result.initial_landmark_rmse =
      AnchorRmse(anchors, camera, aspect_ratio, delta);

  const auto rasterize_current = [&](const Eigen::VectorXd& coefficients) {
    const Eigen::VectorXd vertices =
        base_vertices + scaled_expression_basis * coefficients;
    return std::make_pair(
        vertices,
        RasterizeMesh(vertices, model.triangles(),
                      CameraParametersFromArray(camera, aspect_ratio), width,
                      height));
  };
  const auto shading_inputs = [&](const Eigen::VectorXd& vertices) {
    Eigen::VectorXd camera_normals =
        ComputeVertexNormals(vertices, model.triangles());
    return ApplyCameraRotation(
        camera_normals, CameraParametersFromArray(camera, aspect_ratio));
  };

  std::ostringstream summary;
  const int outer_iterations = std::max(options.outer_iterations, 1);
  for (int outer = 0; outer < outer_iterations; ++outer) {
    const auto [vertices, rasterization] = rasterize_current(delta);
    const Eigen::VectorXd camera_normals = shading_inputs(vertices);
    const cv::Mat mask = BuildSamplingMask(rasterization, options.mask_erosion,
                                           options.occluder_mask_path);
    const std::vector<DenseSample> samples = BuildDenseSamples(
        image, mask, model.triangles(), rasterization, base_vertices,
        scaled_expression_basis, camera_normals,
        photometric_result.fitted_vertex_albedo,
        photometric_result.illumination, options.pixel_stride);
    if (static_cast<int>(samples.size()) < 100) {
      result.summary = "skipped: too few visible pixels (" +
                       std::to_string(samples.size()) + ")";
      return result;
    }
    if (outer == 0) {
      result.initial_photometric_rmse = SampleRmse(samples);
      result.sample_count = static_cast<int>(samples.size());
      if (options.save_diagnostics) {
        SaveShadingOverlay(
            image, model.triangles(), rasterization, camera_normals,
            photometric_result.fitted_vertex_albedo,
            photometric_result.illumination,
            (std::filesystem::path(output_directory) /
             "dense_before_overlay.png")
                .string());
      }
    }

    const ImageInterpolator* interpolator =
        (outer < options.blurred_iterations && blurred_interpolator)
            ? blurred_interpolator.get()
            : &sharp_interpolator;

    ceres::Problem problem;
    for (const DenseSample& sample : samples) {
      auto* cost =
          new ceres::DynamicAutoDiffCostFunction<DensePhotometricResidual, 4>(
              new DensePhotometricResidual(sample, interpolator, aspect_ratio,
                                           width, height));
      cost->AddParameterBlock(7);
      cost->AddParameterBlock(num_expression);
      cost->SetNumResiduals(3);
      problem.AddResidualBlock(cost,
                               options.huber_delta > 0.0
                                   ? new ceres::HuberLoss(options.huber_delta)
                                   : nullptr,
                               camera.data(), delta.data());
    }
    for (const DenseAnchor& anchor : anchors) {
      auto* cost =
          new ceres::DynamicAutoDiffCostFunction<DenseAnchorResidual, 4>(
              new DenseAnchorResidual(anchor, aspect_ratio));
      cost->AddParameterBlock(7);
      cost->AddParameterBlock(num_expression);
      cost->SetNumResiduals(2);
      problem.AddResidualBlock(
          cost, new ceres::HuberLoss(0.01 * std::sqrt(anchor.weight)),
          camera.data(), delta.data());
    }
    if (options.expression_regularization > 0.0) {
      auto* cost =
          new ceres::DynamicAutoDiffCostFunction<ExpressionPrior, 4>(
              new ExpressionPrior(num_expression,
                                  options.expression_regularization));
      cost->AddParameterBlock(num_expression);
      cost->SetNumResiduals(num_expression);
      problem.AddResidualBlock(cost, nullptr, delta.data());
    }
    for (int index = 0; index < num_expression; ++index) {
      problem.SetParameterLowerBound(delta.data(), index, -3.0);
      problem.SetParameterUpperBound(delta.data(), index, 3.0);
    }
    if (options.optimize_pose) {
      // The focal length stays at its landmark-fit value: it is nearly
      // collinear with translation in z over a face-sized depth range.
      problem.SetManifold(camera.data(),
                          new ceres::SubsetManifold(7, {6}));
      problem.SetParameterLowerBound(camera.data(), 3, -500.0);
      problem.SetParameterUpperBound(camera.data(), 3, 500.0);
      problem.SetParameterLowerBound(camera.data(), 4, -500.0);
      problem.SetParameterUpperBound(camera.data(), 4, 500.0);
      problem.SetParameterLowerBound(camera.data(), 5, std::log(1.0));
      problem.SetParameterUpperBound(camera.data(), 5, std::log(5000.0));
    } else {
      problem.SetParameterBlockConstant(camera.data());
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = std::max(options.solver_iterations, 1);
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.minimizer_progress_to_stdout = options.verbose;
    solver_options.num_threads = std::max(
        1u, std::min(8u, std::thread::hardware_concurrency()));
    ceres::Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);
    summary << "outer " << (outer + 1) << " ("
            << (interpolator == &sharp_interpolator ? "sharp" : "blurred")
            << ", " << samples.size() << " samples): "
            << solver_summary.BriefReport() << "\n";
    if (!solver_summary.IsSolutionUsable()) {
      result.summary = summary.str() + "aborted: solver failure";
      return result;
    }
  }

  // Evaluate the refined parameters on a fresh rasterization so the reported
  // improvement includes the visibility and shading changes.
  const auto [final_vertices, final_rasterization] = rasterize_current(delta);
  const Eigen::VectorXd final_normals = shading_inputs(final_vertices);
  const cv::Mat final_mask = BuildSamplingMask(
      final_rasterization, options.mask_erosion, options.occluder_mask_path);
  const std::vector<DenseSample> final_samples = BuildDenseSamples(
      image, final_mask, model.triangles(), final_rasterization, base_vertices,
      scaled_expression_basis, final_normals,
      photometric_result.fitted_vertex_albedo, photometric_result.illumination,
      options.pixel_stride);
  result.final_photometric_rmse = SampleRmse(final_samples);
  result.final_landmark_rmse =
      AnchorRmse(anchors, camera, aspect_ratio, delta);
  if (options.save_diagnostics) {
    SaveShadingOverlay(image, model.triangles(), final_rasterization,
                       final_normals, photometric_result.fitted_vertex_albedo,
                       photometric_result.illumination,
                       (std::filesystem::path(output_directory) /
                        "dense_after_overlay.png")
                           .string());
  }

  result.expression_coefficients.head(num_expression) = delta;
  result.camera = CameraParametersFromArray(camera, aspect_ratio);
  result.vertices = final_vertices;

  // Accept only if the photometric error dropped and the landmarks did not
  // degrade beyond noise; otherwise the caller keeps the sparse-stage fit.
  const double landmark_limit = std::max(
      result.initial_landmark_rmse * 1.5,
      result.initial_landmark_rmse + 0.005);
  result.usable =
      std::isfinite(result.final_photometric_rmse) &&
      result.final_photometric_rmse < result.initial_photometric_rmse &&
      result.final_landmark_rmse <= landmark_limit;
  result.summary = summary.str();
  return result;
}

bool SaveDenseFittingReport(const DenseFittingResult& result,
                            const std::string& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    return false;
  }
  output << std::setprecision(10);
  output << "solution_usable: " << (result.usable ? "yes" : "no") << "\n";
  output << "sample_count: " << result.sample_count << "\n";
  output << "initial_photometric_rmse_rgb: "
         << result.initial_photometric_rmse << "\n";
  output << "final_photometric_rmse_rgb: " << result.final_photometric_rmse
         << "\n";
  output << "initial_landmark_rmse: " << result.initial_landmark_rmse << "\n";
  output << "final_landmark_rmse: " << result.final_landmark_rmse << "\n";
  output << "expression_coefficients: "
         << result.expression_coefficients.size() << "\n";
  for (int index = 0; index < result.expression_coefficients.size();
       ++index) {
    output << "expression[" << index << "]: "
           << result.expression_coefficients[index] << "\n";
  }
  output << "solver_summary:\n" << result.summary;
  return true;
}

}  // namespace face_recon
