// Face2Face-style expression transfer: re-animates a reconstructed target
// photograph with the per-frame expression coefficients tracked from a
// source video. Identity and per-vertex texture stay fixed to the target;
// expression plus source-relative camera motion drive each rendered frame.

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/image_fitting.h"
#include "face_recon/mesh_export.h"
#include "face_recon/photometric_fitting.h"
#include "face_recon/rasterizer.h"

#include <CLI/CLI.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TrajectoryFrame {
  int index = 0;
  std::string status;
  face_recon::CameraParameters camera;
  bool has_pose = false;
  double observed_left_eye_opening = 0.0;
  double observed_right_eye_opening = 0.0;
  double fitted_left_eye_opening = 0.0;
  double fitted_right_eye_opening = 0.0;
  bool has_eye_opening = false;
  // Region-aware tracking reliabilities in [0,1]; 1.0 when the trajectory
  // predates the confidence columns.
  double left_eye_confidence = 1.0;
  double right_eye_confidence = 1.0;
  double mouth_confidence = 1.0;
  double pose_confidence = 1.0;
  bool has_confidence = false;
  Eigen::VectorXd expression;
};

int TriangleIndex(const Eigen::MatrixXi& triangles, int triangle, int corner) {
  return triangles.cols() == 3 ? triangles(triangle, corner)
                               : triangles(corner, triangle);
}

// Reads camera parameters and PCA coefficients back from a fitting report
// written by SaveFittingReport. Vertices are not populated; callers
// regenerate them with GenerateVertices.
face_recon::FittingResult LoadFittingReport(const std::string& filename) {
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("Could not open fitting report: " + filename);
  }
  face_recon::FittingResult result;
  std::string line;
  std::vector<double> shape;
  std::vector<double> expression;
  while (std::getline(input, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    std::istringstream value(line.substr(colon + 1));
    if (key == "solution_usable") {
      std::string flag;
      value >> flag;
      result.usable = flag == "yes";
    } else if (key == "camera_angle_axis") {
      value >> result.camera.angle_axis.x() >> result.camera.angle_axis.y() >>
          result.camera.angle_axis.z();
    } else if (key == "camera_translation") {
      value >> result.camera.translation.x() >>
          result.camera.translation.y() >> result.camera.translation.z();
    } else if (key == "camera_focal_length_normalized") {
      value >> result.camera.focal_length;
    } else if (key == "camera_aspect_ratio") {
      value >> result.camera.aspect_ratio;
    } else if (key.rfind("shape[", 0) == 0) {
      double number;
      value >> number;
      shape.push_back(number);
    } else if (key.rfind("expression[", 0) == 0) {
      double number;
      value >> number;
      expression.push_back(number);
    }
  }
  if (!result.usable || shape.empty() || expression.empty()) {
    throw std::runtime_error(
        "Fitting report has no usable coefficient state: " + filename);
  }
  result.shape_coefficients =
      Eigen::Map<Eigen::VectorXd>(shape.data(), shape.size());
  result.expression_coefficients =
      Eigen::Map<Eigen::VectorXd>(expression.data(), expression.size());
  return result;
}

// Reads normalized [0,1] RGB vertex colors from the ASCII PLY written by
// SaveMeshToPly.
Eigen::VectorXd LoadPlyVertexColors(const std::string& filename,
                                    int expected_vertex_count) {
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("Could not open PLY mesh: " + filename);
  }
  std::string line;
  int vertex_count = 0;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.rfind("element vertex ", 0) == 0) {
      vertex_count = std::stoi(line.substr(15));
    } else if (line.rfind("format ", 0) == 0 &&
               line.find("ascii") == std::string::npos) {
      throw std::runtime_error("Only ASCII PLY meshes are supported: " +
                               filename);
    } else if (line == "end_header") {
      break;
    }
  }
  if (vertex_count != expected_vertex_count) {
    throw std::runtime_error("PLY vertex count does not match the model: " +
                             filename);
  }
  Eigen::VectorXd colors(3 * vertex_count);
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    if (!std::getline(input, line)) {
      throw std::runtime_error("PLY mesh ended before all vertices: " +
                               filename);
    }
    std::istringstream row(line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    if (!(row >> x >> y >> z >> red >> green >> blue)) {
      throw std::runtime_error("PLY vertex row is missing colors: " +
                               filename);
    }
    colors[3 * vertex] = red / 255.0;
    colors[3 * vertex + 1] = green / 255.0;
    colors[3 * vertex + 2] = blue / 255.0;
  }
  return colors;
}

std::vector<TrajectoryFrame> LoadTrajectory(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open tracking trajectory: " + path);
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::runtime_error("Tracking trajectory is empty: " + path);
  }
  std::vector<std::string> header;
  std::istringstream header_stream(line);
  std::string field;
  while (std::getline(header_stream, field, ',')) {
    if (!field.empty() && field.back() == '\r') field.pop_back();
    header.push_back(field);
  }
  int first_expression_column = -1;
  const auto column_index = [&header](const std::string& name) {
    const auto it = std::find(header.begin(), header.end(), name);
    return it == header.end() ? -1 : static_cast<int>(it - header.begin());
  };
  for (std::size_t column = 0; column < header.size(); ++column) {
    if (header[column] == "expression_0") {
      first_expression_column = static_cast<int>(column);
      break;
    }
  }
  if (first_expression_column < 0) {
    throw std::runtime_error(
        "Tracking trajectory has no expression columns: " + path);
  }
  const int expression_count =
      static_cast<int>(header.size()) - first_expression_column;
  const std::array<int, 7> pose_columns = {
      column_index("angle_axis_x"), column_index("angle_axis_y"),
      column_index("angle_axis_z"), column_index("translation_x"),
      column_index("translation_y"), column_index("translation_z"),
      column_index("focal_length")};
  const bool trajectory_has_pose =
      std::all_of(pose_columns.begin(), pose_columns.end(),
                  [](int column) { return column >= 0; });
  const std::array<int, 4> eye_columns = {
      column_index("observed_left_eye_opening"),
      column_index("observed_right_eye_opening"),
      column_index("fitted_left_eye_opening"),
      column_index("fitted_right_eye_opening")};
  const bool trajectory_has_eye_opening =
      std::all_of(eye_columns.begin(), eye_columns.end(),
                  [](int column) { return column >= 0; });
  const std::array<int, 4> confidence_columns = {
      column_index("left_eye_confidence"),
      column_index("right_eye_confidence"),
      column_index("mouth_confidence"),
      column_index("pose_confidence")};
  const bool trajectory_has_confidence =
      std::all_of(confidence_columns.begin(), confidence_columns.end(),
                  [](int column) { return column >= 0; });

  std::vector<TrajectoryFrame> frames;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::vector<std::string> fields;
    std::istringstream row(line);
    while (std::getline(row, field, ',')) {
      if (!field.empty() && field.back() == '\r') field.pop_back();
      fields.push_back(field);
    }
    if (fields.size() != header.size()) {
      throw std::runtime_error("Tracking trajectory row is malformed: " + line);
    }
    TrajectoryFrame frame;
    frame.index = std::stoi(fields[0]);
    frame.status = fields[1];
    frame.has_pose = trajectory_has_pose;
    frame.has_eye_opening = trajectory_has_eye_opening;
    if (trajectory_has_pose) {
      frame.camera.angle_axis = Eigen::Vector3d(
          std::stod(fields[pose_columns[0]]),
          std::stod(fields[pose_columns[1]]),
          std::stod(fields[pose_columns[2]]));
      frame.camera.translation = Eigen::Vector3d(
          std::stod(fields[pose_columns[3]]),
          std::stod(fields[pose_columns[4]]),
          std::stod(fields[pose_columns[5]]));
      frame.camera.focal_length = std::stod(fields[pose_columns[6]]);
    }
    if (trajectory_has_eye_opening) {
      frame.observed_left_eye_opening = std::stod(fields[eye_columns[0]]);
      frame.observed_right_eye_opening = std::stod(fields[eye_columns[1]]);
      frame.fitted_left_eye_opening = std::stod(fields[eye_columns[2]]);
      frame.fitted_right_eye_opening = std::stod(fields[eye_columns[3]]);
    }
    frame.has_confidence = trajectory_has_confidence;
    if (trajectory_has_confidence) {
      frame.left_eye_confidence = std::stod(fields[confidence_columns[0]]);
      frame.right_eye_confidence = std::stod(fields[confidence_columns[1]]);
      frame.mouth_confidence = std::stod(fields[confidence_columns[2]]);
      frame.pose_confidence = std::stod(fields[confidence_columns[3]]);
    }
    frame.expression.resize(expression_count);
    for (int i = 0; i < expression_count; ++i) {
      frame.expression[i] = std::stod(fields[first_expression_column + i]);
    }
    frames.push_back(std::move(frame));
  }
  if (frames.empty()) {
    throw std::runtime_error("Tracking trajectory holds no frames: " + path);
  }
  return frames;
}

// Combines the target's fitted expression with the source frame's tracked
// expression. Absolute mode adopts the source coefficients directly (the
// Face2Face swap); relative mode keeps the target expression and adds the
// source's deviation from its own reference frame, which preserves the
// photograph's resting expression when it is not neutral.
Eigen::VectorXd ComposeExpression(const Eigen::VectorXd& target,
                                  const Eigen::VectorXd& source,
                                  const Eigen::VectorXd& source_reference,
                                  bool relative,
                                  double scale) {
  Eigen::VectorXd composed =
      relative ? target : Eigen::VectorXd::Zero(target.size());
  const int driven = std::min<int>(source.size(), target.size());
  if (relative) {
    composed.head(driven) +=
        scale * (source.head(driven) - source_reference.head(driven));
  } else {
    composed.head(driven) = scale * source.head(driven);
  }
  // The tracker bounds unit-variance coefficients to +/-3; transferred
  // expressions must not leave the space the model was fitted in.
  return composed.cwiseMax(-3.0).cwiseMin(3.0);
}

void ApplyOneEyeCorrective(Eigen::VectorXd* vertices, int outer, int inner,
                           int top, int bottom, double opening_residual) {
  const Eigen::Vector3d outer_point = vertices->segment<3>(3 * outer);
  const Eigen::Vector3d inner_point = vertices->segment<3>(3 * inner);
  const Eigen::Vector3d top_point = vertices->segment<3>(3 * top);
  const Eigen::Vector3d bottom_point = vertices->segment<3>(3 * bottom);
  const double eye_width = (outer_point - inner_point).norm();
  Eigen::Vector3d horizontal = outer_point - inner_point;
  Eigen::Vector3d vertical = top_point - bottom_point;
  if (eye_width < 1.0e-8 || vertical.norm() < 1.0e-8) return;
  horizontal.normalize();
  vertical.normalize();
  const Eigen::Vector3d center =
      0.25 * (outer_point + inner_point + top_point + bottom_point);
  const double residual = std::clamp(opening_residual, -0.25, 0.25);
  for (int vertex = 0; vertex < vertices->size() / 3; ++vertex) {
    const Eigen::Vector3d point = vertices->segment<3>(3 * vertex);
    const Eigen::Vector3d relative = point - center;
    const double x = relative.dot(horizontal) / eye_width;
    const double y = relative.dot(vertical) / eye_width;
    if (std::abs(x) > 0.75 || std::abs(y) > 0.48) continue;
    const double radial = std::exp(
        -2.5 * (x * x / (0.62 * 0.62) + y * y / (0.32 * 0.32)));
    // Move upper and lower eyelid neighborhoods in opposite directions. The
    // total landmark-gap change is approximately residual * eye_width.
    const double side = y >= 0.0 ? 1.0 : -1.0;
    vertices->segment<3>(3 * vertex) +=
        0.5 * residual * eye_width * radial * side * vertical;
  }
}

void ApplyBlinkCorrectives(Eigen::VectorXd* vertices, double left_residual,
                           double right_residual) {
  ApplyOneEyeCorrective(vertices, 15743, 15733, 15736, 14042,
                        left_residual);
  ApplyOneEyeCorrective(vertices, 2039, 2029, 2032, 338, right_residual);
}

// Confidence-gated first-order temporal filter for the per-frame eyelid
// signals. A low tracking confidence (far-side eye at large yaw, occluded or
// ambiguous lids) makes the filter hold its previous state, so the temporal
// prior wins over an unreliable measurement; a confident measurement tracks
// the observation, and a confident large innovation — a blink — is passed
// through unsmoothed so blink timing and depth are preserved.
class EyeSignalFilter {
 public:
  EyeSignalFilter(double minimum_alpha, double blink_confidence,
                  double blink_innovation)
      : minimum_alpha_(minimum_alpha),
        blink_confidence_(blink_confidence),
        blink_innovation_(blink_innovation) {}

  double Filter(double observation, double confidence) {
    if (!std::isfinite(observation)) return state_;
    if (!has_state_) {
      state_ = observation;
      has_state_ = true;
      return state_;
    }
    double alpha = std::clamp(confidence, minimum_alpha_, 1.0);
    if (confidence >= blink_confidence_ &&
        std::abs(observation - state_) >= blink_innovation_) {
      alpha = 1.0;
    }
    state_ += alpha * (observation - state_);
    return state_;
  }

 private:
  double minimum_alpha_;
  double blink_confidence_;
  double blink_innovation_;
  double state_ = 0.0;
  bool has_state_ = false;
};

Eigen::Vector3d PixelColor(const Eigen::VectorXd& colors,
                           const Eigen::MatrixXi& triangles,
                           const face_recon::RasterPixel& pixel) {
  const bool row_major = triangles.cols() == 3;
  Eigen::Vector3d color = Eigen::Vector3d::Zero();
  for (int corner = 0; corner < 3; ++corner) {
    const int vertex = row_major ? triangles(pixel.triangle_id, corner)
                                 : triangles(corner, pixel.triangle_id);
    color += static_cast<double>(pixel.barycentric[corner]) *
             colors.segment<3>(3 * vertex);
  }
  return color;
}

cv::Vec3b ToBgr8(const Eigen::Vector3d& rgb) {
  const Eigen::Vector3d value = rgb.cwiseMax(0.0).cwiseMin(1.0) * 255.0;
  return cv::Vec3b(static_cast<unsigned char>(std::lround(value[2])),
                   static_cast<unsigned char>(std::lround(value[1])),
                   static_cast<unsigned char>(std::lround(value[0])));
}

Eigen::VectorXd BuildProjectiveTextureCoordinates(
    const Eigen::VectorXd& reference_vertices,
    const face_recon::CameraParameters& reference_camera) {
  Eigen::VectorXd uv(2 * reference_vertices.size() / 3);
  for (int vertex = 0; vertex < reference_vertices.size() / 3; ++vertex) {
    const Eigen::Vector2d projected = face_recon::ProjectVertex(
        reference_vertices.segment<3>(3 * vertex), reference_camera);
    uv.segment<2>(2 * vertex) = projected;
  }
  return uv;
}

cv::Vec3b SampleBilinear(const cv::Mat& image, double x, double y) {
  x = std::clamp(x, 0.0, static_cast<double>(image.cols - 1));
  y = std::clamp(y, 0.0, static_cast<double>(image.rows - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const double wx = x - x0;
  const double wy = y - y0;
  cv::Vec3b result;
  for (int channel = 0; channel < 3; ++channel) {
    const double top = (1.0 - wx) * image.at<cv::Vec3b>(y0, x0)[channel] +
                       wx * image.at<cv::Vec3b>(y0, x1)[channel];
    const double bottom = (1.0 - wx) * image.at<cv::Vec3b>(y1, x0)[channel] +
                          wx * image.at<cv::Vec3b>(y1, x1)[channel];
    result[channel] = static_cast<unsigned char>(
        std::lround((1.0 - wy) * top + wy * bottom));
  }
  return result;
}

cv::Vec3b ProjectivePixelColor(
    const cv::Mat& texture, const Eigen::VectorXd& texture_uv,
    const Eigen::MatrixXi& triangles,
    const face_recon::RasterPixel& pixel) {
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  for (int corner = 0; corner < 3; ++corner) {
    const int vertex = TriangleIndex(triangles, pixel.triangle_id, corner);
    uv += static_cast<double>(pixel.barycentric[corner]) *
          texture_uv.segment<2>(2 * vertex);
  }
  return SampleBilinear(texture, uv.x() * (texture.cols - 1),
                        uv.y() * (texture.rows - 1));
}

cv::Mat BuildCleanBackground(
    const cv::Mat& photograph,
    const Eigen::VectorXd& target_vertices,
    const face_recon::CameraParameters& target_camera) {
  cv::Mat mask(photograph.rows, photograph.cols, CV_8UC1, cv::Scalar(0));
  const auto pixel = [&](int vertex) {
    const Eigen::Vector2d normalized = face_recon::ProjectVertex(
        target_vertices.segment<3>(3 * vertex), target_camera);
    return cv::Point(
        static_cast<int>(std::lround(normalized.x() * photograph.cols)),
        static_cast<int>(std::lround(normalized.y() * photograph.rows)));
  };
  const auto feature_ellipse = [&](int first, int second,
                                   double width_scale,
                                   double height_scale) {
    const cv::Point a = pixel(first);
    const cv::Point b = pixel(second);
    const cv::Point center((a.x + b.x) / 2, (a.y + b.y) / 2);
    const double distance = cv::norm(a - b);
    cv::ellipse(mask, center,
                cv::Size(std::max(2, static_cast<int>(
                                        std::lround(width_scale * distance))),
                         std::max(2, static_cast<int>(
                                        std::lround(height_scale * distance)))),
                0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED,
                cv::LINE_AA);
  };
  // Remove only high-information facial features. Inpainting the entire
  // frontal face creates a large flat patch; leaving the original skin while
  // erasing eyes/nose/mouth gives the rotated mesh plausible pixels behind
  // its changing silhouette without exposing a recognizable second face.
  feature_ellipse(15743, 15733, 0.65, 0.32);  // left eye + brow
  feature_ellipse(2039, 2029, 0.65, 0.32);    // right eye + brow
  feature_ellipse(15670, 1966, 0.62, 0.85);   // nose
  feature_ellipse(15427, 1723, 0.62, 0.32);   // mouth
  const int radius = std::max(
      3, static_cast<int>(std::lround(
             0.004 * std::max(photograph.cols, photograph.rows))));
  cv::dilate(mask, mask,
             cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                       cv::Size(2 * radius + 1,
                                                2 * radius + 1)));
  cv::Mat background;
  cv::inpaint(photograph, mask, background,
              std::max(3.0, 0.75 * radius), cv::INPAINT_NS);
  return background;
}

// Renders the re-expressed mesh over the photograph. The synthetic face is
// blended with a feathered alpha mask so the transition into the original
// image has no hard silhouette seam.
cv::Mat CompositeFrame(const cv::Mat& background,
                       const cv::Mat& target_texture,
                       const Eigen::VectorXd& texture_uv,
                       const Eigen::MatrixXi& triangles,
                       const face_recon::RasterizationResult& rasterization,
                       const Eigen::VectorXd& vertex_colors,
                       bool use_projective_texture,
                       int feather) {
  cv::Mat rendered = background.clone();
  cv::Mat mask(background.rows, background.cols, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const face_recon::RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) continue;
      rendered.at<cv::Vec3b>(y, x) = use_projective_texture
          ? ProjectivePixelColor(target_texture, texture_uv, triangles, pixel)
          : ToBgr8(PixelColor(vertex_colors, triangles, pixel));
      mask.at<unsigned char>(y, x) = 255;
    }
  }
  if (feather <= 0) {
    return rendered;
  }
  // Eroding before blurring keeps the feathered alpha inside the rendered
  // region, so the blend never samples undefined mesh colors.
  cv::Mat alpha;
  const int size = 2 * feather + 1;
  cv::erode(mask, alpha,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size)));
  cv::GaussianBlur(alpha, alpha, cv::Size(size, size), 0.0);
  cv::Mat composite = background.clone();
  for (int y = 0; y < background.rows; ++y) {
    for (int x = 0; x < background.cols; ++x) {
      const double weight = alpha.at<unsigned char>(y, x) / 255.0;
      if (weight <= 0.0) continue;
      const cv::Vec3b foreground = rendered.at<cv::Vec3b>(y, x);
      const cv::Vec3b background_pixel = background.at<cv::Vec3b>(y, x);
      cv::Vec3b& out = composite.at<cv::Vec3b>(y, x);
      for (int channel = 0; channel < 3; ++channel) {
        out[channel] = static_cast<unsigned char>(std::lround(
            weight * foreground[channel] +
            (1.0 - weight) * background_pixel[channel]));
      }
    }
  }
  return composite;
}

void WarpOneEyeAperture(cv::Mat* image, const Eigen::VectorXd& vertices,
                        const face_recon::CameraParameters& camera,
                        int outer_vertex, int inner_vertex, double scale) {
  scale = std::clamp(scale, 0.35, 2.6);
  if (std::abs(scale - 1.0) < 0.01) return;
  const Eigen::Vector2d outer = face_recon::ProjectVertex(
      vertices.segment<3>(3 * outer_vertex), camera);
  const Eigen::Vector2d inner = face_recon::ProjectVertex(
      vertices.segment<3>(3 * inner_vertex), camera);
  const cv::Point2d a(outer.x() * image->cols, outer.y() * image->rows);
  const cv::Point2d b(inner.x() * image->cols, inner.y() * image->rows);
  const cv::Point2d center = 0.5 * (a + b);
  const double eye_width = cv::norm(a - b);
  if (eye_width < 4.0) return;
  const int half_width = static_cast<int>(std::lround(0.67 * eye_width));
  const int half_height = static_cast<int>(std::lround(0.34 * eye_width));
  const cv::Rect bounds(
      std::max(0, static_cast<int>(std::floor(center.x)) - half_width),
      std::max(0, static_cast<int>(std::floor(center.y)) - half_height),
      std::min(image->cols - std::max(0, static_cast<int>(std::floor(center.x)) - half_width),
               2 * half_width + 1),
      std::min(image->rows - std::max(0, static_cast<int>(std::floor(center.y)) - half_height),
               2 * half_height + 1));
  if (bounds.width <= 2 || bounds.height <= 2) return;
  const cv::Mat original = image->clone();
  for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
    for (int x = bounds.x; x < bounds.x + bounds.width; ++x) {
      const double nx = (x - center.x) / std::max(0.67 * eye_width, 1.0);
      const double ny = (y - center.y) / std::max(0.34 * eye_width, 1.0);
      const double radius2 = nx * nx + ny * ny;
      if (radius2 >= 1.0) continue;
      const double source_y = center.y + (y - center.y) / scale;
      const cv::Vec3b warped = SampleBilinear(original, x, source_y);
      // Smoothly fade before the eyebrow and cheek so only the aperture is
      // visibly rescaled and no rectangular patch boundary is introduced.
      // Keep most of the eye aperture at full strength and feather only the
      // outer rim. A fully radial fade made even large opening ratios almost
      // invisible after compositing.
      const double alpha = std::clamp((1.0 - radius2) / 0.35, 0.0, 1.0);
      cv::Vec3b& output = image->at<cv::Vec3b>(y, x);
      for (int channel = 0; channel < 3; ++channel) {
        output[channel] = static_cast<unsigned char>(std::lround(
            alpha * warped[channel] + (1.0 - alpha) * output[channel]));
      }
    }
  }
}

void WarpEyeApertures(cv::Mat* image, const Eigen::VectorXd& vertices,
                      const face_recon::CameraParameters& camera,
                      double left_scale, double right_scale) {
  WarpOneEyeAperture(image, vertices, camera, 15743, 15733, left_scale);
  WarpOneEyeAperture(image, vertices, camera, 2039, 2029, right_scale);
}

// Rigid head proxy for the background warp. The BFM face12 mesh covers only
// the frontal face mask, so hair, scalp, ears, and the head silhouette get
// their motion from a coarse ellipsoid placed behind the face surface, plus
// subsampled mesh vertices so the field agrees with the rendered face near
// the jawline. Only the camera-facing half of the ellipsoid is kept: back-of-
// head points would project onto the same pixels with opposite motion and
// corrupt the field.
std::vector<Eigen::Vector3d> BuildHeadProxyPoints(
    const Eigen::VectorXd& vertices) {
  std::vector<Eigen::Vector3d> points;
  const int vertex_count = static_cast<int>(vertices.size() / 3);
  Eigen::Vector3d minimum =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector3d maximum =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    const Eigen::Vector3d point = vertices.segment<3>(3 * vertex);
    minimum = minimum.cwiseMin(point);
    maximum = maximum.cwiseMax(point);
  }
  for (int vertex = 0; vertex < vertex_count; vertex += 40) {
    points.push_back(vertices.segment<3>(3 * vertex));
  }
  const Eigen::Vector3d half = 0.5 * (maximum - minimum);
  // Anthropometric proportions relative to the face-mask bounding box: hair
  // adds width and height, and the skull center sits roughly half a head
  // depth behind the face surface, slightly above the mask center.
  const Eigen::Vector3d center(0.5 * (minimum.x() + maximum.x()),
                               0.5 * (minimum.y() + maximum.y()) +
                                   0.15 * half.y(),
                               maximum.z() - 1.05 * half.x());
  const double radius_x = 1.25 * half.x();
  const double radius_y = 1.35 * half.y();
  const double radius_z = 1.25 * half.x();
  constexpr double kPi = 3.14159265358979323846;
  constexpr int kLatitudes = 24;
  constexpr int kLongitudes = 48;
  for (int i = 1; i < kLatitudes; ++i) {
    const double theta = kPi * i / kLatitudes;
    for (int j = 0; j < kLongitudes; ++j) {
      const double phi = 2.0 * kPi * j / kLongitudes;
      const Eigen::Vector3d point(
          center.x() + radius_x * std::sin(theta) * std::cos(phi),
          center.y() + radius_y * std::cos(theta),
          center.z() + radius_z * std::sin(theta) * std::sin(phi));
      if (point.z() - center.z() >= -0.15 * radius_z) {
        points.push_back(point);
      }
    }
  }
  return points;
}

// Warps the photograph so the head region outside the face mesh follows the
// transferred rigid motion. Proxy displacements (reference camera vs current
// camera) are splatted around their destination pixels into a coarse grid,
// blurred into a smooth field whose magnitude decays to zero away from any
// sample, and applied as an inverse warp. Revealed regions stretch instead
// of leaving holes; far from the head the image stays static.
cv::Mat WarpHeadBackground(
    const cv::Mat& background,
    const std::vector<Eigen::Vector3d>& proxy_points,
    const face_recon::CameraParameters& reference_camera,
    const face_recon::CameraParameters& current_camera) {
  constexpr int kCoarseScale = 8;
  const int coarse_width = (background.cols + kCoarseScale - 1) / kCoarseScale;
  const int coarse_height =
      (background.rows + kCoarseScale - 1) / kCoarseScale;
  cv::Mat flow(coarse_height, coarse_width, CV_32FC2, cv::Scalar(0, 0));
  cv::Mat weight(coarse_height, coarse_width, CV_32FC1, cv::Scalar(0));
  for (const auto& point : proxy_points) {
    const Eigen::Vector2d source_norm =
        face_recon::ProjectVertex(point, reference_camera);
    const Eigen::Vector2d destination_norm =
        face_recon::ProjectVertex(point, current_camera);
    const cv::Point2f source(source_norm.x() * background.cols,
                             source_norm.y() * background.rows);
    const cv::Point2f destination(destination_norm.x() * background.cols,
                                  destination_norm.y() * background.rows);
    const int cell_x = static_cast<int>(destination.x) / kCoarseScale;
    const int cell_y = static_cast<int>(destination.y) / kCoarseScale;
    if (cell_x < 0 || cell_y < 0 || cell_x >= coarse_width ||
        cell_y >= coarse_height) {
      continue;
    }
    flow.at<cv::Vec2f>(cell_y, cell_x) +=
        cv::Vec2f(source.x - destination.x, source.y - destination.y);
    weight.at<float>(cell_y, cell_x) += 1.0f;
  }
  const int kernel =
      2 * std::max(2, std::min(coarse_width, coarse_height) / 12) + 1;
  cv::GaussianBlur(flow, flow, cv::Size(kernel, kernel), 0.0);
  cv::GaussianBlur(weight, weight, cv::Size(kernel, kernel), 0.0);
  cv::Mat flow_full;
  cv::Mat weight_full;
  cv::resize(flow, flow_full, background.size(), 0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(weight, weight_full, background.size(), 0.0, 0.0,
             cv::INTER_LINEAR);
  // offset = flow / (weight + knee): the knee both averages the splats and
  // fades the field smoothly to zero where the sample density vanishes.
  constexpr float kWeightKnee = 0.1f;
  cv::Mat map_x(background.rows, background.cols, CV_32FC1);
  cv::Mat map_y(background.rows, background.cols, CV_32FC1);
  for (int y = 0; y < background.rows; ++y) {
    for (int x = 0; x < background.cols; ++x) {
      const cv::Vec2f accumulated = flow_full.at<cv::Vec2f>(y, x);
      const float denominator = weight_full.at<float>(y, x) + kWeightKnee;
      map_x.at<float>(y, x) = x + accumulated[0] / denominator;
      map_y.at<float>(y, x) = y + accumulated[1] / denominator;
    }
  }
  cv::Mat warped;
  cv::remap(background, warped, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_REPLICATE);
  return warped;
}

bool SaveGeometryConditions(
    const std::filesystem::path& root, const std::string& filename,
    const Eigen::MatrixXi& triangles,
    const face_recon::RasterizationResult& rasterization,
    const Eigen::VectorXd& vertices,
    const face_recon::CameraParameters& camera,
    const Eigen::VectorXd& vertex_colors, const cv::Mat& target_texture,
    const Eigen::VectorXd& texture_uv, bool use_projective_texture,
    const std::array<double, 4>& region_confidences) {
  const std::array<std::string, 5> directories = {
      "coarse_rgb", "depth", "normals", "visibility", "reliability"};
  for (const auto& directory : directories)
    std::filesystem::create_directories(root / directory);
  cv::Mat coarse(rasterization.height, rasterization.width, CV_8UC3,
                 cv::Scalar(0, 0, 0));
  cv::Mat normals_image(rasterization.height, rasterization.width, CV_8UC3,
                        cv::Scalar(0, 0, 0));
  cv::Mat visibility(rasterization.height, rasterization.width, CV_8UC1,
                     cv::Scalar(0));
  const Eigen::VectorXd camera_normals = face_recon::ApplyCameraRotation(
      face_recon::ComputeVertexNormals(vertices, triangles), camera);
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const auto& pixel = rasterization.at(x, y);
      if (!pixel.visible()) continue;
      coarse.at<cv::Vec3b>(y, x) = use_projective_texture
          ? ProjectivePixelColor(target_texture, texture_uv, triangles, pixel)
          : ToBgr8(PixelColor(vertex_colors, triangles, pixel));
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      for (int corner = 0; corner < 3; ++corner) {
        const int vertex =
            TriangleIndex(triangles, pixel.triangle_id, corner);
        normal += static_cast<double>(pixel.barycentric[corner]) *
                  camera_normals.segment<3>(3 * vertex);
      }
      if (normal.squaredNorm() > 1.0e-12) normal.normalize();
      normals_image.at<cv::Vec3b>(y, x) = ToBgr8(0.5 * (normal.array() + 1.0).matrix());
      visibility.at<unsigned char>(y, x) = 255;
    }
  }

  // Per-pixel reliability of the driving signal for the completion backend:
  // the rendered face carries the source pose confidence, with the eye and
  // mouth neighborhoods overwritten by their own region confidences.
  cv::Mat reliability(rasterization.height, rasterization.width, CV_8UC1,
                      cv::Scalar(0));
  const auto to_gray = [](double confidence) {
    return static_cast<unsigned char>(std::lround(
        255.0 * std::clamp(confidence, 0.0, 1.0)));
  };
  reliability.setTo(to_gray(region_confidences[3]), visibility);
  const auto region_ellipse = [&](int first, int second, double width_scale,
                                  double height_scale, double confidence) {
    const auto project = [&](int vertex) {
      const Eigen::Vector2d normalized = face_recon::ProjectVertex(
          vertices.segment<3>(3 * vertex), camera);
      return cv::Point(
          static_cast<int>(std::lround(normalized.x() * rasterization.width)),
          static_cast<int>(std::lround(normalized.y() * rasterization.height)));
    };
    const cv::Point a = project(first);
    const cv::Point b = project(second);
    const double distance = cv::norm(a - b);
    cv::Mat region(reliability.size(), CV_8UC1, cv::Scalar(0));
    cv::ellipse(region, cv::Point((a.x + b.x) / 2, (a.y + b.y) / 2),
                cv::Size(std::max(2, static_cast<int>(
                                        std::lround(width_scale * distance))),
                         std::max(2, static_cast<int>(std::lround(
                                        height_scale * distance)))),
                0.0, 0.0, 360.0, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    cv::bitwise_and(region, visibility, region);
    reliability.setTo(to_gray(confidence), region);
  };
  region_ellipse(15743, 15733, 0.65, 0.32, region_confidences[0]);  // left eye
  region_ellipse(2039, 2029, 0.65, 0.32, region_confidences[1]);    // right eye
  region_ellipse(15427, 1723, 0.62, 0.32, region_confidences[2]);   // mouth

  return cv::imwrite((root / "coarse_rgb" / filename).string(), coarse) &&
         face_recon::SaveRasterDepthImage(
             rasterization, (root / "depth" / filename).string()) &&
         cv::imwrite((root / "normals" / filename).string(), normals_image) &&
         cv::imwrite((root / "visibility" / filename).string(), visibility) &&
         cv::imwrite((root / "reliability" / filename).string(), reliability);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Transfer tracked video expressions onto a reconstructed photo"};
  std::string model_path = "data/model2019_face12.h5";
  std::string target_fitting_path;
  std::string target_mesh_path;
  std::string target_image_path;
  std::string trajectory_path;
  std::string output_dir = "transfer";
  bool relative = false;
  bool disable_pose = false;
  bool disable_background_inpaint = false;
  bool disable_blink_correction = false;
  bool save_conditions = false;
  bool pose_only = false;
  double expression_scale = 1.0;
  double pose_scale = 1.0;
  double blink_scale = 1.0;
  double eye_warp_gain = 2.5;
  std::string texture_mode = "projective";
  int feather = -1;
  int max_frames = 0;
  app.add_option("--model", model_path)->check(CLI::ExistingFile);
  app.add_option("--target-fitting", target_fitting_path,
                 "fitting.txt of the reconstructed target photograph")
      ->required()->check(CLI::ExistingFile);
  app.add_option("--target-mesh", target_mesh_path,
                 "face.ply of the target; provides the fitted vertex colors")
      ->required()->check(CLI::ExistingFile);
  app.add_option("--target-image", target_image_path,
                 "Original target photograph used as the compositing background")
      ->required()->check(CLI::ExistingFile);
  app.add_option("--trajectory", trajectory_path,
                 "tracking.csv with per-frame source expressions")
      ->required()->check(CLI::ExistingFile);
  app.add_option("--output", output_dir);
  app.add_flag("--relative", relative,
               "Add source expression deltas instead of replacing coefficients");
  app.add_option("--expression-scale", expression_scale,
                 "Gain applied to the transferred expression");
  app.add_flag("--no-pose", disable_pose,
               "Transfer expression only; keep the target camera fixed");
  app.add_flag("--pose-only", pose_only,
               "Transfer pose while retaining the target expression");
  app.add_flag("--no-background-inpaint", disable_background_inpaint,
               "Keep the original target face behind pose-transferred frames");
  bool head_warp = false;
  app.add_flag("--head-warp", head_warp,
               "Warp hair/scalp/head silhouette along the transferred rigid "
               "motion using a scalp-ellipsoid proxy");
  app.add_option("--pose-scale", pose_scale,
                 "Gain applied to source-relative head rotation/translation")
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--no-blink-correction", disable_blink_correction,
               "Disable explicit eyelid corrective deformation");
  bool disable_uncertainty_control = false;
  app.add_flag("--no-uncertainty-control", disable_uncertainty_control,
               "Ignore tracker region confidences: use raw eyelid signals "
               "without confidence-gated temporal filtering");
  app.add_option("--blink-scale", blink_scale,
                 "Gain applied to observed-minus-fitted eyelid opening")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--eye-warp-gain", eye_warp_gain,
                 "Gain for visible image-space eye opening relative to frame 0")
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--save-conditions", save_conditions,
               "Save coarse RGB/depth/normal/visibility neural-renderer inputs");
  app.add_option("--feather", feather,
                 "Alpha feather radius in pixels; default scales with image size");
  app.add_option("--texture-mode", texture_mode,
                 "projective samples the target photo; vertex uses fitted PLY RGB")
      ->check(CLI::IsMember({"projective", "vertex"}));
  app.add_option("--max-frames", max_frames, "Optional development/debug limit");
  CLI11_PARSE(app, argc, argv);

  try {
    face_recon::BfmModel model;
    if (!model.LoadFromH5(model_path)) return 1;
    const face_recon::FittingResult target =
        LoadFittingReport(target_fitting_path);
    const int vertex_count = static_cast<int>(model.shape().mean.size() / 3);
    const Eigen::VectorXd vertex_colors =
        LoadPlyVertexColors(target_mesh_path, vertex_count);
    const cv::Mat photograph = cv::imread(target_image_path, cv::IMREAD_COLOR);
    if (photograph.empty()) {
      throw std::runtime_error("Could not read target photograph: " +
                               target_image_path);
    }
    const Eigen::VectorXd target_reference_vertices =
        face_recon::GenerateVertices(model, target.shape_coefficients,
                                     target.expression_coefficients);
    const Eigen::VectorXd texture_uv = BuildProjectiveTextureCoordinates(
        target_reference_vertices, target.camera);
    const bool use_projective_texture = texture_mode == "projective";
    std::vector<TrajectoryFrame> frames = LoadTrajectory(trajectory_path);
    if (max_frames > 0 && static_cast<int>(frames.size()) > max_frames) {
      frames.resize(max_frames);
    }
    if (feather < 0) {
      feather = std::max(
          2, static_cast<int>(std::lround(
                 0.005 * std::max(photograph.cols, photograph.rows))));
    }

    const std::filesystem::path frames_dir =
        std::filesystem::path(output_dir) / "frames";
    std::filesystem::create_directories(frames_dir);
    std::ofstream report(std::filesystem::path(output_dir) / "transfer.csv");
    report << "frame,status,angle_axis_x,angle_axis_y,angle_axis_z,"
              "translation_x,translation_y,translation_z,"
              "left_eye_correction,right_eye_correction,"
              "left_eye_confidence,right_eye_confidence,"
              "mouth_confidence,pose_confidence"
           << std::setprecision(10);
    const int expression_count =
        static_cast<int>(frames.front().expression.size());
    for (int i = 0; i < expression_count; ++i) report << ",expression_" << i;
    report << "\n";

    const Eigen::VectorXd& reference = frames.front().expression;
    const bool transfer_pose = !disable_pose && frames.front().has_pose;
    if (!disable_pose && !frames.front().has_pose) {
      std::cerr << "[WARNING] Trajectory has no camera columns; keeping the "
                   "target pose fixed\n";
    }
    // Confidence-gated temporal filters for the eyelid driving signals. The
    // alpha floor keeps a long-occluded eye from freezing forever; it slowly
    // relaxes toward whatever the tracker still reports.
    const bool uncertainty_control = !disable_uncertainty_control;
    constexpr double kMinimumAlpha = 0.15;
    constexpr double kBlinkConfidence = 0.6;
    constexpr double kBlinkInnovation = 0.05;
    EyeSignalFilter observed_left_filter(kMinimumAlpha, kBlinkConfidence,
                                         kBlinkInnovation);
    EyeSignalFilter observed_right_filter(kMinimumAlpha, kBlinkConfidence,
                                          kBlinkInnovation);
    EyeSignalFilter fitted_left_filter(kMinimumAlpha, kBlinkConfidence,
                                       kBlinkInnovation);
    EyeSignalFilter fitted_right_filter(kMinimumAlpha, kBlinkConfidence,
                                        kBlinkInnovation);
    cv::Mat background = photograph;
    if (transfer_pose && !disable_background_inpaint) {
      background =
          BuildCleanBackground(photograph, target_reference_vertices,
                               target.camera);
      cv::imwrite((std::filesystem::path(output_dir) /
                   "clean_background.png").string(),
                  background);
    }
    std::vector<Eigen::Vector3d> head_proxy;
    if (head_warp && transfer_pose) {
      head_proxy = BuildHeadProxyPoints(target_reference_vertices);
    } else if (head_warp) {
      std::cerr << "[WARNING] --head-warp does nothing without pose "
                   "transfer\n";
    }
    for (std::size_t position = 0; position < frames.size(); ++position) {
      const TrajectoryFrame& frame = frames[position];
      const Eigen::VectorXd expression = ComposeExpression(
          target.expression_coefficients, frame.expression, reference,
          relative, expression_scale);
      const Eigen::VectorXd driven_expression =
          pose_only ? target.expression_coefficients : expression;
      Eigen::VectorXd vertices = face_recon::GenerateVertices(
          model, target.shape_coefficients, driven_expression);
      double observed_left = frame.observed_left_eye_opening;
      double observed_right = frame.observed_right_eye_opening;
      double fitted_left = frame.fitted_left_eye_opening;
      double fitted_right = frame.fitted_right_eye_opening;
      if (uncertainty_control && frame.has_eye_opening) {
        observed_left =
            observed_left_filter.Filter(observed_left,
                                        frame.left_eye_confidence);
        observed_right =
            observed_right_filter.Filter(observed_right,
                                         frame.right_eye_confidence);
        fitted_left =
            fitted_left_filter.Filter(fitted_left, frame.left_eye_confidence);
        fitted_right = fitted_right_filter.Filter(fitted_right,
                                                  frame.right_eye_confidence);
      }
      double left_eye_correction = 0.0;
      double right_eye_correction = 0.0;
      if (!disable_blink_correction && frame.has_eye_opening) {
        left_eye_correction = blink_scale * (observed_left - fitted_left);
        right_eye_correction = blink_scale * (observed_right - fitted_right);
        ApplyBlinkCorrectives(&vertices, left_eye_correction,
                              right_eye_correction);
      }
      const face_recon::CameraParameters frame_camera =
          transfer_pose
              ? face_recon::TransferRelativeCameraPose(
                    target.camera, frames.front().camera, frame.camera,
                    pose_scale)
              : target.camera;
      const face_recon::RasterizationResult rasterization =
          face_recon::RasterizeMesh(vertices, model.triangles(), frame_camera,
                                    photograph.cols, photograph.rows);
      cv::Mat frame_background = background;
      if (!head_proxy.empty()) {
        frame_background = WarpHeadBackground(background, head_proxy,
                                              target.camera, frame_camera);
      }
      cv::Mat composite = CompositeFrame(
          frame_background, photograph, texture_uv, model.triangles(),
          rasterization, vertex_colors, use_projective_texture, feather);
      if (!disable_blink_correction && frame.has_eye_opening &&
          frames.front().has_eye_opening) {
        const double left_reference =
            std::max(frames.front().observed_left_eye_opening, 1.0e-6);
        const double right_reference =
            std::max(frames.front().observed_right_eye_opening, 1.0e-6);
        const double left_scale = 1.0 + eye_warp_gain *
            (observed_left / left_reference - 1.0);
        const double right_scale = 1.0 + eye_warp_gain *
            (observed_right / right_reference - 1.0);
        WarpEyeApertures(&composite, vertices, frame_camera, left_scale,
                         right_scale);
      }
      std::ostringstream name;
      name << "frame_" << std::setw(6) << std::setfill('0') << frame.index
           << ".png";
      const auto frame_path = frames_dir / name.str();
      if (!cv::imwrite(frame_path.string(), composite)) {
        throw std::runtime_error("Could not write transfer frame: " +
                                 frame_path.string());
      }
      if (save_conditions &&
          !SaveGeometryConditions(
              std::filesystem::path(output_dir) / "conditions", name.str(),
              model.triangles(), rasterization, vertices, frame_camera,
              vertex_colors, photograph, texture_uv, use_projective_texture,
              {frame.left_eye_confidence, frame.right_eye_confidence,
               frame.mouth_confidence, frame.pose_confidence})) {
        throw std::runtime_error("Could not write geometry conditions for " +
                                 name.str());
      }
      report << frame.index << ',' << frame.status << ','
             << frame_camera.angle_axis.x() << ','
             << frame_camera.angle_axis.y() << ','
             << frame_camera.angle_axis.z() << ','
             << frame_camera.translation.x() << ','
             << frame_camera.translation.y() << ','
             << frame_camera.translation.z() << ',' << left_eye_correction
             << ',' << right_eye_correction << ','
             << frame.left_eye_confidence << ','
             << frame.right_eye_confidence << ','
             << frame.mouth_confidence << ',' << frame.pose_confidence;
      for (int i = 0; i < expression_count; ++i) {
        report << ','
               << (i < driven_expression.size() ? driven_expression[i] : 0.0);
      }
      report << '\n';
      std::cout << "[TRANSFER] " << position + 1 << "/" << frames.size()
                << " (source frame " << frame.index << ", " << frame.status
                << ")\n";
    }
    std::cout << "[SUCCESS] Expression transfer frames written to "
              << frames_dir << "\n";
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << "\n";
    return 1;
  }
  return 0;
}
