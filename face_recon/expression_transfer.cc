// Face2Face-style expression transfer: re-animates a reconstructed target
// photograph with the per-frame expression coefficients tracked from a
// source video. Identity, camera pose, and per-vertex texture stay fixed to
// the target; only the expression coefficients are exchanged before the mesh
// is re-generated, rasterized, and composited back onto the photograph.

#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/image_fitting.h"
#include "face_recon/mesh_export.h"
#include "face_recon/rasterizer.h"

#include <CLI/CLI.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <algorithm>
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
  Eigen::VectorXd expression;
};

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

// Renders the re-expressed mesh over the photograph. The synthetic face is
// blended with a feathered alpha mask so the transition into the original
// image has no hard silhouette seam.
cv::Mat CompositeFrame(const cv::Mat& photograph,
                       const Eigen::MatrixXi& triangles,
                       const face_recon::RasterizationResult& rasterization,
                       const Eigen::VectorXd& vertex_colors,
                       int feather) {
  cv::Mat rendered = photograph.clone();
  cv::Mat mask(photograph.rows, photograph.cols, CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < rasterization.height; ++y) {
    for (int x = 0; x < rasterization.width; ++x) {
      const face_recon::RasterPixel& pixel = rasterization.at(x, y);
      if (!pixel.visible()) continue;
      rendered.at<cv::Vec3b>(y, x) =
          ToBgr8(PixelColor(vertex_colors, triangles, pixel));
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
  cv::Mat composite = photograph.clone();
  for (int y = 0; y < photograph.rows; ++y) {
    for (int x = 0; x < photograph.cols; ++x) {
      const double weight = alpha.at<unsigned char>(y, x) / 255.0;
      if (weight <= 0.0) continue;
      const cv::Vec3b foreground = rendered.at<cv::Vec3b>(y, x);
      const cv::Vec3b background = photograph.at<cv::Vec3b>(y, x);
      cv::Vec3b& out = composite.at<cv::Vec3b>(y, x);
      for (int channel = 0; channel < 3; ++channel) {
        out[channel] = static_cast<unsigned char>(std::lround(
            weight * foreground[channel] + (1.0 - weight) * background[channel]));
      }
    }
  }
  return composite;
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
  double expression_scale = 1.0;
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
  app.add_option("--feather", feather,
                 "Alpha feather radius in pixels; default scales with image size");
  app.add_option("--max-frames", max_frames, "Optional development/debug limit");
  CLI11_PARSE(app, argc, argv);

  try {
    face_recon::BfmModel model;
    if (!model.LoadFromH5(model_path)) return 1;
    const face_recon::FittingResult target =
        face_recon::LoadFittingReport(target_fitting_path);
    const int vertex_count = static_cast<int>(model.shape().mean.size() / 3);
    const Eigen::VectorXd vertex_colors =
        face_recon::LoadPlyVertexColors(target_mesh_path, vertex_count);
    const cv::Mat photograph = cv::imread(target_image_path, cv::IMREAD_COLOR);
    if (photograph.empty()) {
      throw std::runtime_error("Could not read target photograph: " +
                               target_image_path);
    }
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
    report << "frame,status" << std::setprecision(10);
    const int expression_count =
        static_cast<int>(frames.front().expression.size());
    for (int i = 0; i < expression_count; ++i) report << ",expression_" << i;
    report << "\n";

    const Eigen::VectorXd& reference = frames.front().expression;
    for (std::size_t position = 0; position < frames.size(); ++position) {
      const TrajectoryFrame& frame = frames[position];
      const Eigen::VectorXd expression = ComposeExpression(
          target.expression_coefficients, frame.expression, reference,
          relative, expression_scale);
      const Eigen::VectorXd vertices = face_recon::GenerateVertices(
          model, target.shape_coefficients, expression);
      const face_recon::RasterizationResult rasterization =
          face_recon::RasterizeMesh(vertices, model.triangles(), target.camera,
                                    photograph.cols, photograph.rows);
      const cv::Mat composite = CompositeFrame(
          photograph, model.triangles(), rasterization, vertex_colors, feather);
      std::ostringstream name;
      name << "frame_" << std::setw(6) << std::setfill('0') << frame.index
           << ".png";
      const auto frame_path = frames_dir / name.str();
      if (!cv::imwrite(frame_path.string(), composite)) {
        throw std::runtime_error("Could not write transfer frame: " +
                                 frame_path.string());
      }
      report << frame.index << ',' << frame.status;
      for (int i = 0; i < expression_count; ++i) {
        report << ',' << (i < expression.size() ? expression[i] : 0.0);
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
