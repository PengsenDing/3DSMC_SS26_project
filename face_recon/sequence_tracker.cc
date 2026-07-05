#include "face_recon/bfm_model.h"
#include "face_recon/fitting.h"
#include "face_recon/image_fitting.h"
#include "face_recon/mesh_export.h"
#include "face_reconstruction/landmarks.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FrameInput {
  int index = 0;
  std::string image;
  std::string landmarks;
};

std::vector<std::string> Split(const std::string& line, char separator) {
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, separator)) {
    if (!field.empty() && field.back() == '\r') field.pop_back();
    fields.push_back(field);
  }
  return fields;
}

std::vector<FrameInput> LoadManifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Could not open sequence manifest: " + path);
  std::vector<FrameInput> frames;
  std::string line;
  bool first = true;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    if (first) { first = false; if (line.rfind("frame,", 0) == 0) continue; }
    const auto fields = Split(line, ',');
    if (fields.size() != 3) throw std::runtime_error("Manifest rows require frame,image,landmarks");
    frames.push_back({std::stoi(fields[0]), fields[1], fields[2]});
  }
  if (frames.empty()) throw std::runtime_error("Sequence manifest is empty");
  return frames;
}

face_recon::FittingResult LoadInitialFit(const std::string& path,
                                        const face_recon::BfmModel& model) {
  face_recon::FittingResult result = face_recon::LoadFittingReport(path);
  result.vertices = face_recon::GenerateVertices(
      model, result.shape_coefficients, result.expression_coefficients);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Track fixed BFM identity through an image sequence"};
  std::string model_path = "data/model2019_face12.h5";
  std::string correspondence_path = "data/bfm_mediapipe_correspondence.csv";
  std::string manifest_path;
  std::string initial_fit_path;
  std::string output_dir = "tracking";
  bool save_meshes = false;
  int shape_components = 50;
  int expression_components = 30;
  face_recon::TrackingOptions options;
  app.add_option("--model", model_path)->check(CLI::ExistingFile);
  app.add_option("--correspondences", correspondence_path)->check(CLI::ExistingFile);
  app.add_option("--manifest", manifest_path)->required()->check(CLI::ExistingFile);
  app.add_option("--initial-fitting", initial_fit_path)->required()->check(CLI::ExistingFile);
  app.add_option("--output", output_dir);
  app.add_flag("--save-meshes", save_meshes);
  app.add_option("--shape-components", shape_components);
  app.add_option("--expression-components", expression_components);
  app.add_option("--iterations", options.iterations);
  app.add_option("--pose-temporal-weight", options.pose_temporal_weight);
  app.add_option("--expression-temporal-weight", options.expression_temporal_weight);
  app.add_option("--acceleration-weight", options.acceleration_weight);
  app.add_option("--failure-rmse", options.failure_rmse);
  app.add_flag("--verbose-optimization", options.verbose);
  CLI11_PARSE(app, argc, argv);

  try {
    face_recon::BfmModel model;
    if (!model.LoadFromH5(model_path)) return 1;
    const auto correspondences =
        face_reconstruction::load_bfm_mediapipe_correspondences(correspondence_path);
    const auto frames = LoadManifest(manifest_path);
    face_recon::FittingResult identity = LoadInitialFit(initial_fit_path, model);
    shape_components = std::clamp(
        shape_components, 1,
        static_cast<int>(identity.shape_coefficients.size()));
    expression_components = std::clamp(
        expression_components, 1,
        static_cast<int>(identity.expression_coefficients.size()));
    identity.shape_coefficients.conservativeResize(shape_components);
    identity.expression_coefficients.conservativeResize(expression_components);
    identity.vertices = face_recon::GenerateVertices(
        model, identity.shape_coefficients, identity.expression_coefficients);
    std::filesystem::create_directories(output_dir);
    std::filesystem::create_directories(std::filesystem::path(output_dir) / "overlays");
    if (save_meshes) std::filesystem::create_directories(std::filesystem::path(output_dir) / "meshes");

    std::ofstream trajectory(std::filesystem::path(output_dir) / "tracking.csv");
    trajectory << "frame,status,rmse,angle_axis_x,angle_axis_y,angle_axis_z,"
                  "translation_x,translation_y,translation_z,focal_length";
    for (int i = 0; i < identity.expression_coefficients.size(); ++i)
      trajectory << ",expression_" << i;
    trajectory << "\n" << std::setprecision(10);

    face_recon::FittingResult previous = identity;
    face_recon::FittingResult older = identity;
    for (std::size_t position = 0; position < frames.size(); ++position) {
      const FrameInput& frame = frames[position];
      face_recon::FittingResult current;
      std::string status = "tracked";
      if (position == 0) {
        current = identity;
        current.final_rmse = 0.0;
        status = "initialized";
      } else {
        const auto landmarks = face_reconstruction::load_landmarks_csv(frame.landmarks);
        current = face_recon::TrackBfmFrame(model, landmarks, correspondences,
                                             identity, previous, &older, options);
        if (!current.usable) {
          current = face_recon::TrackBfmFrame(model, landmarks, correspondences,
                                               identity, identity, nullptr, options);
          status = current.usable ? "reinitialized" : "failed_carried";
        }
        if (!current.usable) {
          current = previous;
          current.final_rmse = -1.0;
        }
      }
      const auto overlay_path = std::filesystem::path(output_dir) / "overlays" /
          ("frame_" + std::to_string(frame.index) + ".png");
      if (!current.reprojections.empty())
        face_recon::SaveReprojectionOverlay(frame.image, current, overlay_path.string());
      if (save_meshes) {
        const auto mesh_path = std::filesystem::path(output_dir) / "meshes" /
            ("frame_" + std::to_string(frame.index) + ".ply");
        face_recon::SaveMeshToPly(model, current.vertices, mesh_path.string());
      }
      trajectory << frame.index << ',' << status << ',' << current.final_rmse << ','
                 << current.camera.angle_axis.x() << ',' << current.camera.angle_axis.y() << ','
                 << current.camera.angle_axis.z() << ',' << current.camera.translation.x() << ','
                 << current.camera.translation.y() << ',' << current.camera.translation.z() << ','
                 << identity.camera.focal_length;
      for (int i = 0; i < current.expression_coefficients.size(); ++i)
        trajectory << ',' << current.expression_coefficients[i];
      trajectory << '\n';
      older = previous;
      previous = current;
      std::cout << "[TRACK] " << frame.index << " " << status
                << " rmse=" << current.final_rmse << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << "\n";
    return 1;
  }
  return 0;
}
