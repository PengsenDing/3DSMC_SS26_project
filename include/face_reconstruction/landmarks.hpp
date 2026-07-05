#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace face_reconstruction {

struct Landmark2D {
    int index = 0;
    std::string name;
    float u = 0.0f;
    float v = 0.0f;
    float z_norm = 0.0f;
    float x_norm = 0.0f;
    float y_norm = 0.0f;
};

struct BfmMediaPipeCorrespondence {
    std::string bfm_landmark_name;
    int bfm_vertex_id = 0;
    int mediapipe_index = 0;
};

std::vector<Landmark2D> load_landmarks_csv(const std::filesystem::path& path);
std::string landmark_summary(const std::vector<Landmark2D>& landmarks,
                             const std::filesystem::path& source_path);
std::vector<BfmMediaPipeCorrespondence> load_bfm_mediapipe_correspondences(
    const std::filesystem::path& path);
std::string bfm_mediapipe_correspondence_summary(
    const std::vector<BfmMediaPipeCorrespondence>& correspondences,
    const std::filesystem::path& source_path);

}  // namespace face_reconstruction
