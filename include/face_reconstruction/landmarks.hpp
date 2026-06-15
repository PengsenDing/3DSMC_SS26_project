#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace face_reconstruction {

struct Landmark2D {
    int index = 0;
    float x_px = 0.0f;
    float y_px = 0.0f;
    float z_norm = 0.0f;
    float x_norm = 0.0f;
    float y_norm = 0.0f;
};

std::vector<Landmark2D> load_landmarks_csv(const std::filesystem::path& path);
std::string landmark_summary(const std::vector<Landmark2D>& landmarks,
                             const std::filesystem::path& source_path);

}  // namespace face_reconstruction
