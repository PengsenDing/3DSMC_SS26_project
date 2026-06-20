#pragma once

#include "face_reconstruction/mesh.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace face_reconstruction {

enum class RenderMode {
    Albedo = 0,
    Depth = 1,
    Normal = 2,
    Checkerboard = 3,
};

struct ViewerOptions {
    int max_frames = 0;
    RenderMode initial_mode = RenderMode::Albedo;
    std::optional<std::filesystem::path> screenshot_path;
    std::optional<std::filesystem::path> render_all_directory;
};

[[nodiscard]] std::string render_mode_name(RenderMode mode);
[[nodiscard]] RenderMode parse_render_mode(std::string_view name);
int run_viewer(const Mesh& mesh, const ViewerOptions& options = {});

}  // namespace face_reconstruction
