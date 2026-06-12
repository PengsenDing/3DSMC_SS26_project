#pragma once

#include "face_reconstruction/mesh.hpp"

#include <filesystem>

namespace face_reconstruction {

Mesh load_obj_mesh(const std::filesystem::path& path);

}  // namespace face_reconstruction
