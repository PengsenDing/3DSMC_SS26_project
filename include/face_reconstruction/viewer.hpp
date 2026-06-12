#pragma once

#include "face_reconstruction/mesh.hpp"

namespace face_reconstruction {

struct ViewerOptions {
    int max_frames = 0;
};

int run_viewer(const Mesh& mesh, const ViewerOptions& options = {});

}  // namespace face_reconstruction
