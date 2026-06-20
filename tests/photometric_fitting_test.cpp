#include "face_recon/photometric_fitting.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>

int main() {
  Eigen::VectorXd vertices(12);
  vertices <<
      -1.0, -1.0, 0.0,
       1.0, -1.0, 0.0,
       1.0,  1.0, 0.0,
      -1.0,  1.0, 0.0;
  Eigen::MatrixXi triangles(2, 3);
  triangles << 0, 1, 2,
               0, 2, 3;

  const Eigen::VectorXd normals =
      face_recon::ComputeVertexNormals(vertices, triangles);
  if (normals.size() != vertices.size()) {
    std::cerr << "Normal vector has the wrong size\n";
    return 1;
  }
  for (int vertex = 0; vertex < 4; ++vertex) {
    const Eigen::Vector3d normal = normals.segment<3>(3 * vertex);
    if (std::abs(normal.norm() - 1.0) > 1.0e-8 ||
        normal.z() < 0.999) {
      std::cerr << "Planar vertex normal is incorrect: "
                << normal.transpose() << "\n";
      return 1;
    }
  }
  std::cout << "Photometric normal test passed\n";
  return 0;
}
