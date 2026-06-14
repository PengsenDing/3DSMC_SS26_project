// Execution entry point for the 3D Face Reconstruction Pipeline.
// This binary handles command-line argument parsing, loads the statistical
// Basel Face Model, and orchestrates the facial alignment optimization loops.

#include "face_recon/bfm_model.h"

#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  // Instantiate the App parser framework.
  CLI::App app{"TUM 3D Scanning and Motion Capture: 3D Face Reconstruction Optimization Loop Pipeline"};

  // Set up runtime parameters with safe defaults.
  std::string model_path = "data/model2019_face12.h5";
  std::string output_dir = "results";
  bool check_bfm = false;

  // Define the available CLI flags and options.
  app.add_option("-m,--model", model_path, "Path to the Basel Face Model HDF5 asset container")->check(CLI::ExistingFile);
  app.add_option("-o,--output", output_dir, "Directory to save the reconstruction results");
  app.add_flag("-c,--check-bfm", check_bfm, "Run structural loading checks and output comprehensive database dumps");

  // Execute the parse routine safely
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    // Print the correct usage information and exit with the corresponding exit code on wrong input.
    return app.exit(e);
  }

  // Run the core execution pipeline.
  std::cout << "[INFO] Initializing Face Reconstruction Pipeline...\n";
  face_recon::BfmModel model;
  if (!model.LoadFromH5(model_path)) {
    std::cerr << "[ERROR] Failed to configure internal face model architecture from path: " << model_path << "\n";
    return 1;
  }

  // Ensure the output directory exists
  std::filesystem::create_directories(output_dir);

  // Orchestrate pipeline behavior based on runtime flags.
  if (check_bfm) {
    std::cout << "[INFO] Check BFM requested. Exporting validation logs and summaries...\n";
    
    // Output metrics to verify structural integrity.
    model.PrintModelSummary();

    // Generate static 3D face representations.
    model.SaveMeanMeshToPly(output_dir + "/bfm_mean_face.ply", false);
    model.SaveMeanMeshToPly(output_dir + "/bfm_mean_face_with_landmarks.ply", true);
    model.SaveLandmarksToTxt(output_dir + "/bfm_landmarks.txt");
  } else {
    std::cout << "[INFO] Model loaded successfully. Pass --check-bfm to inspect the structural assets.\n";
    // Future face reconstruction optimization will be orchestrated here.
  }

  return 0;
}