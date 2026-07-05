# Face Reconstruction

Classical optimization-based 3D face reconstruction for the TUM IN2354 3D
Scanning and Motion Capture course: fit the Basel Face Model (BFM 2019) to a
single photograph using MediaPipe landmarks, Ceres, a custom CPU rasterizer,
and a standalone OpenGL mesh viewer.

## Setup

1. Download `model2019_face12.h5` from the
   [BFM 2019 website](https://faces.dmi.unibas.ch/bfm/bfm2019.html) and place
   it at `data/model2019_face12.h5`.
2. Build the C++ side (requires Eigen, Ceres, glog, HDF5, OpenCV, OpenGL,
   GLEW, SFML; CMake fetches CLI11/HighFive/nlohmann-json automatically):
   ```bash
   cmake -S . -B build
   cmake --build build --parallel
   ```
   This produces `build/face_reconstruction` (fitting) and
   `build/landmark_viewer` (mesh viewer).
3. Set up the Python side (MediaPipe landmark detection):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

## Run

```bash
source .venv/bin/activate
python reconstruct.py inputs/face.jpg --render
```

This detects 40 semantic landmark constraints, fits
camera/identity/expression, removes self-occluded semantic constraints with
the rasterizer's Z-buffer, refines identity against a dense face silhouette,
fits appearance (albedo + illumination), fits per-vertex RGB color, exports
the mesh, and (with `--render`) renders the four basic viewer modes. Output
goes to one self-contained directory:

```text
reconstructions/face/
├── face.off                  geometry only, open in MeshLab
├── face.ply                  colored mesh
├── fitting.txt               semantic solver report and RMSE
├── silhouette_fitting.txt    silhouette Chamfer/IoU report
├── rendered_final_overlay.png the result, composited on the input photo
├── run.json
└── renders/                   albedo.png, depth.png, normal.png, checkerboard.png
```

Add `--diagnostics` to also keep landmark/visibility/residual debug images.
Pass `--silhouette-mask mask.png` to use an external face-segmentation mask;
otherwise the detector rasterizes MediaPipe's ordered face oval into a dense
binary mask. The external mask should have the photograph's aspect ratio.
The oval points are not used as sparse BFM correspondences.
Self-occlusion filtering is enabled by default. Pass
`--no-landmark-visibility-filter` only for comparison/debugging, or adjust the
inverse-depth comparison with `--landmark-depth-tolerance` (default `1e-4`).
Run `python reconstruct.py --help` or `./build/face_reconstruction --help` for
the full list of tuning flags (regularization weights, PCA component counts,
pixel strides, etc.).

Other useful commands:

```bash
# Inspect the BFM model itself
./build/face_reconstruction --check-bfm

# View/render any OBJ or colored ASCII PLY mesh standalone
./build/landmark_viewer data/model.obj            # interactive (keys 1-4, S to save)
./build/landmark_viewer data/model.obj --render-all /tmp/out

# Batch-run several photos for a quick regression comparison
./.venv/bin/python scripts/evaluate_perspective.py inputs/1.jpg inputs/2.png
```

## Pipeline Steps

This is organized by *what happens when* (the order `reconstruct.py` actually
runs things), not by directory:

```text
0. Python entry point
   reconstruct.py
   One command that drives the whole pipeline: it calls step 1 to detect
   landmarks, invokes the compiled `face_reconstruction` executable to run
   steps 2-9, and (if --render is passed) invokes `landmark_viewer` for step
   10. Collects every output file into one reconstructions/<name>/ folder.
        ↓
1. Landmark detection
   scripts/detect_landmarks.py
   Runs Google's MediaPipe Face Landmarker on the input photo. Outputs a CSV
   with 478 rows: each detected facial point's index, pixel coordinates, and
   normalized image coordinates. This is the only step that actually looks at
   the photo to find facial features -- every later step works only with
   these point coordinates and the photo's raw pixel colors. It also creates
   a binary face-oval mask for step 4 unless an external segmentation mask was
   supplied.
        ↓
2. Load the BFM model + landmark data
   face_recon/bfm_model.h / .cc
   Opens model2019_face12.h5 and reads the BFM's mean face, the PCA basis
   matrices and variances for shape/expression/color, and the triangle list
   (which 3 vertex indices form each of the 55,040 triangles).

   include/face_reconstruction/landmarks.hpp + src/landmarks.cpp
   Reads the landmark CSV from step 1 and the BFM-to-MediaPipe correspondence
   table (40 rows, e.g. "MediaPipe point #4 = BFM vertex #15841 = nose tip").
   This table is the only link between a detected 2D point and a 3D model
   vertex.
        ↓
3. Camera + identity + expression fitting (sparse landmark loss)
   face_recon/fitting.h / .cc
   tests/fitting_test.cpp   (regression test)
   The core step. Given the 2D landmark positions (steps 1-2) and the BFM
   mean face + shape/expression basis (step 2), this solves for the unknowns
   -- camera rotation/translation/focal length, 199 identity coefficients,
   and 100 expression coefficients -- using Ceres least-squares. It adjusts
   every unknown so that projecting the current 3D face through the current
   camera lands as close as possible to the detected 2D points, in two
   stages. The camera-only stage first supplies a coarse pose. The mean BFM is
   then rasterized from that pose and each semantic vertex is compared with
   the inverse-depth Z-buffer in a 3x3 pixel neighborhood. Landmarks behind a
   nearer facial surface (for example, the far eye in a profile image) are
   removed before the joint camera/identity/expression solve and its automatic
   residual-based outlier rejection. If the mesh has no topology or fewer
   than six landmarks remain visible, the filter safely falls back to the
   original constraints. Output: this person's personalized 3D mesh vertices
   + fitted camera. Only rows in the 40-point correspondence table
   participate; no jaw/oval point-to-vertex constraints are added. The fitting
   report records whether visibility filtering ran and how many constraints it
   removed.
        ↓
4. View-dependent silhouette geometry fitting
   face_recon/silhouette_fitting.h / .cc
   tests/silhouette_fitting_test.cpp
   Renders the current mesh silhouette, samples its visible boundary, and
   forms bidirectional nearest-boundary matches against the target mask.
   Ceres refines identity coefficients while the semantic landmarks and BFM
   PCA prior act as safeguards. The silhouette is rerendered and rematched
   after every outer iteration, so no fixed BFM jaw correspondence is used.
   Output: refined geometry plus Chamfer and IoU diagnostics.
        ↓
5. Rasterization (visibility test, used by the next two steps)
   face_recon/rasterizer.h / .cc
   tests/rasterizer_test.cpp
   Used once after the coarse camera fit for landmark self-occlusion and again
   with the refined mesh after step 4. It answers a geometric
   question: "if I render this mesh from this camera, which triangle is
   visible at each pixel, and exactly where inside that triangle does the
   pixel fall?" It projects every vertex to 2D, scans each triangle's pixel
   range to compute barycentric coordinates, and keeps only the nearest
   surface where triangles overlap (Z-buffer). Output: for every pixel, its
   covering triangle ID, depth, and barycentric weights -- the pixel-to-3D-
   surface lookup table used by steps 6, 7, and 9.
        ↓
6. Appearance fitting (albedo + illumination separation)
   face_recon/photometric_fitting.h / .cc
   tests/photometric_fitting_test.cpp
   Using step 5's pixel-to-surface mapping, looks at the actual photo colors
   and asks: "what skin albedo (BFM color PCA coefficients) and what
   lighting (4 spherical-harmonics coefficients per RGB channel) would
   explain the observed pixel colors?" Solved by alternating least squares:
   fit albedo with lighting fixed, then fit lighting with albedo fixed,
   repeat. Splitting albedo from lighting (instead of fitting one flat "this
   is the color" term) keeps the photo's shadows/highlights from getting
   permanently baked into the estimated skin color.
        ↓
7. Dense per-vertex RGB fitting (final visual result)
   face_recon/texture_fitting.h / .cc
   tests/texture_fitting_test.cpp
   With geometry and visibility already fixed (steps 4-5), this solves
   directly for the RGB color of every vertex so that the interpolated
   render matches the photo at every visible pixel -- assembled as one
   sparse linear least-squares system (with a smoothness term and a prior
   toward the step-6 result), solved directly without Ceres. This is the
   single biggest quality jump in the pipeline: RGB error drops from 0.089
   to 0.013, and it's the main reason rendered_final.png looks close to a
   real photo.
        ↓
8. Mesh export + fitting report
   face_recon/mesh_export.h / .cc
   Pure I/O -- computes nothing new. Writes the step-4 vertices plus
   whichever colors are available (step 6 or 7) to disk as .off (geometry
   only, for MeshLab) and .ply (geometry + color), and writes fitting.txt
   with the solver's RMSE numbers and fitted parameters.
        ↓
9. Diagnostic image export (overlay / depth / visibility / silhouette)
   face_recon/image_fitting.h / .cc
   Renders the intermediate results from earlier steps as PNGs purely for
   inspection -- none of this feeds back into the mesh or its colors:
   overlay.png (detected vs. projected landmarks, to check step 3's
   accuracy), raster_depth.png (visualized depth buffer, to check step 5),
   visibility.png (visible-face mask), and silhouette target/initial/refined
   masks plus a colored boundary overlay. Only generated when --diagnostics
   is passed.
        ↓
10. (Optional, only runs with --render) Four rendering modes
   include/face_reconstruction/viewer.hpp + src/viewer.cpp
   obj_loader.* / ply_loader.* / mesh.*   (mesh file parsing)
   app.* + main.cpp                       (landmark_viewer's CLI)
   A fully separate executable, `landmark_viewer`, that re-reads the
   step-8 face.ply through a real OpenGL pipeline and renders it as
   albedo/depth/normal/checkerboard. It has no data dependency on steps 1-9
   -- it can render any OBJ/PLY file, not just ones this project produced.
```

Directories not part of the runtime pipeline above:

```text
tests/            All 5 regression tests referenced above; run with `ctest --test-dir build`
data/             BFM model file (you provide this), correspondence CSVs, sample mesh
inputs/           Source photographs only
reconstructions/  One output directory per input photo (see `Run` above)
notes/            Running work-log notes
CMakeLists.txt    Build configuration
```
