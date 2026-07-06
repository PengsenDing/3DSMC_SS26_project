# Face Reconstruction

Classical optimization-based 3D face reconstruction for the TUM IN2354 3D
Scanning and Motion Capture course: fit the Basel Face Model (BFM 2019) to a
single photograph using MediaPipe landmarks, Ceres, and a custom CPU
rasterizer.

## Setup

1. Download `model2019_face12.h5` from the
   [BFM 2019 website](https://faces.dmi.unibas.ch/bfm/bfm2019.html) and place
   it at `data/model2019_face12.h5`.
2. Build the C++ side (requires Eigen, Ceres, glog, HDF5, and OpenCV; CMake
   fetches CLI11/HighFive/nlohmann-json automatically):
   ```bash
   cmake -S . -B build
   cmake --build build --parallel
   ```
   This produces `build/face_reconstruction`.
3. Set up the Python side (MediaPipe landmark detection):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
4. Download Google's MediaPipe multiclass segmentation model (the model file
   is ignored by Git):
   ```bash
   curl -L https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite \
     -o models/selfie_multiclass_256x256.tflite
   ```

## Run

```bash
source .venv/bin/activate
python reconstruct.py inputs/face.jpg
```

This detects 40 semantic landmark constraints, fits
camera/identity/expression, removes self-occluded semantic constraints with
the rasterizer's Z-buffer, refines identity against a dense face silhouette,
fits appearance (albedo + illumination), fits per-vertex RGB color, and
exports the mesh and diagnostics. Output goes to one self-contained directory:

```text
reconstructions/face/
├── face.off                  geometry only, open in MeshLab
├── face.ply                  colored mesh
├── fitting.txt               semantic solver report and RMSE
├── silhouette_fitting.txt    silhouette Chamfer/IoU report
├── bfm_surface_overlay.png    translucent fitted BFM surface on the photo
├── rendered_final_overlay.png the result, composited on the input photo
└── run.json
```

Add `--diagnostics` to also keep landmark/visibility/residual debug images.
By default, MediaPipe's six-class image segmenter extracts the face-skin class
and saves a filled, largest-component mask. This replaces the old face-oval
polygon, which does not follow the nose and lips in profile views. Pass
`--silhouette-mask mask.png` to use a manually corrected or external mask;
the mask must have the photograph's dimensions.
Self-occlusion filtering is enabled by default. Pass
`--no-landmark-visibility-filter` only for comparison/debugging, or adjust the
inverse-depth comparison with `--landmark-depth-tolerance` (default `1e-4`).
At strong yaw, landmarks whose semantic name belongs to the far side are also
removed (`--pose-yaw-threshold`, default `0.55` radians). Gross detections
outside the face-skin mask are rejected. Boundary landmarks such as the nose
tip, chin and mouth corners are instead snapped to the nearest mask boundary
when the correction is small, and receive a stronger fitting weight.

For difficult cases, pass `--landmark-overrides corrections.csv`. The CSV may
correct or disable individual MediaPipe points:

```csv
index,x_norm,y_norm,enabled
4,0.918,0.442,true
291,,,false
```

Coordinates are normalized to `[0,1]`. Disabled rows are removed before BFM
fitting, and all applied changes are recorded in `landmark_overrides.json`.
Run `python reconstruct.py --help` or `./build/face_reconstruction --help` for
the full list of tuning flags (regularization weights, PCA component counts,
pixel strides, etc.).

Other useful commands:

```bash
# Inspect the BFM model itself
./build/face_reconstruction --check-bfm

# Open face.off or face.ply in MeshLab to inspect the reconstructed mesh
```

## Video Tracking and One-Shot Reenactment

Video tracking and reenactment are optional modules; the single-image command
above remains unchanged.

```bash
# Track pose, BFM expression, and observed/fitted left/right eye opening.
python track_sequence.py inputs/talking.mp4 --output sequences/talking

# Animate one reconstructed photograph with expression, explicit eyelid
# correction, and source-relative SO(3) head pose.
python transfer_expression.py reconstructions/target_3 sequences/talking \
  --output transfers/talking_to_target3
```

Useful ablations are `--no-pose`, `--pose-only`,
`--no-blink-correction`, `--blink-scale`, and `--expression-scale`. The
tracker stores the observed and BFM-fitted opening of both eyes in
`tracking/tracking.csv`. Their residual drives independent smooth eyelid
correctives, bypassing blink/wide-eye motion that BFM 2019's expression PCA
cannot represent.

Reenactment uses `--texture-mode projective` by default: reference-frame UVs
are projected once and every output pixel bilinearly samples the original
target photograph. This avoids the triangular moire produced when an
under-constrained single view is represented by tens of thousands of
independently fitted vertex colors. Use `--texture-mode vertex` only for the
old fitted-PLY ablation.

Pass `--save-conditions` to export `coarse_rgb`, `depth`, `normals`, and
`visibility` images for every frame. An optional learned renderer can be
connected without changing the 3D pipeline:

```bash
python transfer_expression.py reconstructions/target_3 sequences/talking \
  --save-conditions \
  --completion-command \
  'python neural_renderer.py --target {target} --coarse {coarse} --depth {depth} --normal {normal} --mask {mask} --output {output}'
```

The command is an explicit backend protocol, not a bundled pretrained model;
without it, deterministic background inpainting and feathered CPU
compositing provide the reproducible baseline. See
`notes/full_pose_blink_aware_reenactment.md` for the research questions,
Face2Face comparison, and evaluation plan.

## Pipeline Steps

This is organized by *what happens when* (the order `reconstruct.py` actually
runs things), not by directory:

```text
0. Python entry point
   reconstruct.py
   One command that drives the whole pipeline: it calls step 1 to detect
   landmarks and invokes the compiled `face_reconstruction` executable to run
   steps 2-9. It collects every output file into one
   reconstructions/<name>/ folder.
        ↓
1. Landmark detection
   scripts/detect_landmarks.py
   Runs Google's MediaPipe Face Landmarker on the input photo. Outputs a CSV
   with 478 rows: each detected facial point's index, pixel coordinates, and
   normalized image coordinates. This is the only step that actually looks at
   the photo to find facial features -- every later step works only with
   these point coordinates and the photo's raw pixel colors.

   scripts/segment_face.py
   Runs MediaPipe multiclass image segmentation and extracts class 3
   (face skin). The largest component is closed and filled to create the
   silhouette used by step 4. An external/manual mask can override it.
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
   stages. It first checks observations against the face-skin mask: gross detections are
   removed, while nearby boundary landmarks can be snapped to the observed
   mask. The camera-only stage then supplies a coarse pose. At strong yaw,
   semantic landmarks on the far side are removed. Finally, the mean BFM is
   rasterized from that pose and each remaining semantic vertex is compared
   with the inverse-depth Z-buffer in a 3x3 pixel neighborhood. Landmarks
   behind a nearer facial surface are removed before the joint solve and its
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
   bfm_surface_overlay.png (a translucent normal-shaded BFM surface generated
   by the CPU rasterizer), overlay.png (detected vs. projected landmarks,
   to check step 3's accuracy), raster_depth.png (visualized depth buffer, to check step 5),
   visibility.png (visible-face mask), and silhouette target/initial/refined
   masks plus a colored boundary overlay. Only generated when --diagnostics
   is passed.
```

Directories not part of the runtime pipeline above:

```text
tests/            Regression tests; run with `ctest --test-dir build`
data/             BFM model file, active correspondence CSV, and test datasets
inputs/           Source photographs only
reconstructions/  One output directory per input photo (see `Run` above)
notes/            Running work-log notes
CMakeLists.txt    Build configuration
```
