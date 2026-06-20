# Face Reconstruction

Classical optimization-based 3D face reconstruction for the TUM IN2354 3D
Scanning and Motion Capture course.

The repository combines two parts of the reconstruction pipeline:

- Basel Face Model 2019 loading and inspection.
- MediaPipe landmark detection, correspondence loading, and an OBJ viewer.

## Build

The C++ project requires Eigen, Ceres Solver, glog, HDF5, OpenCV, OpenGL,
GLEW, and SFML. CMake downloads CLI11, HighFive, and nlohmann/json during the
first configuration.

```bash
cmake -S . -B build
cmake --build build --parallel
```

The build produces two executables:

- `build/face_reconstruction`: fits and exports the Basel Face Model.
- `build/landmark_viewer`: views and renders OBJ/PLY meshes.

## One-Command Reconstruction

Activate the Python environment, then provide one input photograph:

```bash
source .venv/bin/activate
python reconstruct.py inputs/face.jpg
```

This single command performs:

```text
MediaPipe landmark detection
→ BFM camera/identity/expression fitting
→ OFF and colored PLY export
→ fitting diagnostics
```

Use `--render` to also generate albedo, depth, normal, and checkerboard images:

```bash
python reconstruct.py inputs/face.jpg --render
```

Every image gets one self-contained directory:

```text
reconstructions/face/
├── landmarks.csv
├── landmarks.png
├── face.off
├── face.ply
├── face_aligned.ply
├── fitting.txt
├── reprojections.csv
├── overlay.png
├── run.json
└── renders/                 # only with --render
    ├── albedo.png
    ├── depth.png
    ├── normal.png
    └── checkerboard.png
```

- Open `face.off` in MeshLab to inspect geometry.
- Open `face_aligned.ply` to inspect the photo-colored fitted mesh.
- Check `overlay.png` before judging the reconstruction quality.
- Re-running the same image updates the same directory instead of scattering
  additional files around the repository.

Useful pipeline options:

```text
--name NAME          Override the run directory name
--output-root DIR    Override the reconstructions/ parent directory
--model PATH         Override the BFM HDF5 file
--render             Export the four rendering modes
--verbose-optimization
```

## Basel Face Model

Download `model2019_face12.h5` from the
[BFM 2019 website](https://faces.dmi.unibas.ch/bfm/bfm2019.html) and place it
at:

```text
data/model2019_face12.h5
```

Inspect the model and export validation files:

```bash
./build/face_reconstruction --check-bfm
```

Useful options:

```text
-c, --check-bfm     Print model dimensions and export validation files
-m, --model PATH    Override the BFM HDF5 path
-o, --output DIR    Choose the output directory (default: results)
```

## Reconstruction Internals

The C++ fitting stage uses a weak-perspective camera, BFM identity
coefficients, and BFM expression coefficients. `reconstruct.py` connects this
stage to MediaPipe, so the lower-level commands normally do not need to be run
manually.

The personalized geometry is

```text
V = mean_shape + mean_expression
    + shape_basis * sqrt(shape_variance) * normalized_shape_coefficients
    + expression_basis * sqrt(expression_variance)
      * normalized_expression_coefficients
```

The weak-perspective camera projects a rotated vertex `V` to normalized image
coordinates:

```text
x = scale * (R * V).x + translation_x
y = -scale * (R * V).y + translation_y
```

Ceres first optimizes only the camera. It then optimizes identity before
jointly optimizing camera, identity, and expression. The fitter uses robust
landmark reprojection error, Gaussian PCA priors, and 21 dynamically assigned
face-contour points. Incorrect semantic correspondences are rejected using
their reprojection residuals. Normalized PCA coefficients are constrained to
three standard deviations.

When `--image` is supplied, the fitted camera projects every mesh vertex into
the input photograph and samples a vertex color. These colors preserve
identity cues such as eyebrows, eye color, facial hair, and skin appearance.
They are photo colors with baked-in illumination, not intrinsic albedo.

Useful fitting options:

```text
--shape-components N
--expression-components N
--shape-regularization WEIGHT
--expression-regularization WEIGHT
--landmark-weight WEIGHT
--contour-weight WEIGHT
--outlier-threshold ERROR
--contour-refinements N
--verbose-optimization
```

Open `face.off` directly in MeshLab. For visual comparison, open
`face_aligned.ply`, enable vertex colors in MeshLab, or render it:

```bash
./build/landmark_viewer reconstructions/face/face_aligned.ply \
  --render-all reconstructions/face/renders
```

This is still a single-view reconstruction. Photo colors improve appearance,
but hidden geometry, hair, ears, and true depth cannot be recovered reliably
from one frontal photograph. Full differentiable RGB/albedo/illumination
optimization remains a later refinement stage.

## Standalone Landmark Detection

The detector remains available separately for debugging:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
python scripts/detect_landmarks.py \
  --image inputs/face.jpg \
  --csv /tmp/face_landmarks.csv \
  --debug /tmp/face_landmarks.png
```

The main CSV contains all detected MediaPipe landmarks:

```csv
index,name,u,v,z_norm,x_norm,y_norm
1,nose_tip,512.3,421.8,-0.038,0.500,0.412
```

The reusable BFM-to-MediaPipe mapping is stored in
`data/bfm_mediapipe_correspondence.csv`. When `--bfm-csv` is supplied, the
script joins the per-image detections with this mapping.

The runtime correspondence file deliberately contains only three columns:

```csv
bfm_landmark_name,bfm_vertex_id,mediapipe_index
center.nose.tip,15841,4
```

These are the only values needed to associate a detected 2D MediaPipe point
with a vertex in the BFM mesh. The current table contains 40 correspondences.
The detector also reads semantic landmark names from this table, so landmark
names and correspondence indices have a single source of truth.

The original eight-column matching result is preserved in
`data/bfm_mediapipe_matches_reference.csv` for auditing and visualization. Its
`mp_u` and `mp_v` values belong to the particular image used during matching,
so they must not be used as landmark observations for a new input image. The
`bfm_X`, `bfm_Y`, and `bfm_Z` values are reference coordinates that can be
recovered from the BFM vertex ID and are therefore not required by the runtime
pipeline.

## Landmark and OBJ Verification

Load a mesh together with landmark data:

```bash
./build/landmark_viewer data/model.obj \
  --info \
  --landmarks reconstructions/face/landmarks.csv \
  --correspondences data/bfm_mediapipe_correspondence.csv
```

Other useful viewer options:

```text
--deps       Print linked dependency versions
--frames N   Render N frames and exit
--info       Print summaries without opening a window
--help       Show all options
```

## Basic Rendering

The C++ viewer supports OBJ meshes and ASCII PLY meshes. PLY vertex colors
exported by the BFM tool are used as albedo; meshes without colors receive a
neutral fallback color.

Open the interactive viewer and switch rendering modes with number keys:

```bash
./build/landmark_viewer data/model.obj
```

```text
1  Albedo
2  Linear camera-space depth
3  Camera-space normals encoded as RGB
4  Procedural checkerboard
S  Save the current mode to the current working directory
```

Render one mode to a PNG and exit:

```bash
./build/landmark_viewer data/model.obj \
  --mode normal \
  --output /tmp/normal.png
```

Render all four modes from the same camera:

```bash
./build/landmark_viewer data/model.obj \
  --render-all /tmp/basic_rendering
```

To render the BFM mean albedo after exporting it, pass its colored PLY file:

```bash
./build/landmark_viewer reconstructions/face/face_aligned.ply \
  --render-all reconstructions/face/renders
```

## Project Layout

```text
reconstruct.py    One-command user entry point
face_recon/       BFM loading, fitting, image sampling, and mesh export
include/          C++ public headers
src/              Mesh loading, landmark CSV loading, and rendering
scripts/          Standalone MediaPipe preprocessing tools
data/             BFM model assets and correspondence table
inputs/           Original source photographs only
reconstructions/  One self-contained directory per input photograph
tests/            Synthetic fitting regression tests
```

Older generated files from before this layout are preserved under
`reconstructions/_legacy/`.
