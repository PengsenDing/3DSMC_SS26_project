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
→ fixed-geometry albedo and illumination fitting
→ constrained dense identity/expression refinement
→ final albedo and illumination re-fitting
→ direct dense per-vertex RGB fitting
→ OFF and colored PLY export
→ compact final output
```

The default output is intentionally compact:

```text
reconstructions/face/
├── face.off
├── face.ply
├── fitting.txt
├── rendered_final.png
├── rendered_final_overlay.png
└── run.json
```

Use `--render` to add the four basic C++ rendering modes:

```bash
python reconstruct.py inputs/face.jpg --render
```

```text
renders/
    ├── albedo.png
    ├── depth.png
    ├── normal.png
    └── checkerboard.png
```

- Open `face.off` in MeshLab to inspect geometry.
- Open `face.ply` to inspect the photo-colored fitted mesh.
- `rendered_final.png` is the primary sample-style result: visible vertex RGB
  values are fitted directly to minimize rendered-versus-input pixel error.
- Re-running the same image updates the same directory instead of scattering
  additional files around the repository.

Use `--diagnostics` when detailed inspection is needed. It additionally saves
landmarks, reprojections, visibility/depth buffers, masks, residuals,
intrinsic albedo/illumination results, dense-refinement reports, and
`face_aligned.ply`.

Useful pipeline options:

```text
--name NAME          Override the run directory name
--output-root DIR    Override the reconstructions/ parent directory
--model PATH         Override the BFM HDF5 file
--albedo-components  Number of normalized color PCA coefficients
--focal-regularization Perspective focal-length prior weight
--photometric-stride Pixel subsampling used for appearance fitting
--texture-stride     Pixel subsampling used for direct RGB fitting
--texture-prior      Prior weight for initialized vertex colors
--texture-smoothness Mesh-edge smoothness for fitted vertex colors
--no-dense-refinement Skip dense geometry refinement
--dense-resolution   Maximum refinement image dimension
--render             Export the four rendering modes
--diagnostics        Export detailed intermediate and debugging artifacts
--verbose-optimization
```

## Perspective Baseline

Before changing the camera model, run a small regression baseline with several
different photographs:

```bash
./.venv/bin/python scripts/evaluate_perspective.py \
  inputs/1.jpg \
  inputs/2.png \
  inputs/exp_face_converted.jpg
```

The script runs the normal reconstruction pipeline independently for every
image and stores the results under:

```text
reconstructions/_baselines/perspective/
├── 1/
├── 2/
├── exp_face_converted/
├── summary.csv
├── summary.md
└── baseline.json
```

The summary records pixel-space semantic and contour RMSE, fitted focal length
and camera distance, rejected landmarks, normalized identity/expression
coefficients, and visible rasterized area. It does not revalidate the
MediaPipe-to-BFM correspondence table.

Pass `--render` if the four viewer diagnostic images are also needed. Future
side-view photographs can be appended to the same command without changing
the script.

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

The C++ fitting stage uses a perspective camera, BFM identity
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

For a rotated vertex `(X,Y,Z)`, normalized image coordinates are:

```text
depth = translation_z - Z
x = 0.5 + focal_length * (X + translation_x) / depth
y = 0.5 - focal_length * aspect_ratio
          * (Y + translation_y) / depth
```

The focal length is normalized by image width. Ceres optimizes rotation, 3D
translation, focal length, identity, and expression. A light focal-length
prior prevents the single-image focal/depth ambiguity from collapsing to an
unrealistic telephoto solution.

Ceres first optimizes only the camera. It then optimizes identity before
jointly optimizing camera, identity, and expression. The fitter uses robust
landmark reprojection error, Gaussian PCA priors, and 21 dynamically assigned
face-contour points. Incorrect semantic correspondences are rejected using
their reprojection residuals. Normalized PCA coefficients are constrained to
three standard deviations.

When `--image` is supplied, the fitted camera projects every mesh vertex into
the input photograph and samples a vertex color. A CPU triangle rasterizer
first computes a Z-buffer, triangle IDs, and barycentric coordinates for every
covered image pixel. Only vertices consistent with the visible Z-buffer
surface receive photo colors; occluded vertices retain BFM albedo. This avoids
painting background or clothing onto hidden facial surfaces. The sampled
colors preserve
identity cues such as eyebrows, eye color, facial hair, and skin appearance.
They are photo colors with baked-in illumination, not intrinsic albedo.

### Why rasterization is required

Projection alone answers where a 3D vertex lands in the image, but multiple
surfaces can project to the same pixel. Rasterization evaluates complete
triangles and resolves this ambiguity using depth:

```text
3D triangle
→ projected screen triangle
→ covered pixels and barycentric coordinates
→ interpolated camera-space depth
→ nearest surface selected by the Z-buffer
```

For the current frontal BFM convention, larger camera-space `z` is closer.
Each output pixel stores:

```text
triangle ID
depth
barycentric coordinates
```

These buffers are required for visibility-aware texture sampling and will also
provide the fixed pixel-to-surface correspondences used by dense photometric
fitting.

### Analysis by synthesis

After landmark fitting, geometry and the perspective camera are fixed.
The rasterizer maps every visible image pixel to a triangle and barycentric
coordinates. Smooth fitted-mesh normals and BFM albedo are then interpolated
at those pixels.

The appearance model is:

```text
albedo(beta) =
    mean_albedo
    + color_basis * sqrt(color_variance) * normalized_albedo_coefficients

lighting(normal, gamma) =
    gamma_0 + gamma_1 * nx + gamma_2 * ny + gamma_3 * nz

rendered_rgb = albedo(beta) * lighting(normal, gamma)
```

The implementation first fits 12 first-order spherical-harmonics illumination
parameters (four per RGB channel), then alternates between 30 normalized BFM
albedo coefficients and illumination. It uses visible face pixels, Huber
weights, a Gaussian PCA prior, and `±3σ` albedo bounds. The optimization does
not alter pose or geometry, which keeps appearance from compensating for
unstable shape updates.

With `--diagnostics`, `photometric.txt` records the initial,
illumination-only, and final RGB RMSE along with all fitted appearance
parameters.

### Direct RGB texture fitting

The primary visual output follows the sample project's objective:

```text
minimize sum_pixels ||barycentric(vertex_rgb) - input_rgb||²
```

With geometry and visibility fixed, rendered color is linear in the three
vertex colors of each visible triangle. The implementation therefore assembles
one sparse least-squares system for all visible pixels and solves the RGB
channels directly. A weak initialization prior and mesh-edge smoothness keep
unobserved or poorly constrained vertices stable. The rasterized face is also
intersected with the detected MediaPipe face oval so geometry mismatch near
the cheek or jaw does not bake background pixels into the mesh. `face.ply`,
`rendered_final.png` and
`rendered_final_overlay.png` use these fitted, illumination-baked colors.
With `--diagnostics`, the intrinsic BFM result is also available as
`face_albedo.ply` and `rendered_intrinsic.png`.

### Dense geometry refinement

After the first appearance fit, the pipeline performs a conservative
low-resolution refinement of the leading identity and expression coefficients.
For every candidate update it:

```text
regenerates the BFM vertices
→ recomputes smooth camera-space normals
→ rerasterizes the complete mesh
→ reevaluates landmark, silhouette, photometric, and PCA-prior losses
```

The CPU rasterizer is not differentiable, so this stage uses bounded numerical
coordinate search instead of back-propagation. By default it refines six
identity and six expression coefficients in one pass at a maximum image
dimension of 192 pixels. Eye, eyebrow, nose, and mouth pixels receive larger
photometric weights. An update is accepted only if the combined objective
decreases and landmark RMSE remains within 5% of its starting value.

After geometry refinement, visibility is recomputed at full resolution and
albedo/illumination are fitted again. With `--diagnostics`,
`dense_refinement.txt` records the loss components and the pre-refinement
appearance is preserved under `initial_appearance/`.

This stage generally provides a modest improvement rather than a photographic
identity match: low-dimensional BFM geometry cannot reproduce hair, iris
detail, exact eyelids, or arbitrary local anatomy, and the albedo PCA space
cannot reproduce all high-frequency facial texture.

Useful fitting options:

```text
--shape-components N
--expression-components N
--shape-regularization WEIGHT
--expression-regularization WEIGHT
--focal-regularization WEIGHT
--landmark-weight WEIGHT
--contour-weight WEIGHT
--outlier-threshold ERROR
--contour-refinements N
--verbose-optimization
```

Open `face.off` directly in MeshLab. For visual comparison, open `face.ply`,
enable vertex colors in MeshLab, or render it:

```bash
./build/landmark_viewer reconstructions/face/face.ply \
  --render-all reconstructions/face/renders
```

This is still a single-view reconstruction. Hidden geometry, hair, ears, and
true depth cannot be recovered reliably from one frontal photograph. The
appearance stage optimizes only the visible BFM face and cannot reproduce
high-frequency details outside the available PCA albedo space.

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

To render the reconstructed photo-colored mesh:

```bash
./build/landmark_viewer reconstructions/face/face.ply \
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
