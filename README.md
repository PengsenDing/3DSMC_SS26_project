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

- `build/face_reconstruction`: loads and inspects the Basel Face Model.
- `build/landmark_viewer`: loads OBJ meshes and landmark/correspondence CSVs.

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

## Landmark Detection

Create a Python environment and install the preprocessing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Put an input image in `inputs/`, then run:

```bash
python scripts/detect_landmarks.py \
  --image inputs/face.jpg \
  --csv outputs/face_landmarks.csv \
  --bfm-csv outputs/bfm_mediapipe_landmarks.csv \
  --txt outputs/face_landmarks.txt \
  --debug outputs/face_landmarks.png
```

The main CSV contains all detected MediaPipe landmarks:

```csv
index,name,u,v,z_norm,x_norm,y_norm
1,nose_tip,512.3,421.8,-0.038,0.500,0.412
```

The reusable BFM-to-MediaPipe mapping is stored in
`data/bfm_mediapipe_correspondence.csv`. When `--bfm-csv` is supplied, the
script joins the per-image detections with this mapping.

## Landmark and OBJ Verification

Load a mesh together with generated landmark data:

```bash
./build/landmark_viewer data/model.obj \
  --info \
  --landmarks outputs/face_landmarks.csv \
  --correspondences data/bfm_mediapipe_correspondence.csv
```

Other useful viewer options:

```text
--deps       Print linked dependency versions
--frames N   Render N frames and exit
--info       Print summaries without opening a window
--help       Show all options
```

## Project Layout

```text
face_recon/    Basel Face Model loader and command-line entry point
include/       Headers for landmark loading, mesh loading, and viewing
src/           Landmark/viewer C++ implementation
scripts/       Python MediaPipe preprocessing
data/          Model assets and landmark correspondence table
inputs/        Local source images
outputs/       Generated landmark files and debug images
results/       Generated BFM meshes and reports
```
