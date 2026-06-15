# Face Reconstruction

Classical optimization based face reconstruction project for the IN2354 3D Scanning and Motion Capture course.

## Dependencies

The project currently links the dependencies needed for the planned reconstruction pipeline:

- Eigen 3.4.0: matrix/vector math and linear algebra.
- OpenCV 4.11.0: image loading, image processing, and later RGB/RGB-D input handling.
- Ceres Solver 2.2.0: nonlinear least squares optimization for later alignment and model fitting.
- OpenGL: rendering backend for the week-1 face model viewer.
- GLEW 2.2.0 package: OpenGL extension loading. The installed headers report 2.3.4.
- SFML 3.0.2: window creation and OpenGL context management.

GLFW is not currently installed on this machine, so SFML is used as the window/context layer.


## Project Layout

```text
.
├── CMakeLists.txt
├── data/
│   └── model.obj
├── inputs/
├── include/
│   └── face_reconstruction/
│       ├── app.hpp
│       ├── landmarks.hpp
│       ├── mesh.hpp
│       ├── obj_loader.hpp
│       └── viewer.hpp
├── outputs/
├── requirements.txt
├── scripts/
│   └── detect_landmarks.py
├── src/
│   ├── app.cpp
│   ├── landmarks.cpp
│   ├── mesh.cpp
│   ├── obj_loader.cpp
│   ├── viewer.cpp
│   └── main.cpp
```

## File Overview

- `CMakeLists.txt`: defines the C++20 project, finds dependencies, builds the core library, and builds the executable.
- `include/face_reconstruction/app.hpp`: declares the top-level application functions.
- `src/app.cpp`: handles command-line options, loads meshes, prints dependency/mesh info, and starts the viewer.
- `src/main.cpp`: small program entry point with error handling.
- `include/face_reconstruction/landmarks.hpp`: defines the 2D landmark data structure and CSV loading API.
- `src/landmarks.cpp`: loads MediaPipe landmark CSV files and prints summary statistics for C++ handoff checks.
- `include/face_reconstruction/mesh.hpp`: defines the `Mesh` and `Triangle` data structures.
- `src/mesh.cpp`: implements mesh helper functions such as center, extent, bounding radius, and summary output.
- `include/face_reconstruction/obj_loader.hpp`: declares the OBJ loading function.
- `src/obj_loader.cpp`: reads OBJ files, parses vertices/faces, supports common OBJ face formats, and triangulates polygon faces.
- `include/face_reconstruction/viewer.hpp`: declares viewer options and the viewer entry point.
- `src/viewer.cpp`: creates the SFML/OpenGL window, renders the mesh without lighting, and implements rotate/pan/zoom controls.
- `data/model.obj`: low-poly placeholder face mesh for the Week 1 demo.
- `scripts/detect_landmarks.py`: detects MediaPipe FaceMesh landmarks in one input image and writes CSV/debug-image outputs.
- `requirements.txt`: pins the Python preprocessing dependencies tested with this project.
- `inputs/`: local input images for landmark preprocessing. Contents are ignored by Git.
- `outputs/`: generated landmark CSV files and debug overlays. Contents are ignored by Git.
- `.gitignore`: keeps generated build files out of version control.

## Replacing the Placeholder Mesh

`data/model.obj` is only a low-poly placeholder. Once the Basel Face Model data is available, convert/export a neutral face mesh to OBJ and replace the file or pass the exported path explicitly:

```bash
./build/face_reconstruction path/to/neutral_face.obj
```

## Image Landmark Preprocessing

The reconstruction pipeline uses a Python preprocessing step for 2D face landmarks before
the C++ optimization code consumes them. This keeps MediaPipe's ML dependency separate
from the C++ geometry and fitting code.

### Setup

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preprocessing script uses MediaPipe's FaceMesh API from the Python package, so no
separate `.task` model download is required for this step.

Put a face image in `inputs/`, for example:

```text
inputs/face.jpg
```

### Run Landmark Detection

```bash
source .venv/bin/activate
python scripts/detect_landmarks.py \
  --image inputs/face.jpg \
  --csv outputs/face_landmarks.csv \
  --debug outputs/face_landmarks.png
```

Expected output:

```text
Detected 478 landmarks
Saved CSV: outputs/face_landmarks.csv
Saved debug image: outputs/face_landmarks.png
```

The CSV format is intentionally simple for later C++ loading:

```csv
index,x_px,y_px,z_norm,x_norm,y_norm
0,512.3,421.8,-0.038,0.500,0.412
```

Use `--max-points-to-label N` to label the first `N` landmark indices in the debug image
when checking landmark-to-model correspondence.

### Verify C++ Landmark Loading

After generating a CSV, load it with the C++ executable:

```bash
cmake --build build
./build/face_reconstruction --info --landmarks outputs/face_landmarks.csv
```

This prints the mesh summary plus the landmark count, index range, and pixel-coordinate
ranges. It is the handoff point between Python landmark detection and later C++ fitting.

## Week 1 Progress

The first project milestone is to set up a working C++ project and prepare the dependencies needed to load and display a face model without lighting.

The Week 1 goal was project setup plus face model loading and display without lighting. The current implementation completes that goal with a minimal but working pipeline:

1. CMake configures and builds the project.
2. The executable loads an OBJ mesh from disk.
3. The loader prints mesh statistics for quick verification.
4. The viewer displays the loaded mesh with unlit OpenGL rendering.
5. A placeholder face-like OBJ is included so the demo works before the real Basel Face Model export is available.
