# Face Reconstruction

Classical optimization based face reconstruction project for the IN2354 3D Scanning and Motion Capture course.

The first project milestone is to set up a working C++ project and prepare the dependencies needed to load and display a face model without lighting.

## Current Status

Week 1 setup is in place:

- CMake based C++20 project skeleton.
- `face_reconstruction_core` library target for reusable project code.
- `face_reconstruction` executable target for demos and smoke tests.
- Dependencies discovered and linked through CMake.
- OBJ mesh loading with triangulation for polygon faces.
- Unlit OpenGL/SFML mesh viewer with basic camera controls.
- A small sample face-like OBJ mesh in `data/model.obj`.

## Week 1 Progress

The Week 1 goal was project setup plus face model loading and display without lighting. The current implementation completes that goal with a minimal but working pipeline:

1. CMake configures and builds the project.
2. The executable loads an OBJ mesh from disk.
3. The loader prints mesh statistics for quick verification.
4. The viewer displays the loaded mesh with unlit OpenGL rendering.
5. A placeholder face-like OBJ is included so the demo works before the real Basel Face Model export is available.

## Dependencies

The project currently links the dependencies needed for the planned reconstruction pipeline:

- Eigen 3.4.0: matrix/vector math and linear algebra.
- OpenCV 4.11.0: image loading, image processing, and later RGB/RGB-D input handling.
- Ceres Solver 2.2.0: nonlinear least squares optimization for later alignment and model fitting.
- OpenGL: rendering backend for the week-1 face model viewer.
- GLEW 2.2.0 package: OpenGL extension loading. The installed headers report 2.3.4.
- SFML 3.0.2: window creation and OpenGL context management.

GLFW is not currently installed on this machine, so SFML is used as the window/context layer.

## Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

## Run Dependency Smoke Test

```bash
./build/face_reconstruction --deps
```

Expected output:

```text
Face Reconstruction project skeleton
C++ target: C++20
Eigen: 3.4.0 (identity trace = 3)
OpenCV: 4.11.0
Ceres Solver: 2.2.0
GLEW headers: 2.3.4
SFML: 3.0.2
OpenGL: linked through CMake OpenGL::GL
```

The executable may take a few seconds on first run while dynamic libraries are loaded.

## Load Mesh Without Opening a Window

```bash
./build/face_reconstruction data/model.obj --info
```

This verifies the OBJ loader and prints mesh statistics.

## Open the Week 1 Viewer

```bash
./build/face_reconstruction data/model.obj
```

Viewer controls:

- Left drag: rotate.
- Right or middle drag: pan.
- Mouse wheel: zoom.
- `R`: reset camera.
- `Esc`: close.

For an automated viewer smoke test that opens the viewer, renders one frame, and exits:

```bash
./build/face_reconstruction data/model.obj --frames 1
```

## Project Layout

```text
.
├── CMakeLists.txt
├── data/
│   └── model.obj
├── include/
│   └── face_reconstruction/
│       ├── app.hpp
│       ├── mesh.hpp
│       ├── obj_loader.hpp
│       └── viewer.hpp
├── src/
│   ├── app.cpp
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
- `include/face_reconstruction/mesh.hpp`: defines the `Mesh` and `Triangle` data structures.
- `src/mesh.cpp`: implements mesh helper functions such as center, extent, bounding radius, and summary output.
- `include/face_reconstruction/obj_loader.hpp`: declares the OBJ loading function.
- `src/obj_loader.cpp`: reads OBJ files, parses vertices/faces, supports common OBJ face formats, and triangulates polygon faces.
- `include/face_reconstruction/viewer.hpp`: declares viewer options and the viewer entry point.
- `src/viewer.cpp`: creates the SFML/OpenGL window, renders the mesh without lighting, and implements rotate/pan/zoom controls.
- `data/model.obj`: low-poly placeholder face mesh for the Week 1 demo.
- `.gitignore`: keeps generated build files out of version control.

## Replacing the Placeholder Mesh

`data/model.obj` is only a low-poly placeholder. Once the Basel Face Model data is available, convert/export a neutral face mesh to OBJ and replace the file or pass the exported path explicitly:

```bash
./build/face_reconstruction path/to/neutral_face.obj
```
