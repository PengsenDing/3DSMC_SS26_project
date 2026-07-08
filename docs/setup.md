# Setup Notes

## Required Local Assets

The repository does not track large or licensed assets. Place them under
`assets/models/`:

```text
assets/models/model2019_face12.h5
assets/models/selfie_multiclass_256x256.tflite
```

The BFM file must be downloaded manually from the Basel Face Model website.
The MediaPipe segmenter can be downloaded with:

```bash
curl -L https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite \
  -o assets/models/selfie_multiclass_256x256.tflite
```

## Build

The C++ side requires Eigen, Ceres, glog, HDF5, and OpenCV. CMake fetches
CLI11, HighFive, and nlohmann-json.

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build
```

## Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional ArcFace evaluation uses a separate environment:

```bash
python3.11 -m venv .venv-arcface
.venv-arcface/bin/python -m pip install -r requirements-evaluation.txt
```

