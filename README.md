# Face Reconstruction

Classical optimization-based 3D face reconstruction and one-shot facial
reenactment. The project fits the Basel Face Model 2019 to a photograph using
MediaPipe landmarks, Ceres, OpenCV, and a custom CPU rasterizer.

## Repository Layout

```text
apps/                         Python command-line entry points
cpp/apps/                     C++ executable entry points
cpp/src/                      C++ reconstruction library sources
cpp/include/                  Public C++ headers
python/face_reconstruction/   Python preprocessing and utility package
data/                         Small tracked metadata files
assets/inputs/                Local input photos/videos, ignored by Git
assets/models/                Downloaded model files, ignored by Git
outputs/                      Generated reconstructions, tracking, transfers
tests/cpp/                    C++ regression tests
docs/                         Setup and pipeline notes
```

## Setup

1. Download `model2019_face12.h5` from the
   [BFM 2019 website](https://faces.dmi.unibas.ch/bfm/bfm2019.html) and place
   it at:

   ```bash
   assets/models/model2019_face12.h5
   ```

2. Download the MediaPipe face segmentation model:

   ```bash
   curl -L https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite \
     -o assets/models/selfie_multiclass_256x256.tflite
   ```

3. Build the C++ executables:

   ```bash
   cmake -S . -B build
   cmake --build build --parallel
   ```

4. Create the Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

## Run

Single-image reconstruction:

```bash
python apps/reconstruct.py assets/inputs/<image-name>
```

Output is written to `outputs/reconstructions/<image-name>/`.

Track a source video:

```bash
python apps/track_sequence.py assets/inputs/<video-name> \
  --output outputs/sequences/<video-name>
```

Transfer expression from a tracked sequence to a reconstructed target:

```bash
python apps/transfer_expression.py \
  outputs/reconstructions/target \
  outputs/sequences/<video-name> \
  --output outputs/transfers/video_to_target
```

## Outputs

Generated files are intentionally ignored by Git. A reconstruction directory
contains the mesh, renderings, reports, and metadata:

```text
outputs/reconstructions/face/
├── face.off
├── face.ply
├── fitting.txt
├── silhouette_fitting.txt
├── bfm_surface_overlay.png
├── rendered_final.png
├── rendered_final_overlay.png
└── run.json
```

## More Documentation

- [Setup notes](docs/setup.md)
- [Pipeline overview](docs/pipeline.md)

