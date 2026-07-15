# Pipeline Overview

## Single-Image Reconstruction

```text
input photo
  -> apps/reconstruct.py
  -> MediaPipe landmark detection
  -> MediaPipe face-skin segmentation
  -> build/face_reconstruction
  -> BFM landmark fitting
  -> landmark visibility and pose filtering
  -> silhouette geometry refinement
  -> CPU rasterization and visibility
  -> albedo + illumination fitting
  -> dense expression/pose refinement
  -> dense per-vertex RGB texture fitting
  -> mesh, render, and report export
```

The main C++ executable is `build/face_reconstruction`. It reads:

- `assets/models/model2019_face12.h5`
- `data/bfm_mediapipe_correspondence.csv`
- the input image
- the generated landmark CSV
- the generated face mask

## Sequence Tracking

```text
video or image sequence
  -> apps/track_sequence.py
  -> extracted frames
  -> per-frame MediaPipe landmarks
  -> first-frame reconstruction or supplied fitting.txt
  -> build/face_sequence_tracker
  -> tracking/tracking.csv
```

The tracker keeps identity fixed and estimates per-frame camera pose,
expression coefficients, region confidences, and observed/fitted eye opening.

## Expression Transfer

```text
target reconstruction + source tracking
  -> apps/transfer_expression.py
  -> build/face_expression_transfer
  -> output frames
  -> transfer.mp4 and optional comparison.mp4
```

The transfer stage can use projective texture sampling from the original
target photo, optional blink correction, optional source-relative head pose,
and optional multi-view keyframes.

