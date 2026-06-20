Implemented the missing sparse reconstruction stage.

## What is now implemented

The pipeline is now:

```text
Face image
→ MediaPipe landmarks
→ MediaPipe/BFM correspondence
→ camera estimation
→ BFM identity and expression fitting
→ personalized 3D mesh
→ OFF/PLY export
→ rendering
```

Main implementation:

- BFM fitting: [fitting.cc](</Users/dingpengsen/Desktop/IN2354 3DSMC/face-reconstruction/face_recon/fitting.cc:32>)
- OFF/PLY export: [mesh_export.cc](</Users/dingpengsen/Desktop/IN2354 3DSMC/face-reconstruction/face_recon/mesh_export.cc:25>)
- Command-line integration: [main.cc](</Users/dingpengsen/Desktop/IN2354 3DSMC/face-reconstruction/face_recon/main.cc:85>)
- Usage documentation: [README.md](</Users/dingpengsen/Desktop/IN2354 3DSMC/face-reconstruction/README.md:51>)
- Synthetic fitting test: [fitting_test.cpp](</Users/dingpengsen/Desktop/IN2354 3DSMC/face-reconstruction/tests/fitting_test.cpp:1>)

## How to run it

First place the licensed BFM file here:

```text
data/model2019_face12.h5
```

Detect landmarks:

```bash
python scripts/detect_landmarks.py \
  --image inputs/1.jpg \
  --csv outputs/1_landmarks.csv \
  --debug outputs/1_landmarks.png
```

Fit the 3D face:

```bash
./build/face_reconstruction \
  --model data/model2019_face12.h5 \
  --landmarks outputs/1_landmarks.csv \
  --correspondences data/bfm_mediapipe_correspondence.csv \
  --output results
```

This generates:

```text
results/1_landmarks_fitted.off
results/1_landmarks_fitted.ply
results/1_landmarks_fitting.txt
results/1_landmarks_reprojections.csv
```

Open the OFF file directly in MeshLab:

```text
File → Import Mesh → 1_landmarks_fitted.off
```

The PLY additionally contains the mean BFM albedo and can be rendered with:

```bash
./build/landmark_viewer results/1_landmarks_fitted.ply \
  --render-all outputs/fitted_rendering
```

## Mathematical implementation

### 1. Personalized BFM geometry

For each mesh vertex:

\[
V(\alpha,\delta)
=
\bar V_s+\bar V_e+B_s\alpha+B_e\delta
\]

where:

- \(\bar V_s\): mean identity shape
- \(\bar V_e\): mean expression
- \(B_s\): identity PCA basis
- \(\alpha\): identity coefficients
- \(B_e\): expression PCA basis
- \(\delta\): expression coefficients

The program initially sets all coefficients to zero, corresponding to the BFM mean face.

### 2. Camera model

A weak-perspective camera is used:

\[
Q_i=R(\omega)V_i
\]

\[
\hat{x}_i=sQ_{i,x}+t_x
\]

\[
\hat{y}_i=-sQ_{i,y}+t_y
\]

The optimized camera parameters are:

- angle-axis rotation \(\omega\)
- scale \(s\)
- image translation \(t_x,t_y\)

The scale is represented internally as \(s=\exp(k)\), ensuring it always remains positive.

### 3. Landmark loss

For each MediaPipe/BFM correspondence:

\[
E_{\text{landmark}}
=
\lambda_l
\sum_i
\rho\left(
\left\|
\hat l_i-l_i
\right\|^2
\right)
\]

where:

- \(l_i\): detected MediaPipe landmark
- \(\hat l_i\): projected BFM vertex
- \(\rho\): Huber robust loss

The Huber loss reduces the influence of inaccurate landmark correspondences.

### 4. PCA regularization

Landmarks alone do not contain enough information to determine a complete face. Therefore, statistically unlikely coefficients are penalized:

\[
E_{\text{shape}}
=
\lambda_s
\sum_j
\left(\frac{\alpha_j}{\sigma_{s,j}}\right)^2
\]

\[
E_{\text{expression}}
=
\lambda_e
\sum_j
\left(\frac{\delta_j}{\sigma_{e,j}}\right)^2
\]

The complete energy is:

\[
E =
E_{\text{landmark}}
+E_{\text{shape}}
+E_{\text{expression}}
\]

Coefficients are also constrained to:

\[
-3\sigma_j \leq c_j \leq 3\sigma_j
\]

This prevents severely distorted faces.

### 5. Two-stage optimization

Ceres runs two stages:

1. Camera fitting

   Shape and expression remain at the mean while rotation, scale and translation are optimized.

2. Joint fitting

   Camera, identity and expression are optimized together.

Levenberg-Marquardt and automatic differentiation are used.

## Verification

The project builds successfully, and the synthetic regression test passed:

```text
Initial normalized RMSE: 0.0108334
Final normalized RMSE:   1.73 × 10⁻⁸
```

This verifies that known camera and face parameters can be recovered from generated landmarks.

I could not generate a real BFM OFF file locally because `model2019_face12.h5` is not currently present in the workspace, Desktop, or Downloads.

## Current limitation

This is a sparse landmark reconstruction, not yet the complete dense project:

- Identity geometry: fitted
- Expression geometry: fitted
- Camera pose: fitted
- Celebrity skin texture/albedo: not fitted
- Illumination: not fitted
- Dense pixel comparison: not implemented

Therefore, the OFF mesh should approximate major facial geometry, but it will not be an exact scan or photorealistic celebrity model from one photograph.