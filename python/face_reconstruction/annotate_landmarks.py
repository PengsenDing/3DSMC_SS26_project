"""Render a landmark-annotated diagnostic video for a tracked sequence.

Draws detected MediaPipe landmarks (green, filled) against the tracked
BFM fit's reprojections (red, hollow) on every frame, with magnified eye
insets and per-frame eyelid gap readouts to diagnose eye-motion tracking.

Usage:
    python -m face_reconstruction.annotate_landmarks outputs/sequences/talking_3
"""

import argparse
import csv
import math
import os
import subprocess
import sys

import cv2
import h5py
import numpy as np

NUM_EXPRESSION = 30

# MediaPipe indices used for the eyelid gap ratio: top, bottom, inner
# corner, outer corner.
LEFT_EYE = (159, 145, 133, 33)
RIGHT_EYE = (386, 374, 362, 263)


def load_correspondence(path):
    names, vertex_ids, mp_ids = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row["bfm_landmark_name"])
            vertex_ids.append(int(row["bfm_vertex_id"]))
            mp_ids.append(int(row["mediapipe_index"]))
    return names, np.array(vertex_ids), np.array(mp_ids)


def load_model_rows(model_path, vertex_ids):
    """Load mean + scaled bases restricted to the correspondence vertices."""
    rows = np.stack([3 * vertex_ids, 3 * vertex_ids + 1, 3 * vertex_ids + 2],
                    axis=1).reshape(-1)
    with h5py.File(model_path, "r") as f:
        shape_mean = f["shape/model/mean"][:][rows]
        expr_mean = f["expression/model/mean"][:][rows]
        shape_basis = f["shape/model/pcaBasis"][:][rows, :]
        shape_var = f["shape/model/pcaVariance"][:]
        expr_basis = f["expression/model/pcaBasis"][:][rows, :NUM_EXPRESSION]
        expr_var = f["expression/model/pcaVariance"][:NUM_EXPRESSION]
    return (shape_mean + expr_mean,
            shape_basis * np.sqrt(shape_var)[None, :],
            expr_basis * np.sqrt(expr_var)[None, :])


def load_shape_coefficients(fitting_txt):
    coeffs = {}
    with open(fitting_txt) as f:
        for line in f:
            if line.startswith("shape["):
                idx = int(line[len("shape["):line.index("]")])
                coeffs[idx] = float(line.split(":")[1])
    return np.array([coeffs[i] for i in range(len(coeffs))])


def project(points, angle_axis, tx, ty, tz, focal, aspect):
    rot, _ = cv2.Rodrigues(np.asarray(angle_axis))
    rotated = points @ rot.T
    depth = tz - rotated[:, 2]
    u = 0.5 + focal * (rotated[:, 0] + tx) / depth
    v = 0.5 - focal * aspect * (rotated[:, 1] + ty) / depth
    return np.stack([u, v], axis=1)


def read_detected(landmark_csv):
    pts = {}
    with open(landmark_csv) as f:
        for row in csv.DictReader(f):
            pts[int(row["index"])] = (float(row["u"]), float(row["v"]))
    return pts


def gap_ratio(pts):
    ratios = []
    for top, bot, ci, co in (LEFT_EYE, RIGHT_EYE):
        if all(k in pts for k in (top, bot, ci, co)):
            gap = math.dist(pts[top], pts[bot])
            width = math.dist(pts[ci], pts[co])
            if width > 1e-6:
                ratios.append(gap / width)
    return sum(ratios) / len(ratios) if ratios else float("nan")


def brighten(image, gamma=1.8):
    lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)
    return cv2.LUT(image, lut)


def eye_inset(image, detected, projected_px, mp_ids, eye, scale=4, margin=1.6):
    top, bot, ci, co = eye
    corners = np.array([detected[ci], detected[co]])
    center = corners.mean(axis=0)
    half_w = max(abs(corners[0][0] - corners[1][0]) * margin / 2, 24)
    half_h = half_w * 0.62
    x0, x1 = int(center[0] - half_w), int(center[0] + half_w)
    y0, y1 = int(center[1] - half_h), int(center[1] + half_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)

    def to_inset(p):
        return int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale))

    for slot, mp_idx in enumerate(mp_ids):
        if mp_idx not in (top, bot, ci, co):
            continue
        if mp_idx in detected:
            cv2.circle(crop, to_inset(detected[mp_idx]), 5, (0, 220, 0), -1,
                       cv2.LINE_AA)
        cv2.circle(crop, to_inset(projected_px[slot]), 5, (0, 0, 255), 2,
                   cv2.LINE_AA)
    return crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="Tracked sequence directory")
    parser.add_argument("--model", default=None)
    parser.add_argument("--correspondences", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    seq = args.sequence.rstrip("/")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = args.model or os.path.join(
        repo, "assets/models/model2019_face12.h5")
    corr_path = args.correspondences or os.path.join(
        repo, "data/bfm_mediapipe_correspondence.csv")
    output = args.output or os.path.join(seq, "landmark_annotated.mp4")

    names, vertex_ids, mp_ids = load_correspondence(corr_path)
    mean, shape_basis, expr_basis = load_model_rows(model_path, vertex_ids)
    shape_coeffs = load_shape_coefficients(
        os.path.join(seq, "initial_reconstruction/fitting.txt"))
    identity = mean + shape_basis @ shape_coeffs

    with open(os.path.join(seq, "tracking/tracking.csv")) as f:
        track = list(csv.DictReader(f))

    first = cv2.imread(os.path.join(seq, "frames/frame_000000.png"))
    height, width = first.shape[:2]
    aspect = width / height

    panel_w = 460
    canvas_w, canvas_h = width + panel_w, height
    fps = args.fps or 24.0

    tmp_dir = os.path.join(seq, "annotated_frames_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    detected_curve, fitted_curve = [], []
    for row in track:
        idx = int(row["frame"])
        frame_path = os.path.join(seq, f"frames/frame_{idx:06d}.png")
        image = cv2.imread(frame_path)
        if image is None:
            continue
        image = brighten(image)

        expr = np.array(
            [float(row[f"expression_{i}"]) for i in range(NUM_EXPRESSION)])
        points = (identity + expr_basis @ expr).reshape(-1, 3)
        angle_axis = [float(row["angle_axis_x"]), float(row["angle_axis_y"]),
                      float(row["angle_axis_z"])]
        proj = project(points, angle_axis, float(row["translation_x"]),
                       float(row["translation_y"]), float(row["translation_z"]),
                       float(row["focal_length"]), aspect)
        proj_px = proj * np.array([width, height])

        detected = read_detected(
            os.path.join(seq, f"landmarks/frame_{idx:06d}.csv"))

        det_ratio = gap_ratio(detected)
        fit_pts = {mp: tuple(proj_px[slot]) for slot, mp in enumerate(mp_ids)}
        fit_ratio = gap_ratio(fit_pts)
        detected_curve.append(det_ratio)
        fitted_curve.append(fit_ratio)

        for mp_idx, pt in detected.items():
            cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (140, 140, 140), -1)
        for slot, mp_idx in enumerate(mp_ids):
            p = (int(round(proj_px[slot][0])), int(round(proj_px[slot][1])))
            if mp_idx in detected:
                d = (int(detected[mp_idx][0]), int(detected[mp_idx][1]))
                cv2.line(image, d, p, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.circle(image, d, 3, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(image, p, 3, (0, 0, 255), 1, cv2.LINE_AA)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :width] = image
        insets = [
            eye_inset(image, detected, proj_px, mp_ids, RIGHT_EYE),
            eye_inset(image, detected, proj_px, mp_ids, LEFT_EYE),
        ]
        y = 10
        for label, inset in zip(("R eye", "L eye"), insets):
            h, w = inset.shape[:2]
            if w > panel_w - 20:
                s = (panel_w - 20) / w
                inset = cv2.resize(inset, None, fx=s, fy=s)
                h, w = inset.shape[:2]
            canvas[y:y + h, width + 10:width + 10 + w] = inset
            cv2.putText(canvas, label, (width + 14, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += h + 14

        for i, (label, value, color) in enumerate([
                ("frame", f"{idx}", (255, 255, 255)),
                ("detected gap", f"{det_ratio:.3f}", (0, 220, 0)),
                ("fitted gap", f"{fit_ratio:.3f}", (0, 0, 255))]):
            cv2.putText(canvas, f"{label}: {value}",
                        (width + 14, y + 26 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx:06d}.png"), canvas)

    det = np.array(detected_curve)
    fit = np.array(fitted_curve)
    print(f"detected eyelid gap ratio: min {np.nanmin(det):.3f} "
          f"max {np.nanmax(det):.3f} range {np.nanmax(det) - np.nanmin(det):.3f}")
    print(f"fitted   eyelid gap ratio: min {np.nanmin(fit):.3f} "
          f"max {np.nanmax(fit):.3f} range {np.nanmax(fit) - np.nanmin(fit):.3f}")

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp_dir, "frame_%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", output],
        check=True)
    print(f"annotated video: {output}")


if __name__ == "__main__":
    main()
