#!/usr/bin/env python3
"""Render BFM semantic landmarks over expression-transfer frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transfer", type=Path, help="Transfer output directory")
    parser.add_argument("--target-fitting", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--correspondences", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--labels", action="store_true", help="Draw landmark names")
    return parser.parse_args()


def load_fitting(path: Path) -> tuple[np.ndarray, float, float]:
    shape: dict[int, float] = {}
    focal = None
    aspect = None
    with path.open(encoding="utf-8") as file:
        for line in file:
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            if key.startswith("shape["):
                shape[int(key[6:key.index("]")])] = float(raw)
            elif key == "camera_focal_length_normalized":
                focal = float(raw)
            elif key == "camera_aspect_ratio":
                aspect = float(raw)
    if not shape or focal is None or aspect is None:
        raise ValueError(f"incomplete fitting report: {path}")
    coefficients = np.asarray([shape[index] for index in range(len(shape))])
    return coefficients, focal, aspect


def load_landmark_model(
    model_path: Path, correspondence_path: Path, shape_coefficients: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = []
    vertex_ids = []
    with correspondence_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            names.append(row["bfm_landmark_name"])
            vertex_ids.append(int(row["bfm_vertex_id"]))
    vertices = np.asarray(vertex_ids, dtype=np.int64)
    rows = np.stack((3 * vertices, 3 * vertices + 1, 3 * vertices + 2), axis=1).reshape(-1)
    with h5py.File(model_path, "r") as model:
        shape_mean = model["shape/model/mean"][:][rows]
        expression_mean = model["expression/model/mean"][:][rows]
        shape_basis = model["shape/model/pcaBasis"][:][rows, : len(shape_coefficients)]
        shape_variance = model["shape/model/pcaVariance"][: len(shape_coefficients)]
        expression_basis = model["expression/model/pcaBasis"][:][rows, :]
        expression_variance = model["expression/model/pcaVariance"][:]
    identity = shape_mean + expression_mean
    identity += (shape_basis * np.sqrt(shape_variance)[None, :]) @ shape_coefficients
    scaled_expression_basis = expression_basis * np.sqrt(expression_variance)[None, :]
    return names, identity, scaled_expression_basis


def project(
    points: np.ndarray, row: dict[str, str], focal: float, aspect: float,
) -> np.ndarray:
    angle_axis = np.asarray(
        [float(row["angle_axis_x"]), float(row["angle_axis_y"]), float(row["angle_axis_z"])]
    )
    rotation, _ = cv2.Rodrigues(angle_axis)
    rotated = points @ rotation.T
    tx = float(row["translation_x"])
    ty = float(row["translation_y"])
    tz = float(row["translation_z"])
    depth = tz - rotated[:, 2]
    u = 0.5 + focal * (rotated[:, 0] + tx) / depth
    v = 0.5 - focal * aspect * (rotated[:, 1] + ty) / depth
    return np.stack((u, v), axis=1)


def main() -> int:
    args = parse_args()
    transfer = args.transfer.expanduser().resolve()
    metadata_path = transfer / "transfer.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    root = Path(__file__).resolve().parents[3]
    target_fitting = (
        args.target_fitting.expanduser().resolve()
        if args.target_fitting
        else Path(metadata["target_reconstruction"]) / "fitting.txt"
    )
    frames_dir = args.frames_dir.expanduser().resolve() if args.frames_dir else transfer / "frames"
    model = (
        args.model.expanduser().resolve()
        if args.model
        else root / "assets" / "models" / "model2019_face12.h5"
    )
    correspondences = (
        args.correspondences.expanduser().resolve()
        if args.correspondences
        else root / "data/bfm_mediapipe_correspondence.csv"
    )
    output = args.output.expanduser().resolve() if args.output else transfer / "landmarks.mp4"
    fps = args.fps or float(metadata.get("fps", 24.0))

    for path in (target_fitting, model, correspondences, transfer / "transfer.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")

    with (transfer / "transfer.csv").open(newline="", encoding="utf-8") as file:
        trajectory = list(csv.DictReader(file))
    if not trajectory:
        raise ValueError("transfer trajectory is empty")
    expression_columns = sorted(
        (name for name in trajectory[0] if name.startswith("expression_")),
        key=lambda name: int(name.split("_")[1]),
    )
    shape, focal, aspect = load_fitting(target_fitting)
    names, identity, expression_basis = load_landmark_model(
        model, correspondences, shape
    )

    first_path = frames_dir / f"frame_{int(trajectory[0]['frame']):06d}.png"
    first = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"could not read transfer frame: {first_path}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create landmark video: {output}")

    used_expression_count = min(len(expression_columns), expression_basis.shape[1])
    for position, row in enumerate(trajectory, start=1):
        frame_index = int(row["frame"])
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            writer.release()
            raise FileNotFoundError(f"could not read transfer frame: {frame_path}")
        expression = np.asarray(
            [float(row[name]) for name in expression_columns[:used_expression_count]]
        )
        points = identity + expression_basis[:, :used_expression_count] @ expression
        projected = project(points.reshape(-1, 3), row, focal, aspect)
        projected *= np.asarray([width, height])
        radius = max(2, round(0.0025 * max(width, height)))
        for name, point in zip(names, projected):
            x, y = int(round(point[0])), int(round(point[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            cv2.circle(image, (x, y), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius, (0, 255, 80), -1, cv2.LINE_AA)
            if args.labels:
                cv2.putText(
                    image, name, (x + radius + 2, y - radius - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    image, name, (x + radius + 2, y - radius - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 80), 1, cv2.LINE_AA,
                )
        cv2.putText(
            image, f"BFM semantic landmarks | frame {frame_index}", (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, f"BFM semantic landmarks | frame {frame_index}", (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 1, cv2.LINE_AA,
        )
        writer.write(image)
        print(f"[LANDMARK VIDEO] {position}/{len(trajectory)}", flush=True)
    writer.release()
    print(f"[SUCCESS] Landmark video written to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
