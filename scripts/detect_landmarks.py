#!/usr/bin/env python3
"""Detect MediaPipe face landmarks in a still image.

This script is intentionally a preprocessing tool: it converts an input image
into a simple CSV that the C++ reconstruction pipeline can load later.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect MediaPipe face landmarks and export them as CSV."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Input face image path.",
    )
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--csv",
        default=Path("outputs/face_landmarks.csv"),
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--debug",
        default=Path("outputs/face_landmarks.png"),
        type=Path,
        help="Output debug image path with landmarks drawn on top.",
    )
    parser.add_argument(
        "--max-points-to-label",
        default=0,
        type=int,
        help="Optionally label the first N landmark indices in the debug image.",
    )
    return parser.parse_args()


def require_input_files(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")


def detect_landmarks(image_rgb: Any, mp_module: Any):
    with mp_module.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    return result.multi_face_landmarks[0].landmark


def draw_landmarks(
    image_bgr: Any,
    landmarks,
    width: int,
    height: int,
    max_points_to_label: int,
    cv2_module: Any,
) -> Any:
    debug_image = image_bgr.copy()

    for index, landmark in enumerate(landmarks):
        x_px = int(round(landmark.x * width))
        y_px = int(round(landmark.y * height))
        cv2_module.circle(debug_image, (x_px, y_px), 1, (0, 255, 0), -1)

        if index < max_points_to_label:
            cv2_module.putText(
                debug_image,
                str(index),
                (x_px + 2, y_px - 2),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 180, 255),
                1,
                cv2_module.LINE_AA,
            )

    return debug_image


def write_landmarks_csv(csv_path: Path, landmarks, width: int, height: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "x_px", "y_px", "z_norm", "x_norm", "y_norm"])

        for index, landmark in enumerate(landmarks):
            writer.writerow(
                [
                    index,
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z,
                    landmark.x,
                    landmark.y,
                ]
            )


def main() -> int:
    args = parse_args()
    require_input_files(args.image)

    import cv2
    import mediapipe as mp

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {args.image}")

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_landmarks(image_rgb, mp)

    if not landmarks:
        raise RuntimeError(f"No face landmarks detected in image: {args.image}")

    write_landmarks_csv(args.csv, landmarks, width, height)

    args.debug.parent.mkdir(parents=True, exist_ok=True)
    debug_image = draw_landmarks(
        image_bgr,
        landmarks,
        width,
        height,
        max(0, args.max_points_to_label),
        cv2,
    )
    if not cv2.imwrite(str(args.debug), debug_image):
        raise RuntimeError(f"OpenCV could not write debug image: {args.debug}")

    print(f"Detected {len(landmarks)} landmarks")
    print(f"Saved CSV: {args.csv}")
    print(f"Saved debug image: {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
