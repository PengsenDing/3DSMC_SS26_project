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

DEFAULT_BFM_CORRESPONDENCES = PROJECT_ROOT / "data" / "bfm_mediapipe_correspondence.csv"
DEFAULT_DEBUG_DIRECTORY = PROJECT_ROOT / "reconstructions" / "_landmark_debug"

# Ordered MediaPipe FACEMESH_FACE_OVAL vertices. They are rasterized into one
# silhouette observation; they are never exported as sparse BFM constraints.
FACE_OVAL_INDICES = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)


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
        default=DEFAULT_DEBUG_DIRECTORY / "landmarks.csv",
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        help="Optional plain-text landmark output path.",
    )
    parser.add_argument(
        "--bfm-csv",
        type=Path,
        help="Optional output CSV with BFM vertices matched to MediaPipe landmarks.",
    )
    parser.add_argument(
        "--bfm-correspondences",
        default=DEFAULT_BFM_CORRESPONDENCES,
        type=Path,
        help="CSV file describing BFM landmark to MediaPipe index correspondences.",
    )
    parser.add_argument(
        "--debug",
        default=DEFAULT_DEBUG_DIRECTORY / "landmarks.png",
        type=Path,
        help="Output debug image path with landmarks drawn on top.",
    )
    parser.add_argument(
        "--silhouette-mask",
        type=Path,
        help="Optional binary face-oval silhouette mask output path.",
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
        u = int(round(landmark.x * width))
        v = int(round(landmark.y * height))
        cv2_module.circle(debug_image, (u, v), 1, (0, 255, 0), -1)

        if index < max_points_to_label:
            cv2_module.putText(
                debug_image,
                str(index),
                (u + 2, v - 2),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 180, 255),
                1,
                cv2_module.LINE_AA,
            )

    return debug_image


def write_silhouette_mask(
    output_path: Path,
    landmarks,
    width: int,
    height: int,
    cv2_module: Any,
) -> None:
    """Rasterize Face Mesh's ordered oval into a dense binary observation."""
    import numpy as np

    points = np.asarray(
        [
            [
                round(landmarks[index].x * (width - 1)),
                round(landmarks[index].y * (height - 1)),
            ]
            for index in FACE_OVAL_INDICES
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2_module.fillPoly(mask, [points], 255, lineType=cv2_module.LINE_8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(output_path), mask):
        raise RuntimeError(f"OpenCV could not write silhouette mask: {output_path}")


def write_landmarks_csv(
    csv_path: Path,
    landmarks,
    width: int,
    height: int,
    landmark_names: dict[int, str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "name", "u", "v", "z_norm", "x_norm", "y_norm"])

        for index, landmark in enumerate(landmarks):
            writer.writerow(
                [
                    index,
                    landmark_names.get(index, ""),
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z,
                    landmark.x,
                    landmark.y,
                ]
            )


def write_landmarks_txt(
    txt_path: Path,
    landmarks,
    width: int,
    height: int,
    landmark_names: dict[int, str],
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with txt_path.open("w", encoding="utf-8") as file:
        file.write("# index\tname\tu\tv\tz_norm\tx_norm\ty_norm\n")

        for index, landmark in enumerate(landmarks):
            file.write(
                f"{index}\t"
                f"{landmark_names.get(index, '')}\t"
                f"{landmark.x * width:.6f}\t"
                f"{landmark.y * height:.6f}\t"
                f"{landmark.z:.6f}\t"
                f"{landmark.x:.6f}\t"
                f"{landmark.y:.6f}\n"
            )


def load_bfm_correspondences(correspondence_path: Path) -> list[dict[str, str]]:
    if not correspondence_path.is_file():
        raise FileNotFoundError(
            f"BFM correspondence CSV does not exist: {correspondence_path}"
        )

    with correspondence_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_fields = {
            "bfm_landmark_name",
            "bfm_vertex_id",
            "mediapipe_index",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(f"BFM correspondence CSV is missing columns: {missing}")

        return list(reader)


def build_landmark_name_map(
    correspondences: list[dict[str, str]],
) -> dict[int, str]:
    landmark_names: dict[int, str] = {}

    for row in correspondences:
        mediapipe_index = int(row["mediapipe_index"])
        name = row["bfm_landmark_name"]
        existing_name = landmark_names.get(mediapipe_index)
        if existing_name is not None and existing_name != name:
            raise RuntimeError(
                "MediaPipe index "
                f"{mediapipe_index} maps to both {existing_name!r} and {name!r}"
            )
        landmark_names[mediapipe_index] = name

    return landmark_names


def write_bfm_landmarks_csv(
    output_path: Path,
    correspondences: list[dict[str, str]],
    landmarks,
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "bfm_landmark_name",
                "bfm_vertex_id",
                "mediapipe_index",
                "mediapipe_name",
                "u",
                "v",
                "z_norm",
                "x_norm",
                "y_norm",
            ]
        )

        for row in correspondences:
            mediapipe_index = int(row["mediapipe_index"])
            landmark = (
                landmarks[mediapipe_index]
                if 0 <= mediapipe_index < len(landmarks)
                else None
            )
            coordinate_values = (
                [
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z,
                    landmark.x,
                    landmark.y,
                ]
                if landmark
                else ["", "", "", "", ""]
            )

            writer.writerow(
                [
                    row["bfm_landmark_name"],
                    row["bfm_vertex_id"],
                    mediapipe_index,
                    row["bfm_landmark_name"],
                    *coordinate_values,
                ]
            )


def run_detection(
    image_path: Path,
    csv_path: Path,
    debug_path: Path | None,
    correspondence_path: Path = DEFAULT_BFM_CORRESPONDENCES,
    txt_path: Path | None = None,
    bfm_csv_path: Path | None = None,
    silhouette_mask_path: Path | None = None,
    max_points_to_label: int = 0,
) -> int:
    """Run MediaPipe once and write the requested landmark artifacts."""
    require_input_files(image_path)

    import cv2
    import mediapipe as mp

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    landmarks = detect_landmarks(image_rgb, mp)
    if not landmarks:
        raise RuntimeError(f"No face landmarks detected in image: {image_path}")

    correspondences = load_bfm_correspondences(correspondence_path)
    landmark_names = build_landmark_name_map(correspondences)
    write_landmarks_csv(csv_path, landmarks, width, height, landmark_names)
    if txt_path:
        write_landmarks_txt(txt_path, landmarks, width, height, landmark_names)
    if bfm_csv_path:
        write_bfm_landmarks_csv(bfm_csv_path, correspondences, landmarks, width, height)
    if silhouette_mask_path:
        write_silhouette_mask(silhouette_mask_path, landmarks, width, height, cv2)

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_image = draw_landmarks(
            image_bgr,
            landmarks,
            width,
            height,
            max(0, max_points_to_label),
            cv2,
        )
        if not cv2.imwrite(str(debug_path), debug_image):
            raise RuntimeError(f"OpenCV could not write debug image: {debug_path}")

    return len(landmarks)


def main() -> int:
    args = parse_args()
    landmark_count = run_detection(
        image_path=args.image,
        csv_path=args.csv,
        debug_path=args.debug,
        correspondence_path=args.bfm_correspondences,
        txt_path=args.txt,
        bfm_csv_path=args.bfm_csv,
        silhouette_mask_path=args.silhouette_mask,
        max_points_to_label=args.max_points_to_label,
    )

    print(f"Detected {landmark_count} landmarks")
    print(f"Saved CSV: {args.csv}")
    if args.txt:
        print(f"Saved TXT: {args.txt}")
    if args.bfm_csv:
        print(f"Saved BFM CSV: {args.bfm_csv}")
    if args.silhouette_mask:
        print(f"Saved silhouette mask: {args.silhouette_mask}")
    print(f"Saved debug image: {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
