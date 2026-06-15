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

LANDMARK_NAMES = {
    1: "nose_tip",
    0: "upper_lip_outer_center",
    2: "nose_base_center",
    4: "nose_bottom",
    10: "forehead_center",
    33: "right_eye_outer_corner",
    61: "right_mouth_corner",
    133: "right_eye_inner_corner",
    145: "right_eye_lower_lid",
    152: "chin",
    159: "right_eye_upper_lid",
    234: "right_cheek_outer",
    263: "left_eye_outer_corner",
    267: "left_upper_lip_philtrum_ridge",
    282: "left_eyebrow_lower_bend",
    285: "left_eyebrow_inner_lower",
    291: "left_mouth_corner",
    294: "left_nose_wing_tip",
    295: "left_eyebrow_upper_bend",
    326: "left_nostril_center",
    327: "left_nose_wing_outer",
    336: "left_eyebrow_inner_upper",
    362: "left_eye_inner_corner",
    374: "left_eye_lower_lid",
    386: "left_eye_upper_lid",
    429: "left_nasolabial_fold_center",
    432: "left_nasolabial_fold_bottom",
    454: "left_cheek_outer",
    468: "right_iris_center",
    473: "left_iris_center",
}

DEFAULT_BFM_CORRESPONDENCES = PROJECT_ROOT / "data" / "bfm_mediapipe_correspondence.csv"


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


def write_landmarks_csv(csv_path: Path, landmarks, width: int, height: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "name", "u", "v", "z_norm", "x_norm", "y_norm"])

        for index, landmark in enumerate(landmarks):
            writer.writerow(
                [
                    index,
                    LANDMARK_NAMES.get(index, ""),
                    landmark.x * width,
                    landmark.y * height,
                    landmark.z,
                    landmark.x,
                    landmark.y,
                ]
            )


def write_landmarks_txt(txt_path: Path, landmarks, width: int, height: int) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with txt_path.open("w", encoding="utf-8") as file:
        file.write("# index\tname\tu\tv\tz_norm\tx_norm\ty_norm\n")

        for index, landmark in enumerate(landmarks):
            file.write(
                f"{index}\t"
                f"{LANDMARK_NAMES.get(index, '')}\t"
                f"{landmark.x * width:.6f}\t"
                f"{landmark.y * height:.6f}\t"
                f"{landmark.z:.6f}\t"
                f"{landmark.x:.6f}\t"
                f"{landmark.y:.6f}\n"
            )


def load_bfm_correspondences(correspondence_path: Path) -> list[dict[str, str]]:
    if not correspondence_path.is_file():
        raise FileNotFoundError(f"BFM correspondence CSV does not exist: {correspondence_path}")

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
            landmark = landmarks[mediapipe_index] if 0 <= mediapipe_index < len(landmarks) else None
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
                    LANDMARK_NAMES.get(mediapipe_index, ""),
                    *coordinate_values,
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
    if args.txt:
        write_landmarks_txt(args.txt, landmarks, width, height)
    if args.bfm_csv:
        correspondences = load_bfm_correspondences(args.bfm_correspondences)
        write_bfm_landmarks_csv(args.bfm_csv, correspondences, landmarks, width, height)

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
    if args.txt:
        print(f"Saved TXT: {args.txt}")
    if args.bfm_csv:
        print(f"Saved BFM CSV: {args.bfm_csv}")
    print(f"Saved debug image: {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
