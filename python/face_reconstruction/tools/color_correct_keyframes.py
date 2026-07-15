#!/usr/bin/env python3
"""Color-normalize registered keyframes to one reference photograph.

The transform is estimated from robust Lab statistics inside each MediaPipe
face hull, then feathered over the face/head region.  The untouched outer
background therefore remains stable while the projective face textures become
consistent enough to cross-fade without a color pulse.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Registered keyframe manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument(
        "--max-gain",
        type=float,
        default=1.15,
        help="Symmetric per-Lab-channel gain limit (default: 1.15)",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=12.0,
        help="Maximum absolute Lab offset in OpenCV units",
    )
    return parser.parse_args()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"frame", "image", "landmarks"} <= set(reader.fieldnames):
            raise ValueError("manifest must contain frame,image,landmarks columns")
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "image": resolve_from(path.parent, row["image"]),
                    "landmarks": resolve_from(path.parent, row["landmarks"]),
                }
            )
    if not rows:
        raise ValueError(f"manifest contains no keyframes: {path}")
    return rows


def load_face_hull(path: Path, width: int, height: int) -> np.ndarray:
    points = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            points.append((float(row["x_norm"]) * width, float(row["y_norm"]) * height))
    if len(points) < 3:
        raise ValueError(f"landmark file has fewer than three points: {path}")
    return cv2.convexHull(np.asarray(points, dtype=np.float32)).astype(np.int32)


def masks_from_hull(hull: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    hard = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(hard, hull, 255)
    x, y, face_width, face_height = cv2.boundingRect(hull)

    # Statistics use the interior, avoiding hair/background at the silhouette.
    erosion = max(1, round(0.035 * max(face_width, face_height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1,) * 2)
    statistics = cv2.erode(hard, kernel)

    # Correction extends slightly beyond the BFM face and fades to zero.
    dilation = max(2, round(0.10 * max(face_width, face_height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
    feather = cv2.dilate(hard, kernel).astype(np.float32) / 255.0
    sigma = max(1.0, 0.45 * dilation)
    feather = cv2.GaussianBlur(feather, (0, 0), sigma)
    return statistics, np.clip(feather, 0.0, 1.0)


def robust_lab_statistics(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask > 0]
    if len(pixels) < 100:
        raise ValueError("face mask contains too few pixels for color correction")
    low, high = np.percentile(pixels, [5.0, 95.0], axis=0)
    valid = np.all((pixels >= low) & (pixels <= high), axis=1)
    pixels = pixels[valid]
    center = np.median(pixels, axis=0)
    scale = 1.4826 * np.median(np.abs(pixels - center), axis=0)
    return center, np.maximum(scale, 1.0)


def correct_image(
    image: np.ndarray,
    alpha: np.ndarray,
    source_center: np.ndarray,
    source_scale: np.ndarray,
    reference_center: np.ndarray,
    reference_scale: np.ndarray,
    max_gain: float,
    max_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_gain = 1.0 / max_gain
    gain = np.clip(reference_scale / source_scale, minimum_gain, max_gain)
    offset = np.clip(reference_center - gain * source_center, -max_offset, max_offset)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    adjusted = np.clip(lab * gain + offset, 0.0, 255.0).astype(np.uint8)
    adjusted = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR).astype(np.float32)
    mixed = alpha[..., None] * adjusted + (1.0 - alpha[..., None]) * image
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8), gain, offset


def main() -> int:
    args = parse_args()
    if args.max_gain < 1.0:
        raise ValueError("--max-gain must be at least 1.0")
    if args.max_offset < 0.0:
        raise ValueError("--max-offset must be non-negative")

    manifest = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest)
    reference_rows = [row for row in rows if row["frame"] == args.reference_frame]
    if len(reference_rows) != 1:
        raise ValueError(f"reference frame {args.reference_frame} is not unique in manifest")

    loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in rows:
        image = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read keyframe image: {row['image']}")
        hull = load_face_hull(row["landmarks"], image.shape[1], image.shape[0])
        statistics_mask, alpha = masks_from_hull(hull, image.shape[:2])
        loaded[row["frame"]] = (image, statistics_mask, alpha)

    reference_image, reference_mask, _ = loaded[args.reference_frame]
    reference_center, reference_scale = robust_lab_statistics(reference_image, reference_mask)
    output = args.output.expanduser().resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "reference_frame": args.reference_frame,
        "reference_lab_center": reference_center.tolist(),
        "reference_lab_scale": reference_scale.tolist(),
        "frames": [],
    }
    corrected_rows = []
    for row in rows:
        image, mask, alpha = loaded[row["frame"]]
        center, scale = robust_lab_statistics(image, mask)
        if row["frame"] == args.reference_frame:
            corrected = image.copy()
            gain = np.ones(3, dtype=np.float32)
            offset = np.zeros(3, dtype=np.float32)
        else:
            corrected, gain, offset = correct_image(
                image,
                alpha,
                center,
                scale,
                reference_center,
                reference_scale,
                args.max_gain,
                args.max_offset,
            )
        target = image_dir / f"frame_{row['frame']:06d}.png"
        if not cv2.imwrite(str(target), corrected):
            raise RuntimeError(f"could not write corrected keyframe: {target}")
        corrected_rows.append((row["frame"], target, row["landmarks"]))
        report["frames"].append(
            {
                "frame": row["frame"],
                "source": str(row["image"]),
                "lab_center": center.tolist(),
                "lab_scale": scale.tolist(),
                "gain": gain.tolist(),
                "offset": offset.tolist(),
            }
        )
        print(f"[COLOR] frame {row['frame']}: gain={gain.round(4)} offset={offset.round(3)}")

    corrected_manifest = output / "manifest.csv"
    with corrected_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "image", "landmarks"])
        writer.writerows(corrected_rows)
    (output / "color_correction.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[SUCCESS] Corrected manifest written to {corrected_manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
