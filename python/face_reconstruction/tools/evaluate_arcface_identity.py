#!/usr/bin/env python3
"""Measure ArcFace identity consistency across generated keyframes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

# InsightFace imports optional plotting/augmentation modules. Keep their cache
# and version check deterministic and offline; neither is needed for inference.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "arcface-matplotlib-cache")
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "arcface-cache"))

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Keyframe manifest.csv")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--model-root", type=Path, default=Path("models/insightface"))
    parser.add_argument("--det-size", type=int, default=640)
    return parser.parse_args()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"frame", "image"} <= set(reader.fieldnames):
            raise ValueError("manifest must contain frame,image columns")
        rows = [
            {"frame": int(row["frame"]), "image": resolve_from(path.parent, row["image"])}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"manifest contains no keyframes: {path}")
    return rows


def largest_face_embedding(analyzer, path: Path) -> tuple[np.ndarray, float, list[float]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    faces = analyzer.get(image)
    if not faces:
        raise RuntimeError(f"ArcFace detected no face: {path}")
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    return embedding, float(face.det_score), [float(value) for value in face.bbox]


def main() -> int:
    args = parse_args()
    try:
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise RuntimeError(
            "ArcFace evaluation requires InsightFace. Install "
            "requirements-evaluation.txt in a separate evaluation environment."
        ) from error

    manifest = args.manifest.expanduser().resolve()
    rows = load_manifest(manifest)
    if args.reference_image:
        reference_path = args.reference_image.expanduser().resolve()
    else:
        matches = [row["image"] for row in rows if row["frame"] == args.reference_frame]
        if len(matches) != 1:
            raise ValueError(f"reference frame {args.reference_frame} is not unique in manifest")
        reference_path = matches[0]

    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    analyzer = FaceAnalysis(
        name=args.model_name,
        root=str(model_root),
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))
    reference_embedding, reference_score, reference_bbox = largest_face_embedding(
        analyzer, reference_path
    )

    results = []
    for row in rows:
        embedding, score, bbox = largest_face_embedding(analyzer, row["image"])
        similarity = float(np.clip(np.dot(reference_embedding, embedding), -1.0, 1.0))
        is_reference = not args.reference_image and row["frame"] == args.reference_frame
        results.append(
            {
                "frame": row["frame"],
                "image": str(row["image"]),
                "is_reference": is_reference,
                "cosine_similarity": similarity,
                "detection_score": score,
                "bbox": bbox,
            }
        )
        print(f"[ARCFACE] frame {row['frame']}: cosine={similarity:.6f}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "arcface_identity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "image", "cosine_similarity", "detection_score"])
        writer.writerows(
            (item["frame"], item["image"], item["cosine_similarity"], item["detection_score"])
            for item in results
        )
    comparisons = [item for item in results if not item["is_reference"]]
    if not comparisons:
        raise ValueError("manifest contains no non-reference image to evaluate")
    similarities = [item["cosine_similarity"] for item in comparisons]
    summary = {
        "model": args.model_name,
        "reference_frame": None if args.reference_image else args.reference_frame,
        "reference_image": str(reference_path),
        "reference_detection_score": reference_score,
        "reference_bbox": reference_bbox,
        "count": len(results),
        "comparison_count": len(comparisons),
        "mean_cosine_similarity": float(np.mean(similarities)),
        "minimum_cosine_similarity": float(np.min(similarities)),
        "maximum_cosine_similarity": float(np.max(similarities)),
        "frames": results,
    }
    (output / "arcface_identity.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[SUCCESS] ArcFace identity: "
        f"mean={summary['mean_cosine_similarity']:.6f}, "
        f"min={summary['minimum_cosine_similarity']:.6f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
