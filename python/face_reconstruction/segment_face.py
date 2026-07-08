#!/usr/bin/env python3
"""Generate a face-skin silhouette with MediaPipe multiclass segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

FACE_SKIN_CLASS = 3
# Multiclass segmenter categories that can occlude the face: hair, body skin
# (hands/arms), clothes, and accessories ("others").
OCCLUDER_CLASSES = (1, 2, 4, 5)
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "models"
    / "selfie_multiclass_256x256.tflite"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/"
    "selfie_multiclass_256x256.tflite"
)


def postprocess_face_mask(category_mask: Any, cv2_module: Any, np_module: Any) -> Any:
    """Keep the largest face-skin component and fill internal feature holes."""
    binary = np_module.where(category_mask == FACE_SKIN_CLASS, 255, 0).astype(
        np_module.uint8
    )
    minimum_dimension = min(binary.shape[:2])
    kernel_size = max(3, int(round(minimum_dimension * 0.012)) | 1)
    kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = cv2_module.morphologyEx(binary, cv2_module.MORPH_CLOSE, kernel)

    contours, _ = cv2_module.findContours(
        binary, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Face segmenter did not produce a face-skin component")
    largest = max(contours, key=cv2_module.contourArea)
    if cv2_module.contourArea(largest) < 0.0025 * binary.shape[0] * binary.shape[1]:
        raise RuntimeError("Face-skin segmentation is too small to be reliable")

    filled = np_module.zeros_like(binary)
    cv2_module.drawContours(filled, [largest], -1, 255, thickness=cv2_module.FILLED)
    return filled


def build_occluder_mask(category_mask: Any, np_module: Any) -> Any:
    """Mark every pixel belonging to a class that can cover the face."""
    occluder = np_module.zeros(category_mask.shape[:2], dtype=np_module.uint8)
    for occluder_class in OCCLUDER_CLASSES:
        occluder[category_mask == occluder_class] = 255
    return occluder


def segment_face_skin(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    debug_path: Path | None = None,
    occluder_output: Path | None = None,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face segmentation model does not exist: {model_path}\n"
            f"Download it from:\n{MODEL_URL}"
        )

    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        )
    if result.category_mask is None:
        raise RuntimeError("Face segmenter returned no category mask")
    category_mask = result.category_mask.numpy_view()
    mask = postprocess_face_mask(category_mask, cv2, np)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask):
        raise RuntimeError(f"Could not save face segmentation mask: {output_path}")

    if occluder_output is not None:
        occluder = build_occluder_mask(category_mask, np)
        occluder_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(occluder_output), occluder):
            raise RuntimeError(f"Could not save occluder mask: {occluder_output}")

    if debug_path is not None:
        tint = image_bgr.copy()
        tint[mask > 0] = (
            0.55 * tint[mask > 0] + 0.45 * np.asarray([255, 180, 40])
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(tint, contours, -1, (255, 220, 60), 2, cv2.LINE_AA)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(debug_path), tint):
            raise RuntimeError(f"Could not save segmentation preview: {debug_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug", type=Path)
    parser.add_argument("--occluders", type=Path)
    args = parser.parse_args()
    segment_face_skin(args.image, args.model, args.output, args.debug, args.occluders)
    print(f"Saved face-skin mask: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
