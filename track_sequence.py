#!/usr/bin/env python3
"""Optional video/image-sequence tracking built on the single-image pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.detect_landmarks import run_detection

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the first frame, then track fixed identity through a sequence."
    )
    parser.add_argument("input", type=Path, help="Video file or directory of ordered images")
    parser.add_argument("--output", type=Path, default=ROOT / "sequences" / "tracking")
    parser.add_argument("--model", type=Path, default=ROOT / "data" / "model2019_face12.h5")
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=ROOT / "data" / "bfm_mediapipe_correspondence.csv",
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--fps", type=float, help="Output FPS for image-directory inputs")
    parser.add_argument("--max-frames", type=int, help="Optional development/debug limit")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=960,
        help="Downscale tracking frames so their longest side is at most this size",
    )
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Keep first-frame diagnostics")
    parser.add_argument("--pose-temporal-weight", type=float, default=2.0)
    parser.add_argument("--expression-temporal-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--failure-rmse", type=float, default=0.06)
    parser.add_argument("--tracking-iterations", type=int, default=10)
    return parser.parse_args()


def resize_for_tracking(image, max_dimension: int, cv2):
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def extract_frames(
    source: Path, destination: Path, limit: int | None, max_dimension: int
) -> tuple[list[Path], float]:
    import cv2

    destination.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        candidates = sorted(
            path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if limit is not None:
            candidates = candidates[:limit]
        frames = []
        for index, image in enumerate(candidates):
            target = destination / f"frame_{index:06d}.png"
            data = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if data is not None:
                data = resize_for_tracking(data, max_dimension, cv2)
            if data is None or not cv2.imwrite(str(target), data):
                raise RuntimeError(f"Could not read sequence image: {image}")
            frames.append(target)
        return frames, 25.0

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    frames = []
    index = 0
    while limit is None or index < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frame = resize_for_tracking(frame, max_dimension, cv2)
        target = destination / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Could not write extracted frame: {target}")
        frames.append(target)
        index += 1
    capture.release()
    return frames, fps


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def encode_overlay_video(frames: list[Path], overlay_dir: Path, output: Path, fps: float) -> None:
    import cv2

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")
    for index, frame_path in enumerate(frames):
        overlay = overlay_dir / f"frame_{index}.png"
        image = cv2.imread(str(overlay if overlay.is_file() else frame_path), cv2.IMREAD_COLOR)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))
        writer.write(image)
    writer.release()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    model = args.model.expanduser().resolve()
    correspondences = args.correspondences.expanduser().resolve()
    tracker = (args.build_dir / "face_sequence_tracker").expanduser().resolve()
    for path, description in ((model, "BFM model"), (correspondences, "correspondences"), (tracker, "tracker executable")):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")

    output = args.output.expanduser().resolve()
    frame_dir = output / "frames"
    landmark_dir = output / "landmarks"
    output.mkdir(parents=True, exist_ok=True)
    if args.max_dimension <= 0:
        raise ValueError("--max-dimension must be positive")
    frames, detected_fps = extract_frames(
        source, frame_dir, args.max_frames, args.max_dimension
    )
    if not frames:
        raise RuntimeError("The input contains no readable frames")
    fps = args.fps or detected_fps

    landmark_dir.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    rows = []
    for index, frame in enumerate(frames):
        landmarks = landmark_dir / f"frame_{index:06d}.csv"
        run_detection(frame, landmarks, None, correspondences)
        rows.append((index, frame, landmarks))
        print(f"[LANDMARKS] {index + 1}/{len(frames)}", flush=True)
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "image", "landmarks"])
        writer.writerows(rows)

    initial_name = "initial_reconstruction"
    reconstruction = [
        sys.executable,
        str(ROOT / "reconstruct.py"),
        str(frames[0]),
        "--name", initial_name,
        "--output-root", str(output),
        "--model", str(model),
        "--correspondences", str(correspondences),
        "--build-dir", str(args.build_dir.expanduser().resolve()),
        "--photometric-stride", "4",
        "--texture-stride", "4",
    ]
    if args.diagnostics:
        reconstruction.append("--diagnostics")
    run(reconstruction)

    tracking_dir = output / "tracking"
    command = [
        str(tracker),
        "--model", str(model),
        "--correspondences", str(correspondences),
        "--manifest", str(manifest),
        "--initial-fitting", str(output / initial_name / "fitting.txt"),
        "--output", str(tracking_dir),
        "--pose-temporal-weight", str(args.pose_temporal_weight),
        "--expression-temporal-weight", str(args.expression_temporal_weight),
        "--acceleration-weight", str(args.acceleration_weight),
        "--failure-rmse", str(args.failure_rmse),
        "--iterations", str(args.tracking_iterations),
    ]
    if args.save_meshes:
        command.append("--save-meshes")
    run(command)
    encode_overlay_video(frames, tracking_dir / "overlays", output / "tracking_overlay.mp4", fps)
    (output / "sequence.json").write_text(
        json.dumps(
            {
                "input": str(source),
                "frame_count": len(frames),
                "fps": fps,
                "initial_reconstruction": initial_name,
                "trajectory": "tracking/tracking.csv",
                "overlay_video": "tracking_overlay.mp4",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nSequence tracking complete: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
