#!/usr/bin/env python3
"""Face2Face-style expression transfer from a source video to a target photo.

Reconstructs the target photograph, tracks per-frame expressions through the
source video, swaps the expression coefficients into the target's fit, and
re-renders the animated face back onto the photograph as a silent video.
Existing reconstruction and tracking outputs are reused when passed directly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate a target photograph with the facial expressions of a "
            "source video."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Target photograph, or an existing reconstruction directory",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source video/image directory, or an existing tracking directory",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "transfers" / "transfer")
    parser.add_argument("--model", type=Path, default=ROOT / "data" / "model2019_face12.h5")
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=ROOT / "data" / "bfm_mediapipe_correspondence.csv",
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--relative",
        action="store_true",
        help=(
            "Add the source's expression change to the target's own expression "
            "instead of replacing it; keeps a non-neutral target expression"
        ),
    )
    parser.add_argument(
        "--expression-scale",
        type=float,
        default=1.0,
        help="Gain applied to the transferred expression coefficients",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=-1,
        help="Compositing feather radius in pixels; -1 scales with image size",
    )
    parser.add_argument("--fps", type=float, help="Output FPS; defaults to the source video")
    parser.add_argument("--max-frames", type=int, help="Optional development/debug limit")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=960,
        help="Longest side used when tracking source video frames",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip the side-by-side source/result comparison video",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def reconstruct_target(target: Path, output: Path, args: argparse.Namespace) -> Path:
    """Returns a reconstruction directory containing fitting.txt and face.ply."""
    if target.is_dir():
        for name in ("fitting.txt", "face.ply", "run.json"):
            if not (target / name).is_file():
                raise FileNotFoundError(
                    f"Reconstruction directory is missing {name}: {target}"
                )
        return target
    run_directory = output / "target_reconstruction"
    run(
        [
            sys.executable,
            str(ROOT / "reconstruct.py"),
            str(target),
            "--name", run_directory.name,
            "--output-root", str(output),
            "--model", str(args.model),
            "--correspondences", str(args.correspondences),
            "--build-dir", str(args.build_dir),
        ]
    )
    return run_directory


def track_source(source: Path, output: Path, args: argparse.Namespace) -> Path:
    """Returns a track_sequence.py output directory with tracking/tracking.csv."""
    if source.is_dir() and (source / "tracking" / "tracking.csv").is_file():
        return source
    tracking_dir = output / "source_tracking"
    command = [
        sys.executable,
        str(ROOT / "track_sequence.py"),
        str(source),
        "--output", str(tracking_dir),
        "--model", str(args.model),
        "--correspondences", str(args.correspondences),
        "--build-dir", str(args.build_dir),
        "--max-dimension", str(args.max_dimension),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    run(command)
    return tracking_dir


def source_fps(tracking_dir: Path, fallback: float = 25.0) -> float:
    sequence_json = tracking_dir / "sequence.json"
    if sequence_json.is_file():
        metadata = json.loads(sequence_json.read_text(encoding="utf-8"))
        fps = float(metadata.get("fps", 0.0))
        if fps > 0:
            return fps
    return fallback


def encode_video(frame_paths: list[Path], output: Path, fps: float) -> None:
    import cv2

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read transfer frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read transfer frame: {path}")
        writer.write(frame)
    writer.release()


def encode_comparison(
    result_frames: list[Path], source_frame_dir: Path, output: Path, fps: float
) -> bool:
    import cv2

    first = cv2.imread(str(result_frames[0]), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    source_frames = sorted(source_frame_dir.glob("frame_*.png"))
    if len(source_frames) < len(result_frames):
        return False
    writer = None
    for result_path, source_path in zip(result_frames, source_frames):
        result = cv2.imread(str(result_path), cv2.IMREAD_COLOR)
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if result is None or source is None:
            return False
        source = cv2.resize(
            source, (round(source.shape[1] * height / source.shape[0]), height)
        )
        combined = cv2.hconcat([source, result])
        if writer is None:
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (combined.shape[1], combined.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create comparison video: {output}")
        writer.write(combined)
    if writer is not None:
        writer.release()
    return writer is not None


def main() -> int:
    args = parse_args()
    target = args.target.expanduser().resolve()
    source = args.source.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target does not exist: {target}")
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    transfer_executable = (args.build_dir / "face_expression_transfer").expanduser().resolve()
    if not transfer_executable.is_file():
        raise FileNotFoundError(
            f"Transfer executable does not exist: {transfer_executable}\n"
            "Run `cmake --build build --parallel` first"
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    reconstruction = reconstruct_target(target, output, args)
    tracking = track_source(source, output, args)
    trajectory = tracking / "tracking" / "tracking.csv"

    run_metadata = json.loads((reconstruction / "run.json").read_text(encoding="utf-8"))
    target_image = Path(run_metadata["input_image"])
    if not target_image.is_file():
        raise FileNotFoundError(
            f"Original target photograph is no longer available: {target_image}"
        )

    command = [
        str(transfer_executable),
        "--model", str(args.model.expanduser().resolve()),
        "--target-fitting", str(reconstruction / "fitting.txt"),
        "--target-mesh", str(reconstruction / "face.ply"),
        "--target-image", str(target_image),
        "--trajectory", str(trajectory),
        "--output", str(output),
        "--expression-scale", str(args.expression_scale),
        "--feather", str(args.feather),
    ]
    if args.relative:
        command.append("--relative")
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    run(command)

    frame_paths = sorted((output / "frames").glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("Expression transfer produced no frames")
    fps = args.fps or source_fps(tracking)
    result_video = output / "transfer.mp4"
    encode_video(frame_paths, result_video, fps)

    comparison_video = None
    if not args.no_comparison:
        comparison_path = output / "comparison.mp4"
        if encode_comparison(frame_paths, tracking / "frames", comparison_path, fps):
            comparison_video = comparison_path

    (output / "transfer.json").write_text(
        json.dumps(
            {
                "target": str(target),
                "source": str(source),
                "target_reconstruction": str(reconstruction),
                "source_tracking": str(tracking),
                "frame_count": len(frame_paths),
                "fps": fps,
                "mode": "relative" if args.relative else "absolute",
                "expression_scale": args.expression_scale,
                "result_video": result_video.name,
                "comparison_video": comparison_video.name if comparison_video else None,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"\nExpression transfer complete: {output}")
    print(f"Result video: {result_video}")
    if comparison_video:
        print(f"Side-by-side comparison: {comparison_video}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
