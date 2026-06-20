#!/usr/bin/env python3
"""Run the complete single-image reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.detect_landmarks import run_detection

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect MediaPipe landmarks, fit the Basel Face Model, and collect "
            "all artifacts in one per-image directory."
        )
    )
    parser.add_argument("image", type=Path, help="Input face photograph")
    parser.add_argument(
        "--name",
        help="Run directory name; defaults to the input filename stem",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "reconstructions",
        help="Parent directory for reconstruction runs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "data" / "model2019_face12.h5",
        help="Basel Face Model HDF5 file",
    )
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=PROJECT_ROOT / "data" / "bfm_mediapipe_correspondence.csv",
        help="MediaPipe-to-BFM correspondence CSV",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "build",
        help="CMake build directory",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also export albedo, depth, normal, and checkerboard PNGs",
    )
    parser.add_argument(
        "--verbose-optimization",
        action="store_true",
        help="Print Ceres iteration details",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    image = require_file(args.image, "Input image")
    model = require_file(args.model, "BFM model")
    correspondences = require_file(args.correspondences, "Correspondence table")
    fitting_executable = require_file(
        args.build_dir / "face_reconstruction",
        "Fitting executable; run `cmake --build build --parallel` first",
    )

    run_name = args.name or image.stem
    if not run_name or run_name in {".", ".."} or "/" in run_name:
        raise ValueError(f"Invalid run name: {run_name!r}")

    run_directory = args.output_root.expanduser().resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    landmarks_csv = run_directory / "landmarks.csv"
    landmarks_preview = run_directory / "landmarks.png"

    total_steps = 3 if args.render else 2
    print(f"[1/{total_steps}] Detecting landmarks in {image.name}")
    landmark_count = run_detection(
        image_path=image,
        csv_path=landmarks_csv,
        debug_path=landmarks_preview,
        correspondence_path=correspondences,
    )
    print(f"Detected {landmark_count} landmarks")

    print(f"[2/{total_steps}] Fitting the Basel Face Model")
    fitting_command = [
        str(fitting_executable),
        "--model",
        str(model),
        "--image",
        str(image),
        "--landmarks",
        str(landmarks_csv),
        "--correspondences",
        str(correspondences),
        "--output",
        str(run_directory),
        "--output-name",
        "face",
    ]
    if args.verbose_optimization:
        fitting_command.append("--verbose-optimization")
    run_command(fitting_command)

    if args.render:
        print("[3/3] Rendering diagnostic image channels")
        viewer = require_file(
            args.build_dir / "landmark_viewer",
            "Renderer executable; run `cmake --build build --parallel` first",
        )
        run_command(
            [
                str(viewer),
                str(run_directory / "face_aligned.ply"),
                "--render-all",
                str(run_directory / "renders"),
            ]
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_image": str(image),
        "bfm_model": str(model),
        "landmark_count": landmark_count,
        "artifacts": {
            "landmarks": "landmarks.csv",
            "landmark_preview": "landmarks.png",
            "mesh_off": "face.off",
            "mesh_ply": "face.ply",
            "aligned_mesh_ply": "face_aligned.ply",
            "fitting_report": "fitting.txt",
            "reprojections": "reprojections.csv",
            "overlay": "overlay.png",
        },
    }
    if args.render:
        manifest["artifacts"]["renders"] = "renders/"
    (run_directory / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nReconstruction complete: {run_directory}")
    print(f"MeshLab mesh: {run_directory / 'face.off'}")
    print(f"Colored mesh: {run_directory / 'face_aligned.ply'}")
    print(f"Fitting check: {run_directory / 'overlay.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
