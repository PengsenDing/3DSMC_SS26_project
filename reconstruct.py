#!/usr/bin/env python3
"""Run the complete single-image reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
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
        "--diagnostics",
        action="store_true",
        help="Save detailed landmark, rasterization, and fitting diagnostics",
    )
    parser.add_argument(
        "--verbose-optimization",
        action="store_true",
        help="Print Ceres iteration details",
    )
    parser.add_argument(
        "--albedo-components",
        type=int,
        default=30,
        help="Number of BFM albedo PCA coefficients",
    )
    parser.add_argument(
        "--focal-regularization",
        type=float,
        default=0.25,
        help="Weight keeping perspective focal length near a portrait prior",
    )
    parser.add_argument(
        "--silhouette-mask",
        type=Path,
        help=(
            "Optional external binary face mask; defaults to a dense mask "
            "rasterized from MediaPipe's face oval"
        ),
    )
    parser.add_argument(
        "--no-silhouette-fitting",
        action="store_true",
        help="Disable mask-driven identity refinement",
    )
    parser.add_argument(
        "--silhouette-resolution",
        type=int,
        default=192,
        help="Maximum mask dimension used for silhouette fitting",
    )
    parser.add_argument(
        "--silhouette-iterations",
        type=int,
        default=3,
        help="Number of view-dependent silhouette rematching iterations",
    )
    parser.add_argument(
        "--photometric-stride",
        type=int,
        default=2,
        help="Use every Nth visible pixel during appearance fitting",
    )
    parser.add_argument(
        "--texture-stride",
        type=int,
        default=1,
        help="Use every Nth visible pixel during direct RGB texture fitting",
    )
    parser.add_argument(
        "--texture-prior",
        type=float,
        default=0.02,
        help="Weight retaining initialized vertex colors",
    )
    parser.add_argument(
        "--texture-smoothness",
        type=float,
        default=0.01,
        help="Mesh-edge smoothness weight for fitted vertex colors",
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
    observed_silhouette = run_directory / "observed_silhouette.png"
    landmarks_preview = run_directory / "landmarks.png" if args.diagnostics else None

    total_steps = 3 if args.render else 2
    print(f"[1/{total_steps}] Detecting landmarks in {image.name}")
    landmark_count = run_detection(
        image_path=image,
        csv_path=landmarks_csv,
        debug_path=landmarks_preview,
        correspondence_path=correspondences,
        silhouette_mask_path=(
            observed_silhouette
            if not args.no_silhouette_fitting and args.silhouette_mask is None
            else None
        ),
    )
    if not args.no_silhouette_fitting and args.silhouette_mask is not None:
        supplied_silhouette = require_file(args.silhouette_mask, "Silhouette mask")
        shutil.copyfile(supplied_silhouette, observed_silhouette)
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
        "--albedo-components",
        str(args.albedo_components),
        "--focal-regularization",
        str(args.focal_regularization),
        "--silhouette-resolution",
        str(args.silhouette_resolution),
        "--silhouette-iterations",
        str(args.silhouette_iterations),
        "--photometric-stride",
        str(args.photometric_stride),
        "--texture-stride",
        str(args.texture_stride),
        "--texture-prior",
        str(args.texture_prior),
        "--texture-smoothness",
        str(args.texture_smoothness),
    ]
    if args.no_silhouette_fitting:
        fitting_command.append("--no-silhouette-fitting")
    else:
        fitting_command.extend(["--silhouette-mask", str(observed_silhouette)])
    if args.diagnostics:
        fitting_command.append("--diagnostics")
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
                str(run_directory / "face.ply"),
                "--render-all",
                str(run_directory / "renders"),
            ]
        )

    artifacts = {
        "mesh_off": "face.off",
        "mesh_ply": "face.ply",
        "fitting_report": "fitting.txt",
        "rendered_final": "rendered_final.png",
        "rendered_final_overlay": "rendered_final_overlay.png",
    }
    if not args.no_silhouette_fitting:
        artifacts["silhouette_fitting_report"] = "silhouette_fitting.txt"
    if args.diagnostics:
        artifacts.update(
            {
                "landmarks": "landmarks.csv",
                "landmark_preview": "landmarks.png",
                "aligned_mesh_ply": "face_aligned.ply",
                "reprojections": "reprojections.csv",
                "overlay": "overlay.png",
                "raster_depth": "raster_depth.png",
                "visibility": "visibility.png",
                "intrinsic_albedo_mesh": "face_albedo.ply",
                "photometric_report": "photometric.txt",
                "texture_fitting_report": "texture_fitting.txt",
                "photometric_mask": "photometric_mask.png",
                "texture_mask": "texture_mask.png",
                "mean_albedo_render": "mean_albedo_render.png",
                "estimated_albedo": "estimated_albedo.png",
                "camera_normal": "camera_normal.png",
                "rendered_initial": "rendered_initial.png",
                "rendered_illumination": "rendered_illumination.png",
                "rendered_intrinsic": "rendered_intrinsic.png",
                "rendered_intrinsic_overlay": "rendered_intrinsic_overlay.png",
                "photometric_residual": "photometric_residual.png",
                "texture_residual": "texture_residual.png",
            }
        )
        if not args.no_silhouette_fitting:
            artifacts.update(
                {
                    "observed_silhouette": "observed_silhouette.png",
                    "silhouette_target": "silhouette_target.png",
                    "silhouette_initial": "silhouette_initial.png",
                    "silhouette_refined": "silhouette_refined.png",
                    "silhouette_overlay": "silhouette_overlay.png",
                }
            )
    if args.render:
        artifacts["renders"] = "renders/"

    if not args.diagnostics:
        diagnostics = [
            "landmarks.csv",
            "landmarks.png",
            "face_aligned.ply",
            "face_albedo.ply",
            "reprojections.csv",
            "overlay.png",
            "raster_depth.png",
            "visibility.png",
            "observed_silhouette.png",
            "silhouette_target.png",
            "silhouette_initial.png",
            "silhouette_refined.png",
            "silhouette_overlay.png",
            "photometric.txt",
            "texture_fitting.txt",
            "photometric_mask.png",
            "texture_mask.png",
            "mean_albedo_render.png",
            "estimated_albedo.png",
            "camera_normal.png",
            "rendered_initial.png",
            "rendered_illumination.png",
            "rendered_intrinsic.png",
            "rendered_intrinsic_overlay.png",
            "photometric_residual.png",
            "texture_residual.png",
        ]
        for filename in diagnostics:
            path = run_directory / filename
            if path.is_file():
                path.unlink()
    if not args.render:
        shutil.rmtree(run_directory / "renders", ignore_errors=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_image": str(image),
        "bfm_model": str(model),
        "landmark_count": landmark_count,
        "artifacts": artifacts,
    }
    (run_directory / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nReconstruction complete: {run_directory}")
    print(f"MeshLab mesh: {run_directory / 'face.off'}")
    print(f"Colored mesh: {run_directory / 'face.ply'}")
    print(f"Final rendering: {run_directory / 'rendered_final_overlay.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
