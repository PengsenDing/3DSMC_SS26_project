#!/usr/bin/env python3
"""Run the complete single-image reconstruction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from face_reconstruction.detect_landmarks import run_detection
from face_reconstruction.segment_face import DEFAULT_MODEL as DEFAULT_SEGMENTATION_MODEL
from face_reconstruction.segment_face import MODEL_URL as SEGMENTATION_MODEL_URL
from face_reconstruction.segment_face import segment_face_skin


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
        default=PROJECT_ROOT / "outputs" / "reconstructions",
        help="Parent directory for reconstruction runs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "assets" / "models" / "model2019_face12.h5",
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
        "--shape-regularization",
        type=float,
        help="Weight of the identity PCA prior (C++ default 0.1)",
    )
    parser.add_argument(
        "--silhouette-weight",
        type=float,
        help="Weight of bidirectional silhouette correspondences (C++ default 10)",
    )
    parser.add_argument(
        "--no-landmark-visibility-filter",
        action="store_true",
        help="Keep self-occluded semantic landmarks in the joint BFM fit",
    )
    parser.add_argument(
        "--landmark-depth-tolerance",
        type=float,
        default=1.0e-4,
        help="Inverse-depth Z-buffer tolerance for landmark visibility",
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
        "--segmentation-model",
        type=Path,
        default=DEFAULT_SEGMENTATION_MODEL,
        help="MediaPipe multiclass model used to generate a face-skin mask",
    )
    parser.add_argument(
        "--landmark-overrides",
        type=Path,
        help=(
            "Optional CSV with index,x_norm,y_norm,enabled columns for "
            "manual landmark correction or disabling"
        ),
    )
    parser.add_argument(
        "--no-pose-landmark-filter",
        action="store_true",
        help="Disable yaw-based rejection of far-side semantic landmarks",
    )
    parser.add_argument(
        "--pose-yaw-threshold",
        type=float,
        default=0.55,
        help="Absolute yaw in radians at which far-side landmarks are rejected",
    )
    parser.add_argument(
        "--landmark-mask-tolerance",
        type=float,
        default=0.015,
        help="Maximum distance outside the face mask, normalized by image size",
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


def apply_landmark_overrides(
    landmarks_path: Path, overrides_path: Path, report_path: Path
) -> None:
    """Apply normalized manual corrections and remove explicitly disabled rows."""
    with landmarks_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Cannot apply overrides to an empty landmark CSV")
    fieldnames = list(rows[0])

    with overrides_path.open(newline="", encoding="utf-8") as input_file:
        override_rows = list(csv.DictReader(input_file))
    overrides: dict[int, dict[str, str]] = {}
    for row in override_rows:
        raw_index = row.get("index") or row.get("mediapipe_index")
        if raw_index is None or raw_index.strip() == "":
            raise ValueError("Every landmark override requires an index")
        overrides[int(raw_index)] = row

    applied: list[dict[str, object]] = []
    output_rows: list[dict[str, str]] = []
    for row in rows:
        index = int(row["index"])
        override = overrides.get(index)
        if override is None:
            output_rows.append(row)
            continue
        enabled = override.get("enabled", "true").strip().lower()
        if enabled in {"false", "0", "no"}:
            applied.append({"index": index, "action": "disabled"})
            continue
        x_value = override.get("x_norm", "").strip()
        y_value = override.get("y_norm", "").strip()
        if bool(x_value) != bool(y_value):
            raise ValueError(
                f"Landmark override {index} must provide both x_norm and y_norm"
            )
        if x_value:
            x_norm = float(x_value)
            y_norm = float(y_value)
            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                raise ValueError(f"Landmark override {index} is outside [0,1]")
            width = float(row["u"]) / max(float(row["x_norm"]), 1.0e-8)
            height = float(row["v"]) / max(float(row["y_norm"]), 1.0e-8)
            row["x_norm"] = str(x_norm)
            row["y_norm"] = str(y_norm)
            row["u"] = str(x_norm * width)
            row["v"] = str(y_norm * height)
            applied.append(
                {
                    "index": index,
                    "action": "corrected",
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                }
            )
        output_rows.append(row)

    missing = sorted(set(overrides) - {int(row["index"]) for row in rows})
    if missing:
        raise ValueError(f"Overrides reference missing landmark indices: {missing}")
    with landmarks_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    report_path.write_text(json.dumps(applied, indent=2) + "\n", encoding="utf-8")


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
    occluder_mask = run_directory / "occluder_mask.png"
    segmentation_preview = run_directory / "face_segmentation.png"
    overrides_report = run_directory / "landmark_overrides.json"
    landmarks_preview = run_directory / "landmarks.png" if args.diagnostics else None

    total_steps = 2
    print(f"[1/{total_steps}] Detecting landmarks in {image.name}")
    landmark_count = run_detection(
        image_path=image,
        csv_path=landmarks_csv,
        debug_path=landmarks_preview,
        correspondence_path=correspondences,
        silhouette_mask_path=None,
    )
    if args.landmark_overrides is not None:
        overrides = require_file(args.landmark_overrides, "Landmark override CSV")
        apply_landmark_overrides(landmarks_csv, overrides, overrides_report)
    if not args.no_silhouette_fitting and args.silhouette_mask is not None:
        supplied_silhouette = require_file(args.silhouette_mask, "Silhouette mask")
        shutil.copyfile(supplied_silhouette, observed_silhouette)
    elif not args.no_silhouette_fitting:
        segmentation_model = args.segmentation_model.expanduser().resolve()
        if not segmentation_model.is_file():
            raise FileNotFoundError(
                f"Face segmentation model does not exist: {segmentation_model}\n"
                f"Download it from:\n{SEGMENTATION_MODEL_URL}"
            )
        segment_face_skin(
            image,
            segmentation_model,
            observed_silhouette,
            segmentation_preview if args.diagnostics else None,
            occluder_mask,
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
        "--albedo-components",
        str(args.albedo_components),
        "--focal-regularization",
        str(args.focal_regularization),
        "--landmark-depth-tolerance",
        str(args.landmark_depth_tolerance),
        "--pose-yaw-threshold",
        str(args.pose_yaw_threshold),
        "--landmark-mask-tolerance",
        str(args.landmark_mask_tolerance),
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
    if args.shape_regularization is not None:
        fitting_command.extend(
            ["--shape-regularization", str(args.shape_regularization)]
        )
    if args.silhouette_weight is not None:
        fitting_command.extend(["--silhouette-weight", str(args.silhouette_weight)])
    if args.no_landmark_visibility_filter:
        fitting_command.append("--no-landmark-visibility-filter")
    if args.no_pose_landmark_filter:
        fitting_command.append("--no-pose-landmark-filter")
    if args.no_silhouette_fitting:
        fitting_command.append("--no-silhouette-fitting")
    else:
        fitting_command.extend(["--silhouette-mask", str(observed_silhouette)])
    if occluder_mask.is_file():
        fitting_command.extend(["--occluder-mask", str(occluder_mask)])
    if args.diagnostics:
        fitting_command.append("--diagnostics")
    if args.verbose_optimization:
        fitting_command.append("--verbose-optimization")
    run_command(fitting_command)

    artifacts = {
        "mesh_off": "face.off",
        "mesh_ply": "face.ply",
        "fitting_report": "fitting.txt",
        "bfm_surface_overlay": "bfm_surface_overlay.png",
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
        if args.silhouette_mask is None and not args.no_silhouette_fitting:
            artifacts["face_segmentation"] = "face_segmentation.png"
            artifacts["occluder_mask"] = "occluder_mask.png"
        if args.landmark_overrides is not None:
            artifacts["landmark_overrides"] = "landmark_overrides.json"
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
    if not args.diagnostics:
        diagnostics = [
            "landmarks.csv",
            "landmarks.png",
            "face_segmentation.png",
            "occluder_mask.png",
            "landmark_overrides.json",
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
