#!/usr/bin/env python3
"""Run and summarize a small perspective reconstruction baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "reconstructions" / "_baselines" / "perspective"
)
COEFFICIENT_LIMIT = 3.0
LIMIT_TOLERANCE = 1.0e-3


@dataclass
class BaselineResult:
    image: str
    run_name: str
    status: str
    width: int | None = None
    height: int | None = None
    solution_usable: bool = False
    matched_landmarks: int | None = None
    semantic_landmarks: int | None = None
    contour_landmarks: int | None = None
    rejected_semantic_landmarks: int | None = None
    initial_rmse_normalized: float | None = None
    final_rmse_normalized: float | None = None
    semantic_rmse_pixels: float | None = None
    contour_rmse_pixels: float | None = None
    all_rmse_pixels: float | None = None
    camera_focal_length: float | None = None
    camera_distance: float | None = None
    shape_max_abs: float | None = None
    expression_max_abs: float | None = None
    shape_at_limit_count: int | None = None
    expression_at_limit_count: int | None = None
    visible_pixel_count: int | None = None
    visible_fraction: float | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current perspective pipeline on several photographs "
            "and write comparable CSV/Markdown baseline summaries."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Input photographs. Use distinct people/poses when possible.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing per-image runs and baseline summaries",
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
        help="Also export the viewer's four diagnostic rendering modes",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed image instead of recording the failure",
    )
    return parser.parse_args()


def unique_run_names(images: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for image in images:
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", image.stem).strip("-") or "image"
        counts[base] = counts.get(base, 0) + 1
        suffix = "" if counts[base] == 1 else f"-{counts[base]}"
        names.append(f"{base}{suffix}")
    return names


def parse_scalar(value: str) -> str | int | float:
    value = value.strip()
    if value in {"yes", "no"}:
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_fitting_report(path: Path) -> tuple[dict[str, object], list[float], list[float]]:
    values: dict[str, object] = {}
    shape: list[float] = []
    expression: list[float] = []
    coefficient_pattern = re.compile(r"^(shape|expression)\[(\d+)\]:\s*(.+)$")

    for line in path.read_text(encoding="utf-8").splitlines():
        match = coefficient_pattern.match(line)
        if match:
            destination = shape if match.group(1) == "shape" else expression
            destination.append(float(match.group(3)))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = parse_scalar(value)
    return values, shape, expression


def reprojection_rmse_pixels(
    path: Path, width: int, height: int
) -> tuple[float | None, float | None, float | None]:
    squared_errors: dict[str, list[float]] = {
        "semantic": [],
        "contour": [],
        "all": [],
    }
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            dx = (
                float(row["projected_x_norm"]) - float(row["observed_x_norm"])
            ) * width
            dy = (
                float(row["projected_y_norm"]) - float(row["observed_y_norm"])
            ) * height
            squared = dx * dx + dy * dy
            category = "contour" if row["name"].startswith("contour_") else "semantic"
            squared_errors[category].append(squared)
            squared_errors["all"].append(squared)

    def rmse(samples: list[float]) -> float | None:
        return math.sqrt(sum(samples) / len(samples)) if samples else None

    return (
        rmse(squared_errors["semantic"]),
        rmse(squared_errors["contour"]),
        rmse(squared_errors["all"]),
    )


def summarize_run(image: Path, run_name: str, run_directory: Path) -> BaselineResult:
    decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError(f"OpenCV could not decode {image}")
    height, width = decoded.shape[:2]

    report, shape, expression = parse_fitting_report(
        run_directory / "fitting.txt"
    )
    semantic_rmse, contour_rmse, all_rmse = reprojection_rmse_pixels(
        run_directory / "reprojections.csv", width, height
    )

    visibility = cv2.imread(
        str(run_directory / "visibility.png"), cv2.IMREAD_GRAYSCALE
    )
    if visibility is None:
        raise RuntimeError(f"Could not read {run_directory / 'visibility.png'}")
    visible_pixels = int(cv2.countNonZero(visibility))

    limit_threshold = COEFFICIENT_LIMIT - LIMIT_TOLERANCE
    return BaselineResult(
        image=str(image),
        run_name=run_name,
        status="success",
        width=width,
        height=height,
        solution_usable=report.get("solution_usable") == "yes",
        matched_landmarks=int(report["matched_landmarks"]),
        semantic_landmarks=int(report["semantic_landmarks"]),
        contour_landmarks=int(report["contour_landmarks"]),
        rejected_semantic_landmarks=int(
            report["rejected_semantic_landmarks"]
        ),
        initial_rmse_normalized=float(report["initial_rmse_normalized"]),
        final_rmse_normalized=float(report["final_rmse_normalized"]),
        semantic_rmse_pixels=semantic_rmse,
        contour_rmse_pixels=contour_rmse,
        all_rmse_pixels=all_rmse,
        camera_focal_length=float(
            report["camera_focal_length_normalized"]
        ),
        camera_distance=float(
            str(report["camera_translation"]).split()[2]
        ),
        shape_max_abs=max(map(abs, shape), default=0.0),
        expression_max_abs=max(map(abs, expression), default=0.0),
        shape_at_limit_count=sum(abs(value) >= limit_threshold for value in shape),
        expression_at_limit_count=sum(
            abs(value) >= limit_threshold for value in expression
        ),
        visible_pixel_count=visible_pixels,
        visible_fraction=visible_pixels / float(width * height),
    )


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summaries(
    output_root: Path,
    results: list[BaselineResult],
    command: list[str],
) -> None:
    fieldnames = list(BaselineResult.__dataclass_fields__)
    with (output_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    successful = [result for result in results if result.status == "success"]
    lines = [
        "# Perspective Baseline",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Camera model: perspective",
        "",
        "| Image | Status | Semantic RMSE (px) | Contour RMSE (px) | Rejected | "
        "Shape max | Expr. max | At limit (S/E) | Visible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {Path(result.image).name} | {result.status} | "
            f"{format_number(result.semantic_rmse_pixels)} | "
            f"{format_number(result.contour_rmse_pixels)} | "
            f"{format_number(result.rejected_semantic_landmarks, 0)} | "
            f"{format_number(result.shape_max_abs)} | "
            f"{format_number(result.expression_max_abs)} | "
            f"{format_number(result.shape_at_limit_count, 0)}/"
            f"{format_number(result.expression_at_limit_count, 0)} | "
            f"{format_number(100.0 * result.visible_fraction if result.visible_fraction is not None else None, 1)}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This baseline is a regression reference for the current camera, "
            "fitting, rasterization, and visibility pipeline.",
            "- It does not revalidate the landmark correspondence table.",
            "- The current local images are near-frontal. Add yawed images before "
            "drawing conclusions about perspective-camera improvements.",
            "- A coefficient is counted as being at its limit when "
            f"`abs(coefficient) >= {COEFFICIENT_LIMIT - LIMIT_TOLERANCE}`.",
            "",
            f"Successful runs: {len(successful)}/{len(results)}",
            "",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "camera_model": "perspective",
        "command": command,
        "coefficient_limit": COEFFICIENT_LIMIT,
        "runs": [asdict(result) for result in results],
    }
    (output_root / "baseline.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    images = [image.expanduser().resolve() for image in args.images]
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    names = unique_run_names(images)
    results: list[BaselineResult] = []

    for index, (image, run_name) in enumerate(zip(images, names), start=1):
        print(f"[{index}/{len(images)}] Perspective baseline: {image.name}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "reconstruct.py"),
            str(image),
            "--name",
            run_name,
            "--output-root",
            str(output_root),
            "--model",
            str(args.model.expanduser().resolve()),
            "--correspondences",
            str(args.correspondences.expanduser().resolve()),
            "--build-dir",
            str(args.build_dir.expanduser().resolve()),
            "--no-dense-refinement",
            "--diagnostics",
        ]
        if args.render:
            command.append("--render")

        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            results.append(
                summarize_run(image, run_name, output_root / run_name)
            )
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            results.append(
                BaselineResult(
                    image=str(image),
                    run_name=run_name,
                    status="failed",
                    error=str(error),
                )
            )
            if args.stop_on_error:
                write_summaries(output_root, results, sys.argv)
                raise

    write_summaries(output_root, results, sys.argv)
    failures = sum(result.status != "success" for result in results)
    print(f"\nBaseline summary: {output_root / 'summary.md'}")
    print(f"Machine-readable metrics: {output_root / 'summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
