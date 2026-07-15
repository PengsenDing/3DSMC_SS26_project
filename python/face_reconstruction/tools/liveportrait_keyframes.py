#!/usr/bin/env python3
"""Generate a yaw x pitch target-keyframe grid with LivePortrait.

Drives LivePortrait's image-retargeting pipeline programmatically (no Gradio
UI): one target photograph in, one synthesized full-resolution image per
requested yaw out. The generated keyframes are meant to be registered with
the BFM pipeline afterwards (face_sequence_tracker with the target's
fitting.txt as fixed identity) and used as additional projective-texture
sources for large-pose reenactment.

Findings from target_3 (2026-07-06): requested yaw is damped nonlinearly by
LivePortrait's stitching (request +-25 delivers roughly +-20 of true rotation,
sign convention is opposite to CameraYaw) — always use the registered fitted
camera, never the nominal request. Requests up to +-40 still produce clean,
identity-preserving views.

Usage:
  <liveportrait>/.venv/bin/python python/face_reconstruction/tools/liveportrait_keyframes.py \
      photo.png --liveportrait-root <liveportrait> --output keyframes/
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def partial_fields(target_class, kwargs):
    return target_class(
        **{k: v for k, v in kwargs.items() if hasattr(target_class, k)}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Target photograph")
    parser.add_argument(
        "--liveportrait-root", required=True,
        help="Path to the LivePortrait repository checkout",
    )
    parser.add_argument("--output", default="keyframes")
    parser.add_argument(
        "--yaws", default="-40,-25,-15,0,15,25,40",
        help="Comma-separated relative yaw angles in degrees",
    )
    parser.add_argument(
        "--pitches", default="-15,0,15",
        help="Comma-separated relative pitch angles in degrees",
    )
    parser.add_argument("--crop-scale", type=float, default=2.5)
    args = parser.parse_args()

    import cv2

    root = os.path.abspath(os.path.expanduser(args.liveportrait_root))
    sys.path.insert(0, root)
    previous_cwd = os.getcwd()
    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    # LivePortrait resolves its model paths relative to its repo root.
    os.chdir(root)

    from src.config.argument_config import ArgumentConfig
    from src.config.crop_config import CropConfig
    from src.config.inference_config import InferenceConfig
    from src.gradio_pipeline import GradioPipeline

    lp_args = ArgumentConfig()
    pipeline = GradioPipeline(
        inference_cfg=partial_fields(InferenceConfig, lp_args.__dict__),
        crop_cfg=partial_fields(CropConfig, lp_args.__dict__),
        args=lp_args,
    )
    # Locks the retargeting sliders to the photo's own eye/lip opening so the
    # only edit applied per keyframe is the head rotation.
    eye_ratio, lip_ratio = pipeline.init_retargeting_image(
        args.crop_scale, 0.0, 0.0, source
    )
    print(f"source eye ratio {eye_ratio}, lip ratio {lip_ratio}")

    os.makedirs(output, exist_ok=True)
    yaws = [float(v) for v in args.yaws.split(",")]
    pitches = [float(v) for v in args.pitches.split(",")]
    for pitch in pitches:
        for yaw in yaws:
            _, blended = pipeline.execute_image_retargeting(
                input_eye_ratio=eye_ratio,
                input_lip_ratio=lip_ratio,
                input_head_pitch_variation=pitch,
                input_head_yaw_variation=yaw,
                input_head_roll_variation=0.0,
                mov_x=0.0,
                mov_y=0.0,
                mov_z=1.0,
                lip_variation_zero=0.0,
                lip_variation_one=0.0,
                lip_variation_two=0.0,
                lip_variation_three=0.0,
                smile=0.0,
                wink=0.0,
                eyebrow=0.0,
                eyeball_direction_x=0.0,
                eyeball_direction_y=0.0,
                input_image=source,
                retargeting_source_scale=args.crop_scale,
            )
            name = f"yaw{yaw:+05.1f}_pitch{pitch:+05.1f}"
            cv2.imwrite(
                os.path.join(output, f"{name}.png"),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
            )
            print(f"wrote {name} ({blended.shape[1]}x{blended.shape[0]})")
    os.chdir(previous_cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
