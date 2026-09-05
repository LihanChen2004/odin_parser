"""Command line interface for odin-parser.

Subcommands
-----------
info     - summarise a raw Odin1 recording directory.
process  - convert a raw recording to the PandaSet-style processed layout.
preview  - overlay a point cloud on a camera image (needs a processed dir).

Examples
--------
    odin-parser info  /data/origin/20260424_231347
    odin-parser process /data/origin/20260424_231347 \
        --out /data/processed/20260424_231347 --overwrite
    odin-parser preview /data/processed/20260424_231347 --frame 500 --out ov.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .process import ProcessOptions, process_dataset
from .recording import find_streams
from .streams import (
    POSE_HDR_SIZE,
    cloud_timestamps,
    image_timestamps,
)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def cmd_info(args: argparse.Namespace) -> int:
    origin = Path(args.origin)
    streams = find_streams(origin)
    if not streams:
        print(f"no Odin streams found under {origin}")
        return 1
    for name, path in sorted(streams.items()):
        print(f"{name:12s} {path.stat().st_size / 1e6:10.1f} MB  {path}")
    if "cloud" in streams:
        cloud = streams["cloud"]
        img = streams.get("OdinImage.bin")
        pose = streams.get("OdinPose.bin")
        cl = cloud_timestamps(cloud)
        print(f"\ncloud frames: {len(cl)}  first={cl[0][1]:.6f}  last={cl[-1][1]:.6f}")
        if img:
            im = image_timestamps(img)
            print(
                f"image frames: {len(im)}  first={im[0][1]:.6f}  last={im[-1][1]:.6f}"
            )
            dt = im[0][1] - cl[0][1]
            dt_frame = (im[1][1] - im[0][1]) if len(im) > 1 else 0.0
            lead = round(dt / dt_frame) if dt_frame else 0
            print(
                f"image stream leads cloud by {dt:+.4f} s (~{lead} frames at {1 / dt_frame:.1f} Hz)"
            )
        if pose:
            print(f"pose frame size: {POSE_HDR_SIZE} bytes")
    calib = origin / "calib_online.yaml"
    if calib.exists():
        print(f"online calib: {calib.stat().st_size / 1e6:.2f} MB")
    for f in ("image/cam_in_ex.txt", "image/info.txt"):
        p = origin / f
        if p.exists():
            print(f"{f}: {p.stat().st_size} bytes")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    opts = ProcessOptions(
        origin_dir=args.origin,
        out_dir=args.out,
        time_tolerance=args.tolerance,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        generate_masks=args.generate_masks,
        mask_sigma=args.mask_sigma,
        workers=args.workers,
    )
    summary = process_dataset(opts)
    print(
        f"done: {summary.num_frames} frames, {summary.cloud_points / 1e6:.1f}M points, "
        f"{summary.image_bytes / 1e6:.1f} MB images -> {summary.out_dir}"
    )
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from PIL import Image

    from .camera import PolynomialCamera
    from .dataset import OdinDataset
    from .visualize import render_depth, render_overlay

    ds = OdinDataset(args.dataset)
    if args.frame >= len(ds):
        print(f"frame {args.frame} out of range (0..{len(ds) - 1})")
        return 1

    intr = ds.intrinsics
    camera = PolynomialCamera(
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        skew=intr.get("skew", 0.0),
        k=tuple(intr.get("distortion_coefficients", (0.0,) * 6)),
    )
    img = np.asarray(ds.image(args.frame))
    cloud = ds.point_cloud(args.frame)
    T_cam_world = ds.camera_pose(args.frame)

    if args.mode == "depth":
        out = render_depth(img, cloud, T_cam_world, camera)
    else:
        out = render_overlay(img, cloud, T_cam_world, camera)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    print(f"saved overlay -> {out_path}  (frame {args.frame}, {cloud.shape[0]} points)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odin-parser",
        description="Parse ManifoldTech Odin1 private binary datasets into a "
        "PandaSet-style layout for gsplat.",
    )
    parser.add_argument("--log", default="info", help="log level (debug/info/warn)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="summarise a raw recording directory")
    p_info.add_argument("origin", help="raw recording directory")

    p_proc = sub.add_parser("process", help="convert raw recording -> processed tree")
    p_proc.add_argument("origin", help="raw recording directory")
    p_proc.add_argument("--out", "-o", required=True, help="output directory")
    p_proc.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="image<->cloud time match tolerance (s)",
    )
    p_proc.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="limit number of frames (for testing)",
    )
    p_proc.add_argument(
        "--generate-masks",
        action="store_true",
        default=True,
        help="emit per-frame mask/confidence (default on)",
    )
    p_proc.add_argument(
        "--no-masks",
        dest="generate_masks",
        action="store_false",
        help="skip per-frame mask/confidence",
    )
    p_proc.add_argument(
        "--mask-sigma",
        type=float,
        default=40.0,
        help="Gaussian sigma (px) for the confidence map",
    )
    p_proc.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel writer threads",
    )
    p_proc.add_argument("--overwrite", action="store_true")

    p_prev = sub.add_parser(
        "preview", help="overlay cloud onto image from a processed dir"
    )
    p_prev.add_argument("dataset", help="processed dataset directory")
    p_prev.add_argument("--frame", "-f", type=int, required=True)
    p_prev.add_argument("--out", "-o", required=True, help="output image path")
    p_prev.add_argument("--mode", choices=["color", "depth"], default="color")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log)
    handlers = {
        "info": cmd_info,
        "process": cmd_process,
        "preview": cmd_preview,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - report nicely at the CLI boundary
        logging.getLogger("cli").error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
