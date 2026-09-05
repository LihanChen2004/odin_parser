#!/usr/bin/env python
"""Render an Odin processed dataset to an MP4 (RGB image + projected point cloud).

Same visual style as ``odin-parser preview``: each frame is the camera image
with the LiDAR points drawn as semi-transparent round dots.

Usage:
  python scripts/make_video.py <dataset_dir> <out.mp4> \
      [--speed 3] [--radius 2] [--alpha 0.7] [--darken 0.85] [--max-frames N]

Defaults render at 3x real-time (the scene is ~10 fps, so output ~30 fps).
"""

from __future__ import annotations

import argparse
import subprocess

import numpy as np

from odin_parser import OdinDataset
from odin_parser.camera import PolynomialCamera
from odin_parser.visualize import (
    DEFAULT_ALPHA,
    DEFAULT_DARKEN,
    DEFAULT_RADIUS,
    render_overlay,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="processed dataset directory")
    parser.add_argument("out", help="output MP4 path")
    parser.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="playback speed-up relative to the recorded rate",
    )
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--darken", type=float, default=DEFAULT_DARKEN)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    ds = OdinDataset(args.dataset_dir)
    intr = ds.intrinsics
    cam = PolynomialCamera(
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        skew=intr.get("skew", 0.0),
        k=tuple(intr.get("distortion_coefficients", (0.0,) * 6)),
    )
    ts = [p["timestamp"] for p in ds.camera_poses]
    base_fps = (len(ts) - 1) / (ts[-1] - ts[0])
    out_fps = base_fps * args.speed
    h, w = intr["image_height"], intr["image_width"]
    n = len(ds) if args.max_frames is None else min(args.max_frames, len(ds))

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{out_fps:.3f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        args.out,
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for k in range(n):
        img = np.asarray(ds.image(k))
        pts = ds.point_cloud(k)
        if pts.shape[0]:
            frame = render_overlay(
                img,
                pts,
                ds.camera_pose(k),
                cam,
                radius=args.radius,
                alpha=args.alpha,
                darken=args.darken,
            )
        else:
            frame = img
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if (k + 1) % 200 == 0:
            print(f"rendered {k + 1}/{n}", flush=True)

    enc.stdin.close()
    enc.wait()
    print(f"done -> {args.out}  fps={out_fps:.2f} frames={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
