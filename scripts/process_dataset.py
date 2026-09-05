#!/usr/bin/env python
"""Process an Odin1 recording into the PandaSet-style layout.

Convenience wrapper around the odin-parser CLI:

    uv run python scripts/process_dataset.py \
        /data/origin/<run> \
        --out /data/processed/<run>
"""

from __future__ import annotations

import argparse
import logging

from odin_parser.cli import _setup_logging
from odin_parser.process import ProcessOptions, process_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="raw Odin1 recording directory")
    parser.add_argument("--out", "-o", required=True, help="output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log", default="info")
    args = parser.parse_args()

    _setup_logging(args.log)
    summary = process_dataset(
        ProcessOptions(
            origin_dir=args.origin,
            out_dir=args.out,
            overwrite=args.overwrite,
            max_frames=args.max_frames,
        )
    )
    logging.getLogger("scripts").info(
        "done: %d frames, %.1fM points, %.1f MB images -> %s",
        summary.num_frames,
        summary.cloud_points / 1e6,
        summary.image_bytes / 1e6,
        summary.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
