"""Full Odin1 -> PandaSet-style processing pipeline.

``process_dataset`` orchestrates the pipeline:

    OdinRecording(origin) -> align(image<->cloud) -> pose offset -> write_dataset

The heavy lifting (stream discovery, per-frame indexes, pose interpolation,
alignment, output schema) lives in ``recording`` / ``output``.  This module only
wires the options together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .masks import DEFAULT_SIGMA
from .output import ProcessSummary, write_dataset
from .recording import OdinRecording

logger = logging.getLogger(__name__)


@dataclass
class ProcessOptions:
    origin_dir: str | Path
    out_dir: str | Path
    time_tolerance: float = 1e-3  # seconds, image<->cloud match tolerance
    max_frames: int | None = None  # limit for quick runs / tests
    pose_offset: float | None = None  # frames; None = auto-estimate
    pose_offset_range: tuple[float, float] = (-2.0, 2.0)  # search range when auto
    align_max_samples: int = 40  # frames sampled to estimate the pose offset
    generate_masks: bool = True  # emit per-frame LiDAR/RGB validity + confidence
    mask_sigma: float = DEFAULT_SIGMA  # Gaussian sigma (px) for the confidence map
    workers: int = 8  # parallel writer processes (frames are independent)
    overwrite: bool = False

    def __post_init__(self):
        self.origin_dir = Path(self.origin_dir)
        self.out_dir = Path(self.out_dir)


def process_dataset(opts: ProcessOptions) -> ProcessSummary:
    """Run the pipeline; returns a summary of what was written."""
    rec = OdinRecording(opts.origin_dir)
    logger.info(
        "discovered recording '%s' (%d cloud frames)", rec.run_name, rec.n_cloud
    )

    mapping = rec.align(opts.time_tolerance)
    cloud_indices = sorted(mapping.keys())
    if opts.max_frames is not None:
        cloud_indices = cloud_indices[: opts.max_frames]
    if not cloud_indices:
        raise RuntimeError("no cloud/image frames could be time-aligned")
    logger.info("aligned %d cloud frames to images", len(cloud_indices))

    # Estimation samples a spread of frames; never exceed those actually aligned.
    stride = max(1, len(cloud_indices) // opts.align_max_samples)
    samples = cloud_indices[::stride]

    pose_offset = opts.pose_offset
    estimate = None
    if pose_offset is None:
        estimate = rec.estimate_pose_offset(
            samples,
            tolerance_seconds=opts.time_tolerance,
            lo=opts.pose_offset_range[0],
            hi=opts.pose_offset_range[1],
        )
        pose_offset = estimate.offset_frames
        logger.info(
            "estimated pose time offset = %.2f frames (mean colour dist %.1f)",
            pose_offset,
            estimate.score,
        )
    else:
        pose_offset = float(pose_offset)

    try:
        return write_dataset(
            rec,
            mapping={k: mapping[k] for k in cloud_indices},
            out_dir=opts.out_dir,
            run_name=rec.run_name,
            pose_offset=pose_offset,
            overwrite=opts.overwrite,
            generate_masks=opts.generate_masks,
            mask_sigma=opts.mask_sigma,
            workers=opts.workers,
        )
    finally:
        rec.close()
