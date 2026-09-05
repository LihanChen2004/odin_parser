"""Timestamp-based alignment helpers.

The Odin1 streams are recorded independently; indices do *not* match across
streams.  The image stream, in particular, starts ~0.88 s (~9 frames at 10 Hz)
before the SLAM cloud / odometry / WIWC streams.

Everything is aligned by timestamp.  For this dataset the mapping is a perfect
constant offset of 9 frames with zero drift over the whole recording, but the
code below uses bisection so it stays correct even if the offset drifts.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable


def nearest_index(sorted_ts: list[float], target: float) -> int:
    """Index of the element in ``sorted_ts`` closest to ``target``."""
    pos = bisect.bisect_left(sorted_ts, target)
    if pos == 0:
        return 0
    if pos == len(sorted_ts):
        return len(sorted_ts) - 1
    lo, hi = sorted_ts[pos - 1], sorted_ts[pos]
    return pos if (hi - target) < (target - lo) else pos - 1


def align_cloud_to_image(
    cloud_timestamps: Iterable[tuple[int, float]],
    image_timestamps: Iterable[tuple[int, float]],
    tolerance: float = 1e-3,
) -> dict[int, int]:
    """Map cloud frame index -> aligned image frame index.

    ``cloud_timestamps``/``image_timestamps`` are ``(index, timestamp)`` pairs
    sorted by timestamp.  Returns ``{cloud_index: image_index}`` for cloud
    frames that have an image within ``tolerance`` seconds (there may be none
    at the stream boundaries).
    """
    img = sorted(image_timestamps, key=lambda kv: kv[1])
    img_ts = [ts for _, ts in img]
    mapping: dict[int, int] = {}
    for c_idx, c_ts in sorted(cloud_timestamps, key=lambda kv: kv[1]):
        j = nearest_index(img_ts, c_ts)
        if abs(img_ts[j] - c_ts) <= tolerance:
            mapping[c_idx] = img[j][0]
    return mapping
