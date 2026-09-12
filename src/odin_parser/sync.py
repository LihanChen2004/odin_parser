"""Cross-sensor time offset estimation.

The odometry pose associated with a cloud/image frame may lag the frame's true
capture instant by a small, constant offset (observed ~1 frame at ~10 fps on
one device; the driver notes the cloud timestamp is ~100 ms ahead of odometry).
This is data-driven rather than hard-coded: it is estimated per recording, and
is expressed in **frames** so it is independent of the frame rate.

``estimate_pose_offset`` scans a candidate offset along the pose timeline. For
each candidate it interpolates the odometry pose (slerp + linear position) and
scores how well the reprojected cloud aligns with the image (colour coherence).
The offset that minimises the mean score is returned, refined to sub-frame
precision via a parabolic fit over neighbouring scan steps.

The interface is callback-based so it works on raw streams without loading the
whole recording; see ``odin_parser.process`` for the wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .camera import PolynomialCamera
from .geometry import interpolate_pose, invert, pose_to_matrix

# Callables handed to the estimator (index = a cloud/output-frame index):
#   read_pose(i)   -> (position (3,), quaternion (4,)) for integer frame i
#   read_image(i)  -> RGB image array (H,W,3)
#   read_cloud(i)  -> (N,7) float32 points in the world frame
#   read_calib(i)  -> (T_cl, T_il) 4x4 matrices for that frame
PoseFn = Callable[[int], tuple[np.ndarray, np.ndarray]]
ImageFn = Callable[[int], np.ndarray]
CloudFn = Callable[[int], np.ndarray]
CalibFn = Callable[[int], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OffsetEstimate:
    offset_frames: float  # pose time relative to the frame index (in frames)
    score: float  # mean colour-distance at the optimum
    scanned: int  # number of candidate offsets tested
    samples: int  # number of frames used for scoring


def _score_frame(
    pose_t: np.ndarray,
    image: np.ndarray,
    cloud: np.ndarray,
    til: np.ndarray,
    tcl: np.ndarray,
    camera: PolynomialCamera,
    max_depth: float,
) -> float | None:
    """Mean colour distance between reprojected points and image pixels."""
    T_wi = pose_t
    T_wc = T_wi @ til @ invert(tcl)
    pc = (invert(T_wc) @ np.hstack([cloud[:, :3], np.ones((cloud.shape[0], 1))]).T).T
    uv = camera.world_to_pixel(pc)
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    h, w = image.shape[:2]
    ok = (
        (pc[:, 2] > 0.05)
        & (pc[:, 2] < max_depth)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
        & np.isfinite(pc).all(axis=1)
    )
    if ok.sum() < 100:
        return None
    return float(np.linalg.norm(cloud[ok][:, 3:6] - image[v[ok], u[ok]], axis=1).mean())


def _pose_at_fraction(
    read_pose: PoseFn,
    index: float,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Pose matrix interpolated at a fractional frame index (clamped)."""
    lo = int(np.floor(index))
    hi = int(np.ceil(index))
    t = index - lo
    if lo not in cache:
        cache[lo] = read_pose(lo)
    if hi == lo:
        pos, quat = cache[lo]
    else:
        if hi not in cache:
            cache[hi] = read_pose(hi)
        pos, quat = interpolate_pose(
            cache[lo][0], cache[lo][1], cache[hi][0], cache[hi][1], t
        )
    return pose_to_matrix(pos, quat)


def estimate_pose_offset(
    read_pose: PoseFn,
    read_image: ImageFn,
    read_cloud: CloudFn,
    read_calib: CalibFn,
    camera: PolynomialCamera,
    indices: list[int],
    max_depth: float = 200.0,
    lo: float = -2.0,
    hi: float = 2.0,
    step: float = 0.25,
) -> OffsetEstimate:
    """Estimate the pose time offset (in frames) for a set of frames.

    ``indices`` are cloud/output-frame indices to score; spread them across the
    recording.  The returned ``offset_frames`` is added to a frame index when
    sampling the odometry pose (``pose_frame = index + offset``).
    """
    # Preload per-frame data once.
    images = [read_image(i) for i in indices]
    clouds = [read_cloud(i) for i in indices]
    calibs = [read_calib(i) for i in indices]

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def mean_score(off: float) -> float | None:
        scores = []
        for k, (im, cld, (tcl, til)) in zip(
            indices, zip(images, clouds, calibs, strict=True), strict=True
        ):
            pose = _pose_at_fraction(read_pose, float(k) + off, cache)
            s = _score_frame(pose, im, cld, til, tcl, camera, max_depth)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else None

    phases = np.arange(lo, hi + 1e-9, step)
    vals = [(off, mean_score(off)) for off in phases]
    vals = [(off, s) for off, s in vals if s is not None]
    if not vals:
        raise RuntimeError("pose-offset estimation produced no valid scores")

    best_off, best_score = min(vals, key=lambda kv: kv[1])

    # Sub-frame refinement: fit a parabola through the best scan point and its
    # two neighbours.
    refined = best_off
    by_off = dict(vals)
    if (best_off - step) in by_off and (best_off + step) in by_off:
        x0, y0 = best_off - step, by_off[best_off - step]
        x1, y1 = best_off, best_score
        x2, y2 = best_off + step, by_off[best_off + step]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-12:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (
                x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)
            ) / denom
            if a != 0:
                refined = -b / (2.0 * a)

    # Ensure the refined point is not off the search range.
    refined = float(np.clip(refined, lo, hi))
    return OffsetEstimate(
        offset_frames=refined,
        score=float(best_score),
        scanned=len(phases),
        samples=len(indices),
    )
