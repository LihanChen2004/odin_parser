"""Per-frame LiDAR <-> RGB validity masks and confidence maps.

Two complementary signals are produced for every emitted frame:

- ``fov_mask`` - the hard LiDAR dToF coverage region projected into the image
  (a convex hull of the projected cloud points).  1 inside the coverage, 0
  outside.  This is the "where is LiDAR/RGB both valid" region.
- ``confidence_map`` - the per-point ``alpha`` field (a range/reliability
  signal, not transparency) scattered into the image, Gaussian-smoothed,
  normalised to [0, 1] and clipped to the ``fov_mask``.

Both are computed the same way for every frame from the world-frame cloud and
the frame's world-from-camera pose; see ``output.py`` / ``docs``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from .camera import PolynomialCamera
from .visualize import project_world_points

# Gaussian smoothing sigma (px) used to spread sparse alpha into a dense field.
DEFAULT_SIGMA = 40.0


def _project_valid(
    points, T_cam_world, camera: PolynomialCamera, width, height, max_depth
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame points; return fractional camera pixel coords + points.

    Uses the exact same projection as the preview/overlay renderer
    (``visualize.project_world_points``), additionally cropped to the image
    bounds.
    """
    proj = project_world_points(points, T_cam_world, camera, max_depth)
    if proj.u.size == 0:
        return np.empty((0, 2)), proj.points
    within = (proj.u >= 0) & (proj.u < width) & (proj.v >= 0) & (proj.v < height)
    uv = np.stack([proj.u[within], proj.v[within]], axis=1)
    return uv, proj.points[within]


def fov_mask(
    points,
    T_cam_world,
    camera: PolynomialCamera,
    width: int,
    height: int,
    max_depth: float = 200.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """LiDAR coverage region as a boolean (H,W) mask.

    Returns ``(mask, uv)`` where ``uv`` are the projected pixel coords used to
    build the hull (set to None if not enough points).
    """
    uv, _ = _project_valid(points, T_cam_world, camera, width, height, max_depth)
    if uv.shape[0] < 3:
        return np.zeros((height, width), dtype=bool), None

    hull = ConvexHull(np.stack([uv[:, 0], uv[:, 1]], axis=1))
    poly = list(zip(uv[hull.vertices, 0], uv[hull.vertices, 1], strict=True))
    poly = [(float(x), float(y)) for x, y in poly]
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return np.asarray(img) > 0, uv


def confidence_map(
    points,
    T_cam_world,
    camera: PolynomialCamera,
    width: int,
    height: int,
    mask,
    max_depth: float = 200.0,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-derived confidence field clipped to ``mask``.

    Returns ``(conf, uv)``.  ``conf`` is float32 [0,1], 0 outside ``mask``.
    """
    uv, pts = _project_valid(points, T_cam_world, camera, width, height, max_depth)
    if uv.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32), uv

    field = np.zeros((height, width), dtype=np.float32)
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, height - 1)
    field[v, u] = pts[:, 6] / 255.0

    if sigma > 0:
        field = ndimage.gaussian_filter(field, sigma)

    m = float(field.max())
    if m > 0:
        field /= m
    if mask is not None:
        field = field * mask
    return field.astype(np.float32), uv
