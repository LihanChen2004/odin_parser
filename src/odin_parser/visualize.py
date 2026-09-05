"""Visualisation helpers: project point clouds onto camera images.

All renderers draw every in-view LiDAR point as a **semi-transparent round
dot** so the underlying camera image stays readable (the points neither vanish
nor overpower the scene).  Defaults: ``radius=2``, ``alpha=0.7``,
``darken=0.85``.

Two modes are provided:

- ``render_overlay`` - dot colour = the point's recorded RGB.
- ``render_depth``  - dot colour = depth (jet colormap).

All projections use the polynomial equidistant camera model
(``camera.PolynomialCamera``) so that points land on the raw distorted JPEG.

``render_overlay`` is used both by the ``odin-parser preview`` command and by
``scripts/make_video.py``, so a single preview frame and a video frame are
visually identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PolynomialCamera
from .geometry import invert, transform_points

# Default visual style shared by the preview command and the video renderer.
DEFAULT_RADIUS = 2
DEFAULT_ALPHA = 0.7
DEFAULT_DARKEN = 0.85


@dataclass
class Projection:
    """Projection result."""

    u: np.ndarray  # (M,) float pixel columns
    v: np.ndarray  # (M,) float pixel rows
    depth: np.ndarray  # (M,) depth in metres
    colors: np.ndarray  # (M, 3) uint8 in [0,255]
    points: np.ndarray | None = None  # (M, F) surviving source points


def project_world_points(
    points_world: np.ndarray,
    T_cam_world: np.ndarray,
    camera: PolynomialCamera,
    max_depth: float = 200.0,
) -> Projection:
    """Project world-frame points into the raw camera image.

    ``T_cam_world`` maps camera frame -> world frame.  Points behind the
    camera or with invalid depth are dropped.
    """
    pts = np.asarray(points_world[:, 0:3], dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1)
    pc = transform_points(invert(T_cam_world), pts[valid])
    ok = camera.valid_mask(pc, max_depth=max_depth)
    pc = pc[ok]
    uv = camera.world_to_pixel(pc)
    colors = np.asarray(points_world[valid][:, 3:6][ok], dtype=np.float64)
    src = np.asarray(points_world, dtype=np.float64)[valid][ok]
    return Projection(
        u=uv[:, 0],
        v=uv[:, 1],
        depth=pc[:, 2],
        colors=colors,
        points=src,
    )


def _circle_offsets(radius: int) -> list[tuple[int, int]]:
    """All (du, dv) integer offsets inside a disc of the given radius."""
    rng = range(-radius, radius + 1)
    return [
        (du, dv) for du in rng for dv in rng if du * du + dv * dv <= radius * radius
    ]


def _draw_points(
    image_rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    colors: np.ndarray,
    radius: int = DEFAULT_RADIUS,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Blend round semi-transparent dots onto a float background.

    ``image_rgb`` is a float HxWx3 background; ``u``/``v``/``colors`` are the
    per-point pixels and RGB colours.  Points are drawn as filled discs with
    opacity ``alpha`` via simple alpha compositing.
    """
    bg = np.asarray(image_rgb, dtype=np.float64).copy()
    h, w = bg.shape[:2]
    if u.size == 0:
        return bg
    u = np.clip(np.round(u).astype(np.int32), 0, w - 1)
    v = np.clip(np.round(v).astype(np.int32), 0, h - 1)
    for du, dv in _circle_offsets(radius):
        uu = np.clip(u + du, 0, w - 1)
        vv = np.clip(v + dv, 0, h - 1)
        bg[vv, uu] = bg[vv, uu] * (1.0 - alpha) + colors * alpha
    return np.round(bg).astype(np.uint8)


def _compose(
    image_rgb: np.ndarray,
    proj: Projection,
    radius: int,
    alpha: float,
    darken: float,
) -> np.ndarray:
    if proj.u.size == 0:
        return image_rgb
    bg = np.asarray(image_rgb, dtype=np.float64) * darken
    return _draw_points(bg, proj.u, proj.v, proj.colors, radius, alpha)


def render_overlay(
    image_rgb: np.ndarray,
    points_world: np.ndarray,
    T_cam_world: np.ndarray,
    camera: PolynomialCamera,
    radius: int = DEFAULT_RADIUS,
    alpha: float = DEFAULT_ALPHA,
    darken: float = DEFAULT_DARKEN,
    max_depth: float = 200.0,
) -> np.ndarray:
    """Return an image with LiDAR points (coloured by their RGB) overlaid.

    ``darken`` dims the background (0.85 default); ``alpha`` (0.7 default) is
    the dot opacity.  Set ``alpha=1`` for opaque dots.
    """
    proj = project_world_points(points_world, T_cam_world, camera, max_depth)
    return _compose(image_rgb, proj, radius, alpha, darken)


def render_depth(
    image_rgb: np.ndarray,
    points_world: np.ndarray,
    T_cam_world: np.ndarray,
    camera: PolynomialCamera,
    radius: int = DEFAULT_RADIUS,
    alpha: float = DEFAULT_ALPHA,
    darken: float = DEFAULT_DARKEN,
    max_depth: float = 200.0,
) -> np.ndarray:
    """Return an image with LiDAR points coloured by depth (jet).

    Depth is normalised against ``max_depth`` when it exceeds a small floor,
    otherwise against the scene's own 99th percentile so close-range captures
    still show a useful colour gradient.
    """
    proj = project_world_points(points_world, T_cam_world, camera, max_depth)
    d = proj.depth
    if d.size == 0:
        return image_rgb
    scale = (
        max_depth if np.median(d) > max_depth * 0.5 else max(5.0, np.percentile(d, 99))
    )
    t = np.clip(d / scale, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
    colors = np.stack([r, g, b], axis=1) * 255.0
    proj = Projection(u=proj.u, v=proj.v, depth=d, colors=colors)
    return _compose(image_rgb, proj, radius, alpha, darken)
