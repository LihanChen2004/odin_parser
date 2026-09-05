"""Polynomial (equidistant) camera projection model.

Mirrors ``PolynomialCamera`` in the Odin1 ROS driver
(``include/polynomial_camera.hpp``), which implements the pinhole + odd-order
polynomial equidistant distortion used by the device::

    theta_d = theta + k2*theta^2 + k3*theta^3 + ... + k7*theta^7

where ``theta`` is the angle between a 3D ray and the optical axis and the
distorted ray direction is ``theta_d``.  ``k1`` is not reported by the
device and treated as zero.

Projection of a camera-frame 3D point ``p = (X, Y, Z)``::

    r   = sqrt(X^2 + Y^2)
    th  = atan2(r, Z)
    thd = theta_d(th)
    u   = fx * X * thd/r + skew * Y * thd/r + cx
    v   = fy * Y * thd/r + cy

The pinhole (undistorted) model is the special case ``thd = tan(th)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calib import Intrinsics


@dataclass(frozen=True)
class PolynomialCamera:
    """Project points from the camera frame to image pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    k: tuple = (0.0,) * 6

    @classmethod
    def from_intrinsics(cls, intr: Intrinsics) -> PolynomialCamera:
        return cls(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
            skew=intr.skew,
            k=intr.k,
        )

    def theta_distorted(self, theta: np.ndarray) -> np.ndarray:
        """Apply the polynomial distortion to polar angles."""
        out = np.zeros_like(theta, dtype=np.float64)
        out += theta
        t = theta
        for coef in self.k:
            t = t * theta
            out = out + coef * t
        return out

    def world_to_pixel(self, points_cam: np.ndarray) -> np.ndarray:
        """(N,3) points in the camera frame -> (N,2) pixel coordinates.

        The returned pixels refer to the raw (distorted) image.
        """
        p = np.asarray(points_cam, dtype=np.float64)
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(X * X + Y * Y)
        theta = np.arctan2(r, np.maximum(Z, 1e-12))
        theta_d = self.theta_distorted(theta)
        scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-12), 0.0)
        xd = X * scale
        yd = Y * scale
        u = xd * self.fx + yd * self.skew + self.cx
        v = yd * self.fy + self.cy
        return np.stack([u, v], axis=1)

    def world_to_pixel_pinhole(self, points_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame points -> pixels on the undistorted image."""
        p = np.asarray(points_cam, dtype=np.float64)
        z = np.maximum(p[:, 2], 1e-12)
        u = self.fx * p[:, 0] / z + self.skew * p[:, 1] / z + self.cx
        v = self.fy * p[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def valid_mask(
        self,
        points_cam: np.ndarray,
        max_depth: float = 200.0,
    ) -> np.ndarray:
        """Points in front of the camera with finite, sane depth."""
        p = np.asarray(points_cam, dtype=np.float64)
        return np.isfinite(p).all(axis=1) & (p[:, 2] > 0.02) & (p[:, 2] < max_depth)
