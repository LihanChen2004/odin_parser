"""Unit tests for the pose time-offset estimator and pose interpolation."""

import math

import numpy as np

from odin_parser.camera import PolynomialCamera
from odin_parser.geometry import interpolate_pose, matrix_to_pose, pose_to_matrix
from odin_parser.sync import estimate_pose_offset

CAM = PolynomialCamera(fx=500.0, fy=500.0, cx=100.0, cy=100.0, k=(0.0,) * 6)
# Off-axis world point so pose rotation moves its projection on the image.
WORLD = np.array([0.5, 0.0, 3.0])
COLOR = np.array([200, 100, 50])
OTHER = np.array([0, 0, 0])
IM_W, IM_H = 200, 200


def _yaw(deg):
    a = math.radians(deg)
    return np.array([0.0, 0.0, math.sin(a / 2.0), math.cos(a / 2.0)])


def _pose(i):
    return pose_to_matrix([0.0, 0.0, 0.0], _yaw(3.0 * i))


def _project_pixel(mat):
    from odin_parser.geometry import invert, transform_points

    # camera-frame point is inv(pose) @ WORLD (matches how the scorer projects)
    uv = CAM.world_to_pixel(transform_points(invert(mat), WORLD[None, :]))
    return round(uv[0, 0]), round(uv[0, 1])


def test_geometry_interpolate_poses():
    p0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    p1 = np.array([1.0, 0.0, 0.0])
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    pos, q = interpolate_pose(p0, q0, p1, q1, 0.5)
    np.testing.assert_allclose(pos, [0.5, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_estimate_pose_offset_recovers_known_offset():
    # The pose that truly aligns frame k is pose(k+1) (odometry lags by 1 frame).
    # For each frame k the *image* has the cloud colour only at the pixel where
    # pose(k+1) projects the off-axis WORLD point; everywhere else it is black.
    # The estimator must find the offset that reproduces pose(k+1), i.e. +1 frame.
    n = 30

    def build_image(k):
        img = np.zeros((IM_H, IM_W, 3), dtype=np.uint8)
        img[:, :] = OTHER
        u, v = _project_pixel(_pose(k + 1))
        if 0 <= u < IM_W and 0 <= v < IM_H:
            img[v, u] = COLOR
        return img

    images = {k: build_image(k) for k in range(n)}

    def read_pose(i):
        m = _pose(round(i))
        return m[:3, 3].copy(), matrix_to_pose(m)[1]

    def cloud():
        pts = np.zeros((200, 7), dtype=np.float32)
        pts[:, 0:3] = WORLD
        pts[:, 3:6] = COLOR
        return pts

    est = estimate_pose_offset(
        read_pose=read_pose,
        read_image=lambda i: images[i],
        read_cloud=lambda i: cloud(),
        read_calib=lambda i: (np.eye(4), np.eye(4)),
        camera=CAM,
        indices=list(range(4, n - 4)),
        lo=-2.0,
        hi=2.0,
        step=0.5,
    )
    # pose(k+1) uses yaw 3*(k+1); the estimator should land near +1 frame.
    assert est.offset_frames > 0.3
    assert est.samples == len(list(range(4, n - 4)))
