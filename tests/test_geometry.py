"""Unit tests for geometry helpers."""

import numpy as np

from odin_parser.geometry import (
    compose,
    invert,
    matrix_to_pose,
    pose_to_matrix,
    quat_to_rot,
    rot_to_quat,
    transform_points,
)


def test_quat_rot_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_rot(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        q2 = rot_to_quat(R)
        assert np.allclose(np.abs(q), np.abs(q2), atol=1e-9) or np.allclose(
            q, -q2, atol=1e-9
        )


def test_identity_quat():
    assert np.allclose(quat_to_rot([0, 0, 0, 1]), np.eye(3))
    assert np.allclose(rot_to_quat(np.eye(3)), [0, 0, 0, 1])


def test_pose_matrix_roundtrip():
    q = np.array([0.1, 0.2, 0.3, 0.9])
    q = q / np.linalg.norm(q)
    T = pose_to_matrix([1.0, 2.0, 3.0], q)
    pos, q2 = matrix_to_pose(T)
    assert np.allclose(pos, [1, 2, 3])
    assert np.allclose(q2, q, atol=1e-9)


def test_invert_compose():
    rng = np.random.default_rng(1)
    A = pose_to_matrix(rng.normal(size=3), rng.normal(size=4))
    B = pose_to_matrix(rng.normal(size=3), rng.normal(size=4))
    assert np.allclose(A @ invert(A), np.eye(4), atol=1e-12)
    assert np.allclose(compose(A, B), A @ B, atol=1e-12)


def test_transform_points():
    T = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = transform_points(T, pts)
    np.testing.assert_allclose(out, [[1.0, 3.0, 3.0], [0.0, 2.0, 3.0]])
