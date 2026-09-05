"""Small SE(3) / quaternion helpers.

Conventions used across the package
----------------------------------
- Rotation/transformation matrices are 4x4 homogeneous, stored as numpy
  float64, row-major when serialised to JSON.
- A transform matrix ``M = T_a_b`` maps a point expressed in frame ``b`` to a
  point expressed in frame ``a``::

      p_a = M @ [p_b; 1]

  Therefore the matrix of a pose "in frame a" is ``T_a_self`` and ``M @
  T_a_b == T_a_c`` when ``b -> c`` is another frame in between.
- Quaternions are stored as ``(x, y, z, w)``.
"""

from __future__ import annotations

import numpy as np

Quat = np.ndarray
Vec3 = np.ndarray
Mat4 = np.ndarray


def quat_to_rot(q: Quat) -> np.ndarray:
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> Quat:
    """3x3 rotation matrix -> unit quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return q


def pose_to_matrix(position: Vec3, quaternion: Quat) -> Mat4:
    """(position, quaternion) -> 4x4 pose matrix."""
    m = np.eye(4)
    m[:3, :3] = quat_to_rot(quaternion)
    m[:3, 3] = np.asarray(position, dtype=np.float64)
    return m


def matrix_to_pose(T: Mat4) -> tuple[Vec3, Quat]:
    """4x4 pose matrix -> (position, quaternion)."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), rot_to_quat(T[:3, :3])


def invert(T: Mat4) -> Mat4:
    """Inverse of a 4x4 rigid transform (closed-form, no linear solve)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(T_a_b: Mat4, T_b_c: Mat4) -> Mat4:
    """Compose transforms: returns ``T_a_c = T_a_b @ T_b_c``."""
    return np.asarray(T_a_b, dtype=np.float64) @ np.asarray(T_b_c, dtype=np.float64)


def transform_points(T: Mat4, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N,3) point array -> (N,3)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return np.asarray(points, dtype=np.float64) @ R.T + t


def quat_slerp(qa: Quat, qb: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two unit quaternions."""
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sa = np.sin((1.0 - t) * theta)
    sb = np.sin(t * theta)
    return (sa * qa + sb * qb) / np.sin(theta)


def interpolate_pose(
    pos0: Vec3, quat0: Quat, pos1: Vec3, quat1: Quat, t: float
) -> tuple[Vec3, Quat]:
    """Linearly interpolate position and slerp the quaternion for ``t`` in [0,1]."""
    pos = np.asarray(pos0, dtype=np.float64) + t * (
        np.asarray(pos1, dtype=np.float64) - np.asarray(pos0, dtype=np.float64)
    )
    return pos, quat_slerp(quat0, quat1, t)


def matrix_to_json(T: Mat4) -> list[list[float]]:
    """4x4 matrix -> 4x4 nested list (row-major) for JSON."""
    return [[float(v) for v in row] for row in np.asarray(T, dtype=np.float64)]
