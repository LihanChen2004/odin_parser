"""Shared fixtures for the odin_parser test suite."""

from __future__ import annotations

import struct
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PoseFn = Callable[[int], tuple[float, float, float, float, float, float, float]]

# Default odometry pose used by the synthetic recording: a constant
# translation with an identity orientation.  Pass a ``pose_fn`` to vary it.
DEFAULT_POSE = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)


def _make_origin(
    root: Path,
    n_cloud: int = 5,
    offset: int = 9,
    fps: float = 10.0,
    pose_fn: PoseFn | None = None,
    include_rotate_imu: bool = True,
) -> None:
    """Build a small synthetic Odin1 recording directory.

    cloud ts:      100 + i * dt
    image ts:      (100 - offset*dt) + j * dt   -> image[j] ~ cloud[j - offset]
    poses/rotate:  aligned with cloud ts

    ``pose_fn`` maps a cloud index to ``(x, y, z, qx, qy, qz, qw)`` and
    defaults to :data:`DEFAULT_POSE`.  ``include_rotate_imu`` writes the
    optional ``OdinRotate.bin`` / ``OdinIMU.bin`` streams; set it to
    ``False`` to exercise a recording that omits them.
    """
    pose_fn = pose_fn or (lambda _i: DEFAULT_POSE)
    dt = 1.0 / fps
    (root / "image").mkdir(parents=True)
    info = "device=OdinOne\npointcloud=xyzrgbi\ncreated_at=20260424_231347\n"
    (root / "image" / "info.txt").write_text(info)
    (root / "image" / "cam_in_ex.txt").write_text(
        "Tcl_0: [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "
        "0.0, 0.0, 0.0, 1.0]\n"
        "cam_0:\n  image_width: 1600\n  image_height: 1296\n"
        "  k2: 0.0\n  k3: 0.0\n  k4: 0.0\n  k5: 0.0\n  k6: 0.0\n  k7: 0.0\n"
        "  A11: 500.0\n  A12: 0.0\n  A22: 500.0\n  u0: 800.0\n  v0: 648.0\n"
    )

    # calib_online.yaml: index i matches cloud index i
    calib_blocks = []
    for i in range(n_cloud):
        ts = 100.0 + i * dt
        tcl = (
            1.0,
            0.0,
            0.0,
            0.5,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        til = (
            1.0,
            0.0,
            0.0,
            0.1,
            0.0,
            1.0,
            0.0,
            0.2,
            0.0,
            0.0,
            1.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        calib_blocks.append(
            f"frame:\n    index: {i}\n    timestamp: {ts:.10f}\n    cam_num: 1\n"
            f"    Tcl_0: [{', '.join(map(str, tcl))}]\n\n"
            f"    body_T_lidar: !!opencv-matrix\n       rows: 4\n       cols: 4\n"
            f"       dt: d\n       data: [{', '.join(map(str, til))}]\n\n"
        )
    (root / "calib_online.yaml").write_text("".join(calib_blocks))

    # pose stream (matches cloud ts)
    pose = b""
    for i in range(n_cloud):
        pose += struct.pack("<Id7f", i, 100.0 + i * dt, *pose_fn(i))
    (root / "OdinPose.bin").write_bytes(pose)

    if include_rotate_imu:
        (root / "OdinRotate.bin").write_bytes(
            b"".join(
                struct.pack("<Id4f", i, 100.0 + i * dt, 0, 0, 0, 1)
                for i in range(n_cloud)
            )
        )
        (root / "OdinIMU.bin").write_bytes(
            b"".join(
                struct.pack("<d6f", 100.0 + i * 0.0025, 0, 0, 0, 0, 0, 0)
                for i in range(40)
            )
        )

    # image stream (leads cloud by `offset` frames); real-size JPEG so that
    # projected points land inside the image (needed by pose-offset estimation)
    n_img = n_cloud + offset
    img = b""
    for j in range(n_img):
        ts = (100.0 - offset * dt) + j * dt
        arr = np.zeros((1296, 1600, 3), dtype=np.uint8)
        arr[:, :, 0] = np.arange(1600)[None, :] * 255 // 1600  # horizontal R gradient
        arr[:, :, 1] = j * 40 % 256
        arr[:, :, 2] = 128
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        payload = struct.pack("<IdI", j, ts, len(buf.getvalue())) + buf.getvalue()
        img += payload
    (root / "OdinImage.bin").write_bytes(img)

    # cloud stream: 200 points in front of the camera so that pose-offset
    # estimation can reproject them in-bounds (projection needs >=100 points).
    cloud = b""
    for i in range(n_cloud):
        pts = b"".join(
            struct.pack("<fffBBBB", 0.5 + 0.01 * j, 0.3, 8.0, 255, 0, 0, 200)
            for j in range(200)
        )
        cloud += struct.pack("<IdI", i, 100.0 + i * dt, 200) + pts
    (root / "MT20260424_231347.olx").write_bytes(cloud)


@pytest.fixture
def make_origin() -> Callable[..., None]:
    """Return the synthetic-recording builder (see :func:`_make_origin`)."""
    return _make_origin
