"""Unit tests for the Odin1 binary stream readers."""

import struct

import numpy as np

from odin_parser.streams import (
    CLOUD_POINT_SIZE,
    read_cloud_stream,
    read_image_stream,
    read_imu_stream,
    read_pose_stream,
    read_rotate_stream,
)


def _write(path, chunks: list[bytes]) -> str:
    with open(path, "wb") as f:
        f.writelines(chunks)
    return path


def test_image_stream(tmp_path):
    p = tmp_path / "OdinImage.bin"
    jpgs = [b"\xff\xd8JPEG%d" % i + b"\x00" * 20 for i in range(3)]
    chunks = []
    for i, j in enumerate(jpgs):
        chunks.append(struct.pack("<IdI", i, 100.0 + i, len(j)))
        chunks.append(j)
    _write(str(p), chunks)

    frames = list(read_image_stream(p))
    assert len(frames) == 3
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp == 101.0
    assert frames[2].jpeg == jpgs[2]


def test_cloud_stream(tmp_path):
    p = tmp_path / "MT20260424_231347.olx"
    # two points for frame 0, one for frame 1
    pt = struct.pack("<fffBBBB", 1.0, 2.0, 3.0, 10, 20, 30, 40)
    pt2 = struct.pack("<fffBBBB", 4.0, 5.0, 6.0, 11, 21, 31, 41)
    frame0 = struct.pack("<IdI", 0, 147.0, 2) + pt + pt2
    frame1 = struct.pack("<IdI", 1, 147.1, 1) + pt
    _write(str(p), [frame0, frame1])

    frames = list(read_cloud_stream(p))
    assert len(frames) == 2
    f0 = frames[0]
    assert f0.index == 0 and f0.timestamp == 147.0
    assert f0.points.shape == (2, 7)
    np.testing.assert_allclose(f0.points[0, :3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(f0.points[0, 3:], [10.0, 20.0, 30.0, 40.0])
    assert frames[1].points.shape == (1, 7)


def test_pose_stream(tmp_path):
    p = tmp_path / "OdinPose.bin"
    raw = struct.pack(
        "<Id7f",
        0,
        147.997862,
        0.0,
        0.0,
        -0.0003,
        -0.017416,
        -0.000580,
        0.000023,
        0.999848,
    )
    raw += struct.pack("<Id7f", 1, 148.095338, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
    _write(str(p), [raw])
    frames = list(read_pose_stream(p))
    assert len(frames) == 2
    assert frames[0].position.shape == (3,)
    assert frames[0].quaternion.shape == (4,)
    np.testing.assert_allclose(frames[1].position, [0.1, 0.2, 0.3])
    assert frames[1].timestamp == 148.095338


def test_rotate_stream(tmp_path):
    p = tmp_path / "OdinRotate.bin"
    _write(str(p), [struct.pack("<Id4f", 3, 148.0, 0.1, 0.2, 0.3, 0.9)])
    (f,) = list(read_rotate_stream(p))
    assert f.index == 3
    np.testing.assert_allclose(f.quaternion, [0.1, 0.2, 0.3, 0.9])


def test_imu_stream(tmp_path):
    p = tmp_path / "OdinIMU.bin"
    _write(str(p), [struct.pack("<d6f", 148.0, 1, 2, 3, 4, 5, 6)])
    (f,) = list(read_imu_stream(p))
    assert f.timestamp == 148.0
    np.testing.assert_allclose(f.acceleration, [1, 2, 3])
    np.testing.assert_allclose(f.angular_velocity, [4, 5, 6])


def test_truncated_tail_ignored(tmp_path):
    p = tmp_path / "OdinPose.bin"
    full = struct.pack("<Id7f", 0, 1.0, 0, 0, 0, 0, 0, 0, 1)
    _write(str(p), [full, full[:10]])  # partial trailing frame
    assert len(list(read_pose_stream(p))) == 1


def test_cloud_point_size_consistency():
    assert CLOUD_POINT_SIZE == 3 * 4 + 4 * 1 == 16
