"""Streaming readers for the Odin1 binary data streams.

All multi-byte values are little-endian.  Each stream is a concatenation of
fixed-size (or length-prefixed) frames.  The layouts below were extracted
from the authoritative ``BinaryDataLogger`` + ``host_sdk_sample`` sources in
``odin_ros_driver`` (see ``docs/binary-format.md`` for details).

Image stream (``OdinImage.bin``) - variable length frame::

    offset  0  uint32   frame_index
    offset  4  float64  timestamp_sec
    offset 12  uint32   jpeg_size
    offset 16  <jpeg_size bytes of JPEG>

Pose stream (``OdinPose.bin``) - fixed 40-byte frame::

    uint32 frame_index, float64 timestamp_sec,
    float32 x, y, z, qx, qy, qz, qw       # world_from_imu, pos in m, quat

Rotate stream (``OdinRotate.bin``) - fixed 28-byte frame::

    uint32 frame_index, float64 timestamp_sec,
    float32 qx, qy, qz, qw                # WIWC orientation only

IMU stream (``OdinIMU.bin``) - fixed 32-byte frame::

    float64 timestamp_sec, float32 ax, ay, az, wx, wy, wz

SLAM cloud stream (``MT<run>.olx``) - variable length frame::

    uint32 frame_index, float64 timestamp_sec, uint32 num_points,
    then num_points * 16 bytes: (float32 x, y, z) + (uint8 r, g, b, a)

The cloud points are already expressed in the SLAM odometry (world) frame.

Two reader shapes are offered on purpose:

- the **streaming** generators (``read_*_stream``) walk a file once and are
  the tool for extension passes that stay inside a bounded memory budget;
- the **random-access** indexes (``build_*_toc`` + ``*Index``) build a
  header-only table once and then serve individual frames by seek.

Both are built on :func:`iter_frame_spans`, so the header layout and the
"ignore a truncated trailing frame" rule live in exactly one place.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

# ---------------------------------------------------------------------------
# Header layouts
# ---------------------------------------------------------------------------
_IMAGE_HDR = struct.Struct("<IdI")  # idx, ts, jpeg_size
_CLOUD_HDR = struct.Struct("<IdI")  # idx, ts, num_points
_POSE_HDR = struct.Struct("<Id7f")  # idx, ts, x y z qx qy qz qw
_ROTATE_HDR = struct.Struct("<Id4f")  # idx, ts, qx qy qz qw
_IMU_HDR = struct.Struct("<d6f")  # ts, ax ay az wx wy wz

IMAGE_HDR_SIZE = _IMAGE_HDR.size  # 16
CLOUD_HDR_SIZE = _CLOUD_HDR.size  # 16
POSE_HDR_SIZE = _POSE_HDR.size  # 40
ROTATE_HDR_SIZE = _ROTATE_HDR.size  # 28
IMU_HDR_SIZE = _IMU_HDR.size  # 32

CLOUD_POINT_SIZE = 16  # 3 * float32 + 4 * uint8

# np structured dtype describing one raw 16-byte cloud point.
CLOUD_POINT_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgba", "u1", (4,))])


@dataclass(frozen=True)
class ImageFrame:
    """One JPEG image frame."""

    index: int
    timestamp: float
    jpeg: bytes


@dataclass(frozen=True)
class CloudFrame:
    """One SLAM point-cloud frame.

    ``points`` has shape (N, 7), float32: x, y, z, r, g, b, a with colors
    in [0, 255].  Points are already in the world (SLAM odom) frame.
    """

    index: int
    timestamp: float
    points: np.ndarray


@dataclass(frozen=True)
class PoseFrame:
    """One odometry pose: world_from_imu position + unit quaternion."""

    index: int
    timestamp: float
    position: np.ndarray  # (3,) float64
    quaternion: np.ndarray  # (4,) float64 x,y,z,w


@dataclass(frozen=True)
class RotateFrame:
    """One WIWC orientation sample (rotation only)."""

    index: int
    timestamp: float
    quaternion: np.ndarray  # (4,) float64 x,y,z,w


@dataclass(frozen=True)
class ImuFrame:
    """One IMU sample.  Acceleration in m/s^2, angular velocity in rad/s."""

    timestamp: float
    acceleration: np.ndarray  # (3,) float32 (ax, ay, az)
    angular_velocity: np.ndarray  # (3,) float32 (wx, wy, wz)


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------
def _open(path) -> BinaryIO:
    return open(path, "rb")


# ---------------------------------------------------------------------------
# Frame framing (the single header-scanning seam)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameSpan:
    """One variable-length frame's header fields and payload location."""

    index: int
    timestamp: float
    count: int  # payload units: jpeg bytes (image) or number of points (cloud)
    offset: int  # byte offset of the payload
    length: int  # payload length in bytes


def iter_frame_spans(
    path: str | Path, hdr: struct.Struct, unit_size: int
) -> Iterator[FrameSpan]:
    """Walk a header-prefixed stream without loading payloads.

    ``hdr`` unpacks to ``(index, timestamp, count)``; the payload is
    ``count * unit_size`` bytes.  A truncated trailing frame is skipped, so
    every yielded span is fully present in the file.
    """
    total = Path(path).stat().st_size
    with _open(path) as f:
        while True:
            raw = f.read(hdr.size)
            if len(raw) < hdr.size:
                break
            index, ts, count = hdr.unpack(raw)
            offset = f.tell()
            length = count * unit_size
            if offset + length > total:
                break  # truncated final frame
            f.seek(offset + length)
            yield FrameSpan(index, ts, count, offset, length)


def _read_span_payload(f: BinaryIO, span: FrameSpan) -> bytes:
    f.seek(span.offset)
    return f.read(span.length)


def _decode_cloud_payload(raw: bytes, count: int) -> np.ndarray:
    """Decode raw cloud bytes into an (count, 7) float32 array."""
    rec = np.frombuffer(raw, dtype=CLOUD_POINT_DTYPE)
    points = np.empty((count, 7), dtype=np.float32)
    points[:, 0:3] = rec["xyz"]
    points[:, 3:7] = rec["rgba"].astype(np.float32)
    return points


# ---------------------------------------------------------------------------
# Streaming readers (one pass, bounded memory)
# ---------------------------------------------------------------------------
def read_image_stream(path: str | Path) -> Iterator[ImageFrame]:
    """Yield image frames without buffering the whole file.

    The JPEG payload of each frame is returned as bytes; decode with the
    standard library ``jpeg`` codec of your choice (``imageio``, ``PIL``,
    ``cv2``, ...).
    """
    with _open(path) as f:
        for span in iter_frame_spans(path, _IMAGE_HDR, 1):
            yield ImageFrame(span.index, span.timestamp, _read_span_payload(f, span))


def read_cloud_stream(path: str | Path) -> Iterator[CloudFrame]:
    """Yield SLAM cloud frames as (N, 7) float32 arrays."""
    with _open(path) as f:
        for span in iter_frame_spans(path, _CLOUD_HDR, CLOUD_POINT_SIZE):
            raw = _read_span_payload(f, span)
            yield CloudFrame(
                span.index, span.timestamp, _decode_cloud_payload(raw, span.count)
            )


def read_pose_stream(path: str | Path) -> Iterator[PoseFrame]:
    """Yield odometry poses (40-byte fixed frames)."""
    with _open(path) as f:
        while True:
            raw = f.read(_POSE_HDR.size)
            if len(raw) < _POSE_HDR.size:
                break
            index, ts, x, y, z, qx, qy, qz, qw = _POSE_HDR.unpack(raw)
            yield PoseFrame(
                index,
                ts,
                np.array([x, y, z], dtype=np.float64),
                np.array([qx, qy, qz, qw], dtype=np.float64),
            )


def read_rotate_stream(path: str | Path) -> Iterator[RotateFrame]:
    """Yield WIWC orientation samples (28-byte fixed frames)."""
    with _open(path) as f:
        while True:
            raw = f.read(_ROTATE_HDR.size)
            if len(raw) < _ROTATE_HDR.size:
                break
            index, ts, qx, qy, qz, qw = _ROTATE_HDR.unpack(raw)
            yield RotateFrame(index, ts, np.array([qx, qy, qz, qw], dtype=np.float64))


def read_imu_stream(path: str | Path) -> Iterator[ImuFrame]:
    """Yield IMU samples (32-byte fixed frames)."""
    with _open(path) as f:
        while True:
            raw = f.read(_IMU_HDR.size)
            if len(raw) < _IMU_HDR.size:
                break
            ts, ax, ay, az, wx, wy, wz = _IMU_HDR.unpack(raw)
            yield ImuFrame(
                ts,
                np.array([ax, ay, az], dtype=np.float32),
                np.array([wx, wy, wz], dtype=np.float32),
            )


def image_timestamps(path: str | Path) -> list[tuple[int, float]]:
    """[(index, timestamp), ...] for the image stream (payload skipped)."""
    return [(s.index, s.timestamp) for s in iter_frame_spans(path, _IMAGE_HDR, 1)]


def cloud_timestamps(path: str | Path) -> list[tuple[int, float]]:
    """[(index, timestamp), ...] for the cloud stream (payload skipped)."""
    return [
        (s.index, s.timestamp)
        for s in iter_frame_spans(path, _CLOUD_HDR, CLOUD_POINT_SIZE)
    ]


# ---------------------------------------------------------------------------
# Random-access indexes
#
# A StreamToc lists every frame's (index, timestamp, byte offset, byte length,
# count).  Building it is a single header-only pass; the same TOC is then used
# for time alignment AND for random access, so a stream is never scanned twice.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamToc:
    """One frame's record in a stream index."""

    index: int
    timestamp: float
    data_offset: int
    byte_length: int
    count: int  # payload units: jpeg bytes (image) or number of points (cloud)


def build_image_toc(path: str | Path) -> list[StreamToc]:
    """Header-only TOC for the image stream."""
    return [
        StreamToc(s.index, s.timestamp, s.offset, s.length, s.count)
        for s in iter_frame_spans(path, _IMAGE_HDR, 1)
    ]


def build_cloud_toc(path: str | Path) -> list[StreamToc]:
    """Header-only TOC for the SLAM cloud stream."""
    return [
        StreamToc(s.index, s.timestamp, s.offset, s.length, s.count)
        for s in iter_frame_spans(path, _CLOUD_HDR, CLOUD_POINT_SIZE)
    ]


class StreamIndex:
    """Random-access reader over a stream TOC.

    ``read_bytes(index)`` returns the raw payload for a frame via one seek.
    Subclasses decode the payload into frame objects / arrays.
    """

    def __init__(self, path: str | Path, toc: list[StreamToc]):
        self._f = open(path, "rb")  # noqa: SIM115 - handle held by the reader
        self.toc = toc
        self._by_index = {t.index: t for t in toc}

    def entry(self, index: int) -> StreamToc:
        e = self._by_index.get(index)
        if e is None:
            raise IndexError(f"frame {index} not found")
        return e

    def read_bytes(self, index: int) -> bytes:
        e = self.entry(index)
        self._f.seek(e.data_offset)
        return self._f.read(e.byte_length)

    def close(self):
        self._f.close()


class ImageIndex(StreamIndex):
    """Random-access reader for the image stream."""

    def __init__(self, path: str | Path, toc: list[StreamToc] | None = None):
        super().__init__(path, toc if toc is not None else build_image_toc(path))

    def read(self, index: int) -> tuple[int, float, bytes]:
        e = self.entry(index)
        return e.index, e.timestamp, self.read_bytes(index)


class CloudIndex(StreamIndex):
    """Random-access reader for the SLAM cloud stream."""

    def __init__(self, path: str | Path, toc: list[StreamToc] | None = None):
        super().__init__(path, toc if toc is not None else build_cloud_toc(path))

    def read(self, index: int) -> np.ndarray:
        e = self.entry(index)
        return _decode_cloud_payload(self.read_bytes(index), e.count)


class PoseIndex(StreamIndex):
    """Random-access reader for the fixed-size OdinPose.bin stream.

    Pose frames are fixed-size, so the index is implicit (frame k at
    ``k * POSE_HDR_SIZE``) and no header scan is needed.
    """

    def __init__(self, path: str | Path):
        self._path = path
        self._f = open(path, "rb")  # noqa: SIM115 - handle held by the reader
        self.count = Path(path).stat().st_size // POSE_HDR_SIZE

    def read(self, index: int) -> PoseFrame:
        if index < 0 or index >= self.count:
            raise IndexError(f"pose index {index} out of range (0..{self.count - 1})")
        self._f.seek(index * POSE_HDR_SIZE)
        raw = self._f.read(POSE_HDR_SIZE)
        idx, ts, x, y, z, qx, qy, qz, qw = _POSE_HDR.unpack(raw)
        if idx != index:
            raise ValueError(f"pose index mismatch: {idx} != {index}")
        return PoseFrame(
            index,
            ts,
            np.array([x, y, z], dtype=np.float64),
            np.array([qx, qy, qz, qw], dtype=np.float64),
        )

    def read_pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position, quaternion)`` for an integer frame index."""
        f = self.read(index)
        return f.position, f.quaternion

    def close(self):
        self._f.close()
