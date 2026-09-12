"""A random-access view of one Odin1 raw recording.

``OdinRecording`` binds to a recording directory and lazily builds a single
header index (TOC) per stream, which is shared by:

- cross-stream **time alignment** (``align``),
- **random access** to image / cloud / pose / calib frames
  (``image``, ``cloud``, ``pose``, ``pose_at``),
- the **pose time-offset estimator** and the writer.

Because the TOC is built once and reused, a streaming pass over the payloads is
the only time the whole file is read, and the header scans are not repeated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import streams
from .align import align_cloud_to_image
from .calib import CalibFrame, parse_info, parse_intrinsics, parse_online_calib
from .camera import PolynomialCamera
from .geometry import interpolate_pose, pose_to_matrix

# Canonical names of the files produced by the Odin1 BinaryDataLogger.
IMAGE_STREAM = "OdinImage.bin"
POSE_STREAM = "OdinPose.bin"
ROTATE_STREAM = "OdinRotate.bin"
IMU_STREAM = "OdinIMU.bin"
CLOUD_STREAM_RE = re.compile(r"^MT(?P<run>\d{8}_\d{6})\.olx$")

# The Odin1 ships a single front camera; the output sub-directory name.
DEFAULT_CAMERA_NAME = "front_camera"


@dataclass(frozen=True)
class SyncedFrame:
    """One time-synchronised frame bundle emitted by the pipeline."""

    cloud_index: int
    image_index: int
    timestamp: float
    jpeg: bytes
    points: np.ndarray  # (N,7) float32, world frame
    camera_pose: np.ndarray  # 4x4 world-from-camera
    lidar_pose: np.ndarray  # 4x4 world-from-lidar
    calib: CalibFrame


def find_streams(origin_dir: Path) -> dict[str, Path]:
    """Locate the binary streams inside a recording directory."""
    found: dict[str, Path] = {}
    for p in sorted(Path(origin_dir).rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name in (IMAGE_STREAM, POSE_STREAM, ROTATE_STREAM, IMU_STREAM):
            found[name] = p
        elif CLOUD_STREAM_RE.match(name):
            found["cloud"] = p
    return found


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    """Decode a JPEG payload to an RGB uint8 array."""
    from io import BytesIO

    from PIL import Image

    return np.asarray(Image.open(BytesIO(jpeg)).convert("RGB"))


class OdinRecording:
    """Random-access reader + alignment for a raw Odin1 recording."""

    def __init__(self, origin_dir: str | Path):
        self.root = Path(origin_dir)
        self.streams_map = find_streams(self.root)
        # Only cloud/image/pose are needed by the pipeline.  Rotate (WIWC) and
        # IMU are parsed by this module's readers but unused here, so a
        # recording that omits them stays readable.
        required = ("cloud", IMAGE_STREAM, POSE_STREAM)
        missing = [k for k in required if k not in self.streams_map]
        if missing:
            raise FileNotFoundError(f"missing streams in {self.root}: {missing}")

        self.image_path = self.streams_map[IMAGE_STREAM]
        self.pose_path = self.streams_map[POSE_STREAM]
        self.cloud_path = self.streams_map["cloud"]
        self.rotate_path = self.streams_map.get(ROTATE_STREAM)
        self.imu_path = self.streams_map.get(IMU_STREAM)
        self.calib_path = self.root / "calib_online.yaml"
        self.intr_path = self.root / "image" / "cam_in_ex.txt"
        self.info_path = self.root / "image" / "info.txt"

        # --- static calibration -------------------------------------------
        self.intrinsics = parse_intrinsics(self.intr_path)
        self.info = parse_info(self.info_path) if self.info_path.exists() else {}
        match = CLOUD_STREAM_RE.match(self.cloud_path.name)
        self.run_name = self.info.get("created_at") or (
            match.group("run") if match else "unknown"
        )

        self.online = parse_online_calib(self.calib_path)

        # --- shared, lazily-built indexes (built once each) ---------------
        self._image_toc: list[streams.StreamToc] | None = None
        self._cloud_toc: list[streams.StreamToc] | None = None
        self._pose_index: streams.PoseIndex | None = None

    # --- indexes ----------------------------------------------------------
    @property
    def image_toc(self) -> list[streams.StreamToc]:
        if self._image_toc is None:
            self._image_toc = streams.build_image_toc(self.image_path)
        return self._image_toc

    @property
    def cloud_toc(self) -> list[streams.StreamToc]:
        if self._cloud_toc is None:
            self._cloud_toc = streams.build_cloud_toc(self.cloud_path)
        return self._cloud_toc

    @property
    def n_cloud(self) -> int:
        return len(self.cloud_toc)

    @property
    def n_image(self) -> int:
        return len(self.image_toc)

    # --- timestamps -------------------------------------------------------
    def image_timestamps(self) -> list[tuple[int, float]]:
        return [(t.index, t.timestamp) for t in self.image_toc]

    def cloud_timestamps(self) -> list[tuple[int, float]]:
        return [(t.index, t.timestamp) for t in self.cloud_toc]

    # --- alignment --------------------------------------------------------
    def align(self, tolerance: float = 1e-3) -> dict[int, int]:
        """Map cloud frame index -> aligned image frame index."""
        return align_cloud_to_image(
            self.cloud_timestamps(), self.image_timestamps(), tolerance
        )

    # --- random access ----------------------------------------------------
    def image(self, index: int) -> streams.ImageFrame:
        idx, ts, jpeg = streams.ImageIndex(self.image_path, self.image_toc).read(index)
        return streams.ImageFrame(idx, ts, jpeg)

    def jpeg(self, index: int) -> bytes:
        return streams.ImageIndex(self.image_path, self.image_toc).read(index)[2]

    def cloud(self, index: int) -> np.ndarray:
        return streams.CloudIndex(self.cloud_path, self.cloud_toc).read(index)

    def _pose(self) -> streams.PoseIndex:
        if self._pose_index is None:
            self._pose_index = streams.PoseIndex(self.pose_path)
        return self._pose_index

    def pose(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(position, quaternion) at integer frame index."""
        return self._pose().read_pose(self._clamp_pose(index))

    def pose_at(self, index: float) -> np.ndarray:
        """4x4 odom->imu pose interpolated at a fractional frame index."""
        lo = int(np.floor(index))
        hi = int(np.ceil(index))
        lo, hi = self._clamp_pose(lo), self._clamp_pose(hi)
        if lo == hi:
            pos, quat = self._pose().read_pose(lo)
        else:
            t = index - int(np.floor(index))
            pos, quat = interpolate_pose(
                *self._pose().read_pose(lo), *self._pose().read_pose(hi), t
            )
        return pose_to_matrix(pos, quat)

    def _clamp_pose(self, index: int) -> int:
        return max(0, min(int(index), self.n_poses - 1))

    @property
    def n_poses(self) -> int:
        return self._pose().count

    def calib(self, index: int) -> CalibFrame:
        c = self.online.get(index)
        if c is None:
            raise KeyError(
                f"online calibration has no frame {index} "
                f"({self.calib_path.name} has {len(self.online)} frames)"
            )
        return c

    def camera_model(self) -> PolynomialCamera:
        i = self.intrinsics
        return PolynomialCamera(fx=i.fx, fy=i.fy, cx=i.cx, cy=i.cy, skew=i.skew, k=i.k)

    def estimate_pose_offset(
        self,
        cloud_indices,
        tolerance_seconds: float = 1e-3,
        lo: float = -2.0,
        hi: float = 2.0,
        step: float = 0.25,
    ):
        """Estimate the odometry pose time offset (frames) for the recording."""
        from .sync import estimate_pose_offset

        cam = self.camera_model()
        mapping = self.align(tolerance_seconds)

        def read_cloud(i):
            return self.cloud(i)

        def read_calib(i):
            c = self.calib(i)
            return c.Tcl_0, c.body_T_lidar

        def read_image(i):
            return decode_jpeg(self.jpeg(i))

        return estimate_pose_offset(
            read_pose=self.pose,
            read_image=lambda i: read_image(mapping[i]),
            read_cloud=read_cloud,
            read_calib=read_calib,
            camera=cam,
            indices=cloud_indices,
            lo=lo,
            hi=hi,
            step=step,
        )

    # --- poses for a frame ---------------------------------------------------
    def camera_pose(self, cloud_index: int, pose_offset: float = 0.0) -> np.ndarray:
        """T_wc = T_wi(pose at cloud_index + offset) * T_il * inv(T_cl)."""
        from .geometry import invert

        T_wi = self.pose_at(float(cloud_index) + pose_offset)
        c = self.calib(cloud_index)
        return T_wi @ c.body_T_lidar @ invert(c.Tcl_0)

    def lidar_pose(self, cloud_index: int, pose_offset: float = 0.0) -> np.ndarray:
        """T_wl = T_wi(pose at cloud_index + offset) * T_il."""
        T_wi = self.pose_at(float(cloud_index) + pose_offset)
        c = self.calib(cloud_index)
        return T_wi @ c.body_T_lidar

    # --- synced frame iteration ------------------------------------------------
    def iter_frames(self, cloud_indices, mapping, pose_offset: float = 0.0):
        """Yield ``SyncedFrame`` for each cloud index in ``cloud_indices``.

        ``mapping`` is ``{cloud_index: image_index}`` (from ``align``) and
        ``pose_offset`` shifts the odometry pose by a fraction of a frame.
        Every index in ``cloud_indices`` must be present in ``mapping``.
        """
        img_idx = streams.ImageIndex(self.image_path, self.image_toc)
        cl_idx = streams.CloudIndex(self.cloud_path, self.cloud_toc)
        try:
            for cloud_index in cloud_indices:
                image_index = mapping[cloud_index]
                calib = self.calib(cloud_index)
                points = cl_idx.read(cloud_index)
                jpeg = img_idx.read(image_index)[2]
                yield SyncedFrame(
                    cloud_index=cloud_index,
                    image_index=image_index,
                    timestamp=cl_idx.entry(cloud_index).timestamp,
                    jpeg=jpeg,
                    points=points,
                    camera_pose=self.camera_pose(cloud_index, pose_offset),
                    lidar_pose=self.lidar_pose(cloud_index, pose_offset),
                    calib=calib,
                )
        finally:
            img_idx.close()
            cl_idx.close()

    def close(self):
        if self._pose_index is not None:
            self._pose_index.close()
