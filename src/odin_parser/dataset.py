"""Read a PandaSet-style dataset produced by ``odin_parser.process``.

Provides convenient access to frames for downstream use (visualisation,
gsplat ingestion, camera/LiDAR consistency checks).
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    """One synchronized frame (image + cloud + poses)."""

    index: int
    timestamp: float
    image_path: Path
    cloud_path: Path
    camera_pose: np.ndarray  # 4x4 world_from_camera
    lidar_pose: np.ndarray  # 4x4 world_from_lidar


class OdinDataset:
    """Read a processed Odin dataset directory tree."""

    def __init__(self, root: str | Path, camera_name: str = "front_camera"):
        self.root = Path(root)
        self.camera_name = camera_name
        cam_dir = self.root / "camera" / camera_name
        lidar_dir = self.root / "lidar"
        if not cam_dir.is_dir() or not lidar_dir.is_dir():
            raise FileNotFoundError(f"not a processed Odin dataset: {root}")
        self.cam_dir = cam_dir
        self.lidar_dir = lidar_dir
        self.intrinsics = self._load_json(cam_dir / "intrinsics.json")
        self.camera_poses = self._load_json(cam_dir / "poses.json")
        self.lidar_poses = self._load_json(lidar_dir / "poses.json")
        self.tcl_frames = self._load_json(cam_dir / "tcl_frames.json")
        self.info = self._load_json(self.root / "dataset_info.json") or {}

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def __len__(self) -> int:
        return len(self.camera_poses)

    def camera_pose(self, frame: int) -> np.ndarray:
        return np.asarray(self.camera_poses[frame]["matrix"], dtype=np.float64)

    def lidar_pose(self, frame: int) -> np.ndarray:
        return np.asarray(self.lidar_poses[frame]["matrix"], dtype=np.float64)

    def image(self, frame: int):
        """Return decoded RGB image (PIL Image)."""
        from PIL import Image

        path = self.cam_dir / f"{frame:05d}.jpg"
        return Image.open(path).convert("RGB")

    def image_path(self, frame: int) -> Path:
        return self.cam_dir / f"{frame:05d}.jpg"

    def point_cloud(self, frame: int) -> np.ndarray:
        """Load (N,7) float32 cloud: x,y,z,r,g,b,a."""
        path = self.lidar_dir / f"{frame:05d}.pkl.gz"
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def mask(self, frame: int) -> np.ndarray:
        """Load the per-frame LiDAR/RGB validity mask (bool HxW)."""
        from PIL import Image

        return np.asarray(Image.open(self.mask_path(frame)).convert("L")) > 0

    def confidence(self, frame: int) -> np.ndarray:
        """Load the per-frame alpha confidence map (float32 HxW in [0,1])."""
        return np.load(self.confidence_path(frame))

    def mask_path(self, frame: int) -> Path:
        return self.cam_dir / "masks" / f"{frame:05d}.png"

    def confidence_path(self, frame: int) -> Path:
        return self.cam_dir / "confidence" / f"{frame:05d}.npy"

    def frames(self) -> Iterator[FrameRecord]:
        """Yield one FrameRecord per synchronized frame."""
        for i, pose in enumerate(self.camera_poses):
            yield FrameRecord(
                index=i,
                timestamp=float(pose["timestamp"]),
                image_path=self.image_path(i),
                cloud_path=self.lidar_dir / f"{i:05d}.pkl.gz",
                camera_pose=self.camera_pose(i),
                lidar_pose=self.lidar_pose(i),
            )

    def __repr__(self) -> str:
        return (
            f"OdinDataset(root={self.root}, frames={len(self)}, "
            f"camera={self.camera_name})"
        )
