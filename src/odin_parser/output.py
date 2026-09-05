"""Write a processed (PandaSet-style) dataset to disk.

``write_dataset`` turns an :class:`~odin_parser.recording.OdinRecording` into a
processed tree.  The on-disk schema is the single source of truth here; see
``docs/output-dataset.md``.  ``dataset.py`` reads it back with the same field
names.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .masks import confidence_map, fov_mask
from .recording import DEFAULT_CAMERA_NAME, OdinRecording

logger = logging.getLogger(__name__)

POINT_FIELDS = ["x", "y", "z", "r", "g", "b", "a"]
COLOR_RANGE = [0, 255]


@dataclass
class ProcessSummary:
    out_dir: Path
    run_name: str
    num_frames: int
    cloud_points: int
    image_bytes: int
    dropped_frames: int
    missing_calib_frames: int
    masked_frames: int = 0


def _pose_record(index: int, timestamp: float, image_index, T: np.ndarray) -> dict:
    from .geometry import matrix_to_json, matrix_to_pose

    pos, quat = matrix_to_pose(T)
    return {
        "index": index,
        "timestamp": float(timestamp),
        "image_index": image_index,
        "position": [float(v) for v in pos],
        "quaternion": [float(v) for v in quat],
        "matrix": matrix_to_json(T),
    }


def _write_frame_payload(
    frame,
    index: int,
    cam_dir: Path,
    lidar_dir: Path,
    camera,
    width: int,
    height: int,
    generate_masks: bool,
    mask_sigma: float,
) -> tuple[int, int]:
    """Write one frame's files (image, masks, cloud); returns (img_bytes, n_pts).

    Pure w.r.t. shared state - safe to run on worker processes (the Gaussian
    smoothing holds the GIL, so threads would not parallelise it).
    """
    (cam_dir / f"{index:05d}.jpg").write_bytes(frame.jpeg)
    if generate_masks:
        mask, _ = fov_mask(frame.points, frame.camera_pose, camera, width, height)
        conf, _ = confidence_map(
            frame.points,
            frame.camera_pose,
            camera,
            width,
            height,
            mask,
            sigma=mask_sigma,
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            cam_dir / "masks" / f"{index:05d}.png", compress_level=1
        )
        np.save(cam_dir / "confidence" / f"{index:05d}.npy", conf.astype(np.float16))

    pts = frame.points
    pts = pts[np.isfinite(pts[:, 0:3]).all(axis=1)]
    with gzip.open(lidar_dir / f"{index:05d}.pkl.gz", "wb") as f:
        pickle.dump(pts, f, protocol=4)
    return len(frame.jpeg), int(pts.shape[0])


def write_dataset(
    rec: OdinRecording,
    mapping: dict[int, int],
    out_dir: str | Path,
    run_name: str,
    pose_offset: float = 0.0,
    camera_name: str = DEFAULT_CAMERA_NAME,
    overwrite: bool = False,
    generate_masks: bool = True,
    mask_sigma: float = 40.0,
    workers: int = 8,
) -> ProcessSummary:
    """Write the recording's aligned frames to ``out_dir``.

    ``mapping`` maps cloud index -> image index (from ``rec.align``).  Only
    cloud indices present in ``mapping`` are emitted (i.e. frames that have a
    matched image).  ``pose_offset`` shifts the odometry pose in frames.
    ``workers`` is the number of parallel writer processes (frames are
    independent; the input decode loop stays sequential).

    When ``generate_masks`` is set, per-frame LiDAR/RGB validity masks
    (``camera/<name>/masks/NNNNN.png``) and alpha confidence maps
    (``camera/<name>/confidence/NNNNN.npy``, float16) are also written.
    """
    out = Path(out_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {out} (use --overwrite)"
        )

    cam_dir = out / "camera" / camera_name
    lidar_dir = out / "lidar"
    cam_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = cam_dir / "masks"
    conf_dir = cam_dir / "confidence"
    camera = rec.camera_model()
    width, height = rec.intrinsics.image_width, rec.intrinsics.image_height
    if generate_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

    cloud_indices = sorted(mapping.keys())
    num_frames = 0
    total_pts = 0
    total_img_bytes = 0
    missing_calib = 0
    n_masks = 0

    camera_poses: list[dict] = []
    lidar_poses: list[dict] = []
    tcl_frames: list[dict] = []

    pending: deque = deque()
    total_img_bytes = 0
    total_pts = 0
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:

        def drain() -> None:
            """Tally completed futures; log progress every 200 frames."""
            nonlocal total_img_bytes, total_pts, done, n_masks
            while pending and pending[0].done():
                img_bytes, n_pts = pending.popleft().result()
                total_img_bytes += img_bytes
                total_pts += n_pts
                done += 1
                if generate_masks:
                    n_masks += 1
                if done % 200 == 0:
                    logger.info("wrote %d/%d frames", done, len(cloud_indices))

        for frame in rec.iter_frames(cloud_indices, mapping, pose_offset):
            if frame.jpeg is None:
                continue  # frame has a cloud but no matched image
            calib = frame.calib
            camera_poses.append(
                _pose_record(
                    frame.cloud_index,
                    frame.timestamp,
                    frame.image_index,
                    frame.camera_pose,
                )
            )
            lidar_poses.append(
                _pose_record(
                    frame.cloud_index,
                    frame.timestamp,
                    None,
                    frame.lidar_pose,
                )
            )
            tcl_frames.append(calib.to_dict())

            index = num_frames
            num_frames += 1
            pending.append(
                pool.submit(
                    _write_frame_payload,
                    frame,
                    index,
                    cam_dir,
                    lidar_dir,
                    camera,
                    width,
                    height,
                    generate_masks,
                    mask_sigma,
                )
            )
            if len(pending) >= max(4, workers * 4):
                wait(pending)
                drain()
        while pending:
            wait(pending)
            drain()

    # --- metadata -----------------------------------------------------------
    (cam_dir / "intrinsics.json").write_text(
        json.dumps(rec.intrinsics.to_dict(), indent=2)
    )
    (cam_dir / "poses.json").write_text(json.dumps(camera_poses, indent=2))
    (cam_dir / "tcl_frames.json").write_text(json.dumps(tcl_frames, indent=2))
    (lidar_dir / "poses.json").write_text(json.dumps(lidar_poses, indent=2))

    dataset_info = {
        "run": run_name,
        "origin_dir": str(rec.root),
        "num_frames": num_frames,
        "point_dim": 7,
        "point_fields": POINT_FIELDS,
        "color_range": COLOR_RANGE,
        "cloud_frame": "world (SLAM odom)",
        "camera": camera_name,
        "alignment": {
            "method": "timestamp-nearest",
            "notes": "image stream leads cloud stream by a constant number of frames",
        },
        "pose_offset": {
            "frames": pose_offset,
            "description": "odometry pose is sampled at frame_index + pose_offset frames",
        },
        "masks": {
            "enabled": generate_masks,
            "count": n_masks,
            "sigma": mask_sigma,
            "mask_path": f"camera/{camera_name}/masks/{{index:05d}}.png",
            "confidence_path": f"camera/{camera_name}/confidence/{{index:05d}}.npy",
            "description": "fov_mask: LiDAR dToF coverage hull (0/1); confidence: smoothed "
            "alpha reliability in [0,1] (float16) clipped to the mask",
        },
        "transforms": {
            "T_wc": "camera pose in world = T_wi * T_il * inv(T_cl)",
            "T_wl": "lidar pose in world = T_wi * T_il",
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    return ProcessSummary(
        out_dir=out,
        run_name=run_name,
        num_frames=num_frames,
        cloud_points=total_pts,
        image_bytes=total_img_bytes,
        dropped_frames=len(rec.image_toc) - num_frames,
        missing_calib_frames=missing_calib,
        masked_frames=n_masks,
    )
