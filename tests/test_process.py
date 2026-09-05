"""End-to-end test of the processing pipeline on a synthetic origin dir."""

import gzip
import pickle
import struct
from pathlib import Path

import numpy as np

from odin_parser.process import ProcessOptions, process_dataset


def make_origin(root: Path, n_cloud: int = 5, offset: int = 9, fps: float = 10.0):
    """Build a small synthetic Odin1 recording directory.

    cloud ts:      100 + i * dt
    image ts:      (100 - offset*dt) + j * dt   -> image[j] ~ cloud[j - offset]
    poses/rotate:  aligned with cloud ts
    """
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
        tcl = [
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
        ]
        til = [
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
        ]
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
        pose += struct.pack(
            "<Id7f", i, 100.0 + i * dt, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0
        )
    (root / "OdinPose.bin").write_bytes(pose)
    (root / "OdinRotate.bin").write_bytes(
        b"".join(
            struct.pack("<Id4f", i, 100.0 + i * dt, 0, 0, 0, 1) for i in range(n_cloud)
        )
    )
    (root / "OdinIMU.bin").write_bytes(
        b"".join(
            struct.pack("<d6f", 100.0 + i * 0.0025, 0, 0, 0, 0, 0, 0) for i in range(40)
        )
    )

    # image stream (leads cloud by `offset` frames); real-size JPEG so that
    # projected points land inside the image (needed by pose-offset estimation)
    from io import BytesIO

    from PIL import Image

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


def test_process_dataset(tmp_path):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)

    summary = process_dataset(
        ProcessOptions(origin_dir=origin, out_dir=out, overwrite=True, pose_offset=0.0)
    )
    assert summary.num_frames == 5  # all cloud frames have a matching image
    assert summary.dropped_frames == 9

    cam_dir = out / "camera" / "front_camera"
    lidar_dir = out / "lidar"
    assert (cam_dir / "intrinsics.json").exists()
    assert (cam_dir / "poses.json").exists()
    assert (cam_dir / "tcl_frames.json").exists()
    assert (lidar_dir / "poses.json").exists()
    for i in range(5):
        assert (cam_dir / f"{i:05d}.jpg").exists()
        assert (lidar_dir / f"{i:05d}.pkl.gz").exists()

    # per-frame LiDAR/RGB validity masks + confidence maps
    from odin_parser import OdinDataset

    ds = OdinDataset(out)
    for i in range(5):
        assert ds.mask_path(i).exists()
        assert ds.confidence_path(i).exists()
        m = ds.mask(i)
        c = ds.confidence(i)
        assert m.dtype == bool and m.shape == (
            ds.intrinsics["image_height"],
            ds.intrinsics["image_width"],
        )
        assert c.dtype == np.float16 and c.shape == m.shape
        assert float(c.max()) <= 1.0 + 1e-6
        # confidence is zero outside the mask
        assert (c[~m] == 0).all()
    assert summary.masked_frames == 5
    assert ds.info["masks"]["enabled"] is True

    # image/cloud alignment: output frame 0 <- cloud index 0, image index 9
    poses = __import__("json").loads((cam_dir / "poses.json").read_text())
    assert poses[0]["index"] == 0
    assert poses[0]["image_index"] == 9

    # camera pose = T_wi * til * inv(tcl)
    # T_wi translation (1,2,3); til translation (0.1,0.2,0.3) -> lidar pose (1.1,2.2,3.3)
    # inv(tcl) translation (-0.5,0,0) -> camera pose (0.6, 2.2, 3.3)
    np.testing.assert_allclose(poses[0]["position"], [0.6, 2.2, 3.3], atol=1e-9)

    lidar_poses = __import__("json").loads((lidar_dir / "poses.json").read_text())
    np.testing.assert_allclose(lidar_poses[0]["position"], [1.1, 2.2, 3.3], atol=1e-9)

    # point cloud round trip
    with gzip.open(lidar_dir / "00000.pkl.gz", "rb") as f:
        pts = pickle.load(f)
    assert pts.shape == (200, 7)
    assert pts.dtype == np.float32


def test_process_overwrite_guard(tmp_path):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    out.mkdir()
    try:
        process_dataset(ProcessOptions(origin_dir=origin, out_dir=out, pose_offset=0.0))
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass


def test_process_no_masks(tmp_path):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    summary = process_dataset(
        ProcessOptions(
            origin_dir=origin,
            out_dir=out,
            pose_offset=0.0,
            generate_masks=False,
        )
    )
    cam_dir = out / "camera" / "front_camera"
    assert summary.masked_frames == 0
    assert not (cam_dir / "masks").exists()
    assert not (cam_dir / "confidence").exists()
    assert summary.num_frames == 5
    assert (cam_dir / "00000.jpg").exists()  # images still written


def test_process_masks_sigma_zero(tmp_path):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    process_dataset(
        ProcessOptions(
            origin_dir=origin,
            out_dir=out,
            pose_offset=0.0,
            mask_sigma=0.0,
        )
    )
    from odin_parser import OdinDataset

    ds = OdinDataset(out)
    conf = ds.confidence(0)
    # sigma=0 -> no smoothing, so confidence is a sparse scatter fit to [0,1]
    assert conf.dtype == np.float16 and conf.shape == (
        ds.intrinsics["image_height"],
        ds.intrinsics["image_width"],
    )
    assert float(conf.max()) <= 1.0 + 1e-6
