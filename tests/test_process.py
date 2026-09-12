"""End-to-end test of the processing pipeline on a synthetic origin dir."""

import gzip
import json
import pickle

import numpy as np
import pytest

from odin_parser import OdinRecording
from odin_parser.output import write_dataset
from odin_parser.process import ProcessOptions, process_dataset


def test_process_dataset(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)

    summary = process_dataset(
        ProcessOptions(origin_dir=origin, out_dir=out, overwrite=True, pose_offset=0.0)
    )
    assert summary.num_frames == 5  # all cloud frames have a matching image
    assert summary.skipped_cloud_frames == 0

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
    poses = json.loads((cam_dir / "poses.json").read_text())
    assert poses[0]["index"] == 0
    assert poses[0]["image_index"] == 9

    # camera pose = T_wi * til * inv(tcl)
    # T_wi translation (1,2,3); til translation (0.1,0.2,0.3)
    #   -> lidar pose (1.1,2.2,3.3)
    # inv(tcl) translation (-0.5,0,0) -> camera pose (0.6, 2.2, 3.3)
    np.testing.assert_allclose(poses[0]["position"], [0.6, 2.2, 3.3], atol=1e-9)

    lidar_poses = json.loads((lidar_dir / "poses.json").read_text())
    np.testing.assert_allclose(lidar_poses[0]["position"], [1.1, 2.2, 3.3], atol=1e-9)

    # point cloud round trip
    with gzip.open(lidar_dir / "00000.pkl.gz", "rb") as f:
        pts = pickle.load(f)
    assert pts.shape == (200, 7)
    assert pts.dtype == np.float32


def test_process_overwrite_guard(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    out.mkdir()
    try:
        process_dataset(ProcessOptions(origin_dir=origin, out_dir=out, pose_offset=0.0))
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass


def test_process_no_masks(tmp_path, make_origin):
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


def test_process_masks_sigma_zero(tmp_path, make_origin):
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


def test_write_dataset_index_matches_filename(tmp_path, make_origin):
    """An emitted frame's JSON `index` must equal its file suffix, even when the
    input mapping skips cloud frames."""
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    rec = OdinRecording(origin)
    try:
        summary = write_dataset(
            rec,
            mapping={1: 10, 3: 12},  # cloud 0 and 2 are skipped
            out_dir=out,
            run_name=rec.run_name,
            pose_offset=0.0,
            overwrite=True,
        )
        assert summary.num_frames == 2
        assert summary.skipped_cloud_frames == 3  # 5 cloud frames, 2 written

        cam_dir = out / "camera" / "front_camera"
        poses = json.loads((cam_dir / "poses.json").read_text())
        assert [p["index"] for p in poses] == [0, 1]
        assert [p["cloud_index"] for p in poses] == [1, 3]
        tcl = json.loads((cam_dir / "tcl_frames.json").read_text())
        assert [t["index"] for t in tcl] == [0, 1]
        for k in (0, 1):
            assert (cam_dir / f"{k:05d}.jpg").exists()
            assert (out / "lidar" / f"{k:05d}.pkl.gz").exists()
    finally:
        rec.close()


def test_process_missing_calib_fails_fast(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    # Keep only the first frame's online calibration.
    calib = (origin / "calib_online.yaml").read_text()
    (origin / "calib_online.yaml").write_text("frame:" + calib.split("frame:")[1])

    with pytest.raises(RuntimeError, match="online calibration"):
        process_dataset(ProcessOptions(origin_dir=origin, out_dir=out, pose_offset=0.0))
