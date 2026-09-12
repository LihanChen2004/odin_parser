"""Tests for reading a processed dataset back."""

import numpy as np

from odin_parser import OdinDataset
from odin_parser.process import ProcessOptions, process_dataset


def test_dataset_frames_iterator(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    process_dataset(
        ProcessOptions(origin_dir=origin, out_dir=out, overwrite=True, pose_offset=0.0)
    )

    ds = OdinDataset(out)
    assert len(ds) == 5
    records = list(ds.frames())
    assert len(records) == 5
    r0 = records[0]
    assert r0.index == 0
    assert r0.image_path.exists()
    assert r0.cloud_path.exists()
    assert r0.camera_pose.shape == (4, 4)
    assert r0.lidar_pose.shape == (4, 4)

    img = np.asarray(ds.image(0))
    assert img.shape == (
        ds.intrinsics["image_height"],
        ds.intrinsics["image_width"],
        3,
    )
    assert ds.point_cloud(0).shape[1] == 7


def test_dataset_rejects_non_dataset_dir(tmp_path):
    try:
        OdinDataset(tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
