"""Unit tests for the OdinRecording abstraction."""

import numpy as np

from odin_parser import OdinRecording
from odin_parser.process import ProcessOptions, process_dataset


def test_recording_discovery_and_align(tmp_path, make_origin):
    origin = tmp_path / "origin"
    make_origin(origin)
    rec = OdinRecording(origin)

    assert rec.run_name == "20260424_231347"
    assert rec.n_cloud == 5
    assert rec.n_image == 14  # 5 cloud + 9 image lead
    # image stream leads cloud by 9 frames, exact timestamps
    mapping = rec.align(1e-3)
    assert len(mapping) == 5
    assert all(mapping[c] - c == 9 for c in mapping)

    # random access round-trips
    pts = rec.cloud(0)
    assert pts.shape == (200, 7) and pts.dtype == np.float32
    jpg = rec.jpeg(0)
    assert isinstance(jpg, bytes) and len(jpg) > 0

    # camera / lidar poses are finite 4x4
    Twc = rec.camera_pose(0)
    Twl = rec.lidar_pose(0)
    assert Twc.shape == (4, 4) and np.all(np.isfinite(Twc))
    assert Twl.shape == (4, 4) and np.all(np.isfinite(Twl))
    rec.close()


def test_recording_pose_at_interpolation(tmp_path, make_origin):
    origin = tmp_path / "origin"
    # Vary the pose per frame so interpolation cannot pass by accident.
    make_origin(origin, pose_fn=lambda i: (float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    rec = OdinRecording(origin)
    m0 = rec.pose_at(0.0)
    m1 = rec.pose_at(1.0)
    mh = rec.pose_at(0.5)
    np.testing.assert_allclose(m0[:3, 3], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(m1[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    # position should be halfway between the two distinct poses
    np.testing.assert_allclose(mh[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
    rec.close()


def test_recording_iter_frames_yields_bundle(tmp_path, make_origin):
    origin = tmp_path / "origin"
    make_origin(origin)
    rec = OdinRecording(origin)
    mapping = rec.align(1e-3)
    frames = list(rec.iter_frames(sorted(mapping.keys()), mapping, 0.0))
    assert len(frames) == 5
    f0 = frames[0]
    assert f0.jpeg is not None
    assert f0.camera_pose.shape == (4, 4)
    assert f0.lidar_pose.shape == (4, 4)
    assert f0.points.shape[1] == 7
    rec.close()


def test_recording_estimate_pose_offset_runs(tmp_path, make_origin):
    origin = tmp_path / "origin"
    make_origin(origin)
    rec = OdinRecording(origin)
    mapping = rec.align(1e-3)
    samples = sorted(mapping.keys())[1:-1]
    est = rec.estimate_pose_offset(samples)
    assert -2.0 <= est.offset_frames <= 2.0
    assert est.samples == len(samples)
    rec.close()


def test_recording_tolerates_missing_rotate_imu(tmp_path, make_origin):
    origin = tmp_path / "origin"
    # Rotate/IMU are parsed but unused by the pipeline; their absence must not
    # make the whole recording unreadable.
    make_origin(origin, include_rotate_imu=False)
    rec = OdinRecording(origin)
    assert rec.n_cloud == 5
    assert rec.align(1e-3)
    rec.close()


def test_process_uses_recording_under_the_hood(tmp_path, make_origin):
    origin = tmp_path / "origin"
    out = tmp_path / "out"
    make_origin(origin)
    summary = process_dataset(
        ProcessOptions(origin_dir=origin, out_dir=out, overwrite=True, pose_offset=0.0)
    )
    assert summary.num_frames == 5
    assert (out / "dataset_info.json").exists()
