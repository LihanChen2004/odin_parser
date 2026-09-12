# odin-parser

[![CI](https://github.com/LihanChen2004/odin_parser/actions/workflows/ci.yml/badge.svg)](https://github.com/LihanChen2004/odin_parser/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A parser for the **ManifoldTech Odin1** device's private binary recording
format, converting a recording into a PandaSet-style layout suitable for
[gsplat](https://www.gsplat.studio/).

The binary formats and coordinate conventions are defined by the Odin1 ROS
driver (ManifoldTech's `odin_ros_driver`, shipped with the device SDK); see
[`docs/`](docs/) for the details.

https://github.com/user-attachments/assets/1805f3f0-2b13-4ccd-a253-be46c2ce1f88

## Installation

Requires `uv` (or Python >= 3.11 + pip). Tested on Python 3.11 and 3.12:

```bash
uv sync                  # creates .venv with Python 3.11
uv run odin-parser --help
```

## Development

```bash
uv sync                              # create .venv with the dev dependencies
uv run pytest -q                     # run the test suite
uvx ruff check src tests scripts     # lint
uvx ruff format src tests scripts    # format
pre-commit install                   # optional: run the hooks on every commit
```

## Pipeline

```
raw recording ──process──▶ processed/  ──preview / make_video──▶ image / MP4
```

### Input: raw recording

```
<run>/
├── OdinImage.bin        # JPEG image stream (16-byte frame header)
├── OdinPose.bin         # 7-DOF odometry (pos + quaternion)
├── OdinRotate.bin       # WIWC rotation-only stream
├── OdinIMU.bin          # 6-DoF IMU stream
├── MT<run>.olx          # SLAM point cloud stream (dTOF lidar)
├── calib_online.yaml    # per-frame online lidar<->camera extrinsics
└── image/
    ├── cam_in_ex.txt    # static intrinsics + distortion + static Tcl_0
    └── info.txt         # device / run metadata
```

### Output: processed (PandaSet-style)

```
processed/<run>/
├── dataset_info.json            # provenance + transform documentation
├── camera/front_camera/
│   ├── intrinsics.json          # intrinsics + distortion + static Tcl_0
│   ├── tcl_frames.json          # per-frame online Tcl_0 + body_T_lidar
│   ├── poses.json               # camera poses in the world frame
│   └── 00.jpg ... NNNNN.jpg     # RGB images
└── lidar/
    ├── poses.json               # lidar poses in the world frame
    └── 00.pkl.gz ... NNNNN.pkl.gz  # point clouds (N,7) float32 x,y,z,r,g,b,a
```

Conventions:

- Frames are **time-aligned by timestamp**; only frames that have both a cloud
  and an image are emitted.
- **Pose time offset**: the odometry pose can lag the frame's capture instant
  by a fraction of a frame.  The pipeline estimates this from the data (colour
  coherence) and stores it in `dataset_info.json` under `pose_offset.frames`; the
  emitted poses sample the odometry at `frame_index + pose_offset`.
- The camera sub-directory is always `front_camera` (Odin1 has a single camera).
- Cloud points are already in the SLAM odometry (world) frame; no transform is
  applied.
- Poses `T_wc` / `T_wl` are 4x4 **world-from-sensor** matrices
  (`p_world = T @ p_sensor`), also stored as `position` + `quaternion`.
- Point clouds are `(N,7)` float32 `x,y,z,r,g,b,a`, colours in `[0,255]`.

See [`docs/output-dataset.md`](docs/output-dataset.md) for the full schema.

## CLI

```bash
# Inspect a raw recording
odin-parser info <run>

# Convert to the PandaSet-style processed layout
odin-parser process <run> --out <out> --overwrite

# Visualise a single frame (camera image + projected point cloud)
odin-parser preview <out> --frame 500 --out overlay.png --mode color

# Render the whole sequence to an MP4 (Image + projected cloud, 3x speed)
python scripts/make_video.py <out> out.mp4
```

`make_video.py [--speed=3] [--radius=2] [--alpha=0.7] [--darken=0.85]
[--max-frames=N]`.  The scene is ~10 fps, so by default the MP4 runs at ~30 fps
(3x speed).

## Python API

```python
from odin_parser import process_dataset, ProcessOptions, OdinDataset

summary = process_dataset(
    ProcessOptions(
        origin_dir="/data/origin/<run>",
        out_dir="/data/processed/<run>",
        overwrite=True,
    )
)

ds = OdinDataset("/data/processed/<run>")
img = ds.image(0)  # PIL RGB image
cloud = ds.point_cloud(0)  # (N,7) float32, world frame
Twc = ds.camera_pose(0)  # 4x4 world_from_camera
Twl = ds.lidar_pose(0)  # 4x4 world_from_lidar
```

## Notes

- Raw recordings are read-only; they are never modified.
- `OdinRotate.bin` (WIWC) is parsed but not used for poses; camera/lidar poses
  come from odometry + online extrinsics.

## Documentation

- [`docs/binary-format.md`](docs/binary-format.md) - binary layouts of every stream.
- [`docs/coordinates-and-calibration.md`](docs/coordinates-and-calibration.md) -
  frames, transforms and the camera projection model.
- [`docs/output-dataset.md`](docs/output-dataset.md) - full processed output schema.
- [`docs/processing-pipeline.md`](docs/processing-pipeline.md) - pipeline design and streaming notes.
- [`docs/validation.md`](docs/validation.md) - how alignment/extrinsics are checked.
- [`docs/raw-data-analysis.md`](docs/raw-data-analysis.md) - what is recorded vs what the device exposes.
- [`docs/ros-driver-notes.md`](docs/ros-driver-notes.md) - reference to the driver source.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Disclaimer

Unofficial tool, not affiliated with or endorsed by ManifoldTech.  Binary
formats were documented from recordings of our own device and the driver
shipped with it.
