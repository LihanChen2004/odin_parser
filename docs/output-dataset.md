# Processed output format

Description of every file produced in a processed (PandaSet-style) directory.

## Layout

```
processed/<run>/
├── dataset_info.json
├── camera/
│   └── front_camera/             # fixed name (Odin1 has a single camera)
│       ├── intrinsics.json       # static intrinsics + distortion + static Tcl_0
│       ├── tcl_frames.json       # per-frame online lidar<->camera extrinsics
│       ├── poses.json            # camera poses in the world frame
│       ├── masks/                # per-frame LiDAR/RGB validity mask (0/1)
│       ├── confidence/           # per-frame alpha confidence map (float16 [0,1])
│       └── 00000.jpg .. NNNNN.jpg
└── lidar/
    ├── poses.json                # lidar poses in the world frame
    └── 00000.pkl.gz .. NNNNN.pkl.gz
```

Per-frame files use the same **5-digit zero-padded index**, so frame `k` is
`index == k` in every `poses.json` array.

Masks and confidence are emitted by default (`--generate-masks` /
`ProcessOptions.generate_masks`, default `True`); the flag also accepts a
`--no-mask`/`generate_masks=False` to skip them.  The number of files matches
`num_frames`.

## Conventions

- **World frame** = the SLAM odometry frame (= the map frame at start).
- **Poses are world-from-sensor**: a 4x4 `T` maps `p_world = T @ [p_sensor; 1]`.
  Stored both as a row-major `matrix` and as `position` (m) + `quaternion`
  (x,y,z,w).
- **Timestamps** are seconds since the device boot.  The image stream is offset
  in the raw files; the pipeline has already time-aligned all streams.
- **Point clouds** are `(N,7)` float32 `x,y,z,r,g,b,a`; positions in m, colours
  in `[0,255]`.  Non-finite `xyz` are dropped.  Cloud points are already in the
  world frame.
- Raw recordings are never modified.

## Files

### `dataset_info.json`

| field | meaning |
|-------|---------|
| `run` | run id (`created_at`, e.g. `20260424_231347`) |
| `origin_dir` | absolute path of the recording that was parsed |
| `num_frames` | number of synchronised frames emitted |
| `point_dim` / `point_fields` | `7` / `["x","y","z","r","g","b","a"]` |
| `color_range` | `[0, 255]` |
| `cloud_frame` | `"world (SLAM odom)"` |
| `camera` | `"front_camera"` |
| `alignment` | method used to match image/cloud frames by timestamp |
| `pose_offset` | `{"frames": float, "description": str}` - the odometry pose is sampled at `frame_index + frames` (auto-estimated from colour coherence) |
| `masks` | `{"enabled": bool, "count": int, "sigma": float, "mask_path": str, "confidence_path": str, "description": str}` - how the per-frame validity/confidence files are written |
| `transforms` | the pose formulas used |

### `camera/front_camera/masks/NNNNN.png`

A **binary PNG** (0/1, `L` mode) defining the LiDAR/RGB valid region in the
image.  Pixel `1` = the projected LiDAR dToF coverage; `0` = the surrounding
margin where LiDAR and RGB do not overlap.

- Computed as the **convex hull** of the projected cloud points (see
  `masks.py::fov_mask`), filled and rasterised at the full image size.
- Geometry depends on the camera model + frame pose only, so it is stable frame
  to frame and unaffected by motion.
- Load as a boolean mask:

```python
mask = ds.mask(k)  # bool array, shape (image_height, image_width)
```

### `camera/front_camera/confidence/NNNNN.npy`

A **float16 array** in `[0, 1]`, same shape as the mask, giving the per-pixel
reliability of the LiDAR point cloud **within** the valid region (0 outside
the mask).

- Built from the `a` attribute of each projected point (a range/reliability
  signal, not opacity), scattered into the image, Gaussian-smoothed
  (`sigma = masks.sigma`, default 40 px), normalised to `[0,1]`, then clipped
  to the mask.
- Warm colours (high confidence) correspond to near surfaces; cool (low) to far
  ones, because `a` falls off with range.
- The mask defines the boundary; the confidence map weights *inside* it.  The
  two are emitted together because a single confidence map cannot reproduce the
  (dense, filled) coverage boundary on its own.

```python
import numpy as np

conf = ds.confidence(k)  # float16 (image_height, image_width) in [0,1]
```

### `camera/front_camera/intrinsics.json`

| field | meaning |
|-------|---------|
| `name` / `model` | `"front_camera"` / `"polynomial_equidistant"` |
| `image_width` / `image_height` | image size (e.g. 1600 x 1296) |
| `A11` / `A12` / `A22` | `K = [[A11, A12, u0], [0, A22, v0], [0, 0, 1]]` |
| `fx` / `fy` / `cx` / `cy` / `skew` | same values in pinhole notation |
| `K` | 3x3 intrinsic matrix (row-major) |
| `distortion_coefficients` | `[k2..k7]` of the equidistant model |
| `Tcl_0_static` | static 4x4 `lidar -> camera` matrix |

Projection of a camera-frame point `p=(X,Y,Z)` to a raw pixel:

```
r  = sqrt(X^2 + Y^2)
th = atan2(r, Z)
thd = th + k2*th^2 + ... + k7*th^7
u = A11 * X * thd/r + A12 * Y * thd/r + u0
v = A22 * Y * thd/r + v0
```

### `camera/front_camera/tcl_frames.json`

One entry per frame (ordered): `{index, timestamp, Tcl_0, body_T_lidar}`.

- `Tcl_0` : 4x4 `lidar -> camera` (`p_cam = Tcl_0 @ p_lidar`), the online value.
- `body_T_lidar` : 4x4 `lidar -> imu`.

Use the per-frame `Tcl_0` (not the static `Tcl_0_static`); the online values
update over time and are more accurate.

### `camera/front_camera/poses.json`

One entry per frame (ordered by index): `{index, timestamp, image_index,
position, quaternion, matrix}`.

- `matrix` : 4x4 `T_wc` (world-from-camera), row-major.
- `position` / `quaternion` : the same pose.
- `T_wc = T_wi * body_T_lidar * inv(Tcl_0)` where `T_wi` is the odometry pose
  (world-from-imu) and the extrinsics are from the matching frame.  See
  `coordinates-and-calibration.md`.

### `camera/front_camera/NNNNN.jpg`

Raw (distorted) RGB JPEG as recorded by the device.

### `lidar/poses.json`

Same schema as the camera `poses.json`, holding the lidar pose in the world
frame: `T_wl = T_wi * body_T_lidar` (`matrix` = world-from-lidar).

### `lidar/NNNNN.pkl.gz`

Pickle of a `numpy.ndarray`, dtype `float32`, shape `(N, 7)`:

```
x, y, z  (metres)   r, g, b, a  ([0,255])
```

Load:

```python
import gzip, pickle

with gzip.open("lidar/00000.pkl.gz", "rb") as f:
    pts = pickle.load(f)  # (N,7) float32
```

## Consuming (gsplat example)

```python
import numpy as np
from odin_parser import OdinDataset

ds = OdinDataset("processed/<run>")
for k in range(len(ds)):
    img = np.asarray(ds.image(k))  # HxWx3 uint8
    pts = ds.point_cloud(k)  # (N,7) float32, world frame
    Twc = ds.camera_pose(k)  # 4x4 world-from-camera
    Twl = ds.lidar_pose(k)  # 4x4 world-from-lidar

    xyz = pts[:, :3].astype(np.float32)
    rgb = (pts[:, 3:6] / 255.0).astype(np.float32)  # gsplat expects [0,1]

    # optional LiDAR/RGB validity + confidence (present when masks were written)
    mask = ds.mask(k)  # bool HxW, 1 = LiDAR/RGB overlap
    conf = ds.confidence(k)  # float32 HxW in [0,1]
```

Cloud points are world-frame and camera poses are world-from-camera, so you can
render directly: invert `Twc` to map camera rays, or feed `xyz`/`rgb` with
`Twc[:3,3]` / `Twc[:3,:3]`.

The `masks`/`confidence` files let you weight the rendered loss by where the
sensor is actually reliable: multiply splat alpha or the photometric loss by
`conf` (or by `mask`) before accumulating, so empty/marginal regions contribute
little.
