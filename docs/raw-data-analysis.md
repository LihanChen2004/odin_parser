# Raw data: recorded vs available

What the proprietary recording persists versus what the Odin1 device exposes
via the SDK / ROS driver (`lidar_api_type.h`, `data_logger.h`,
`host_sdk_sample.h`, README).

## Recorded files

| file | per-frame contents |
|------|--------------------|
| `OdinImage.bin` | `u32 idx, f64 ts, u32 jpeg_size` + JPEG. Plain JPEG, no EXIF/APP metadata. |
| `OdinPose.bin` | `u32 idx, f64 ts` + 7xf32 odometry pose. |
| `OdinRotate.bin` | `u32 idx, f64 ts` + 4xf32 WIWC orientation. |
| `OdinIMU.bin` | `f64 ts` + 6xf32 (ax,ay,az,wx,wy,wz). 6-DoF; no magnetometer, no temp. |
| `MT<run>.olx` | `u32 idx, f64 ts, u32 npts` + per point `3xf32 xyz + 4xu8 rgba` (SLAM cloud). |
| `calib_online.yaml` | per-frame online `Tcl_0` + `body_T_lidar`. |
| `image/cam_in_ex.txt` | static intrinsics + distortion + static `Tcl_0`. |
| `image/info.txt` | device / firmware / algorithm / created_at. |
| `control_command.yaml` | capture-time device configuration. |

`BinaryDataLogger` (data_logger.h) writes the `.bin`/`.olx` files.  It does
**not** write depth, intensity, confidence, device status, AO/AE frames, or
masks.

## Signals the device exposes but the recording discards

- **raw DTOF cloud (`/odin1/cloud_raw`, `LIDAR_DT_RAW_DTOF`)** - richer than
  the SLAM cloud.  Per point (256x192 grid): `x,y,z (f32)`, `intensity (u8,
  reflectivity 0-255)`, `confidence (u16, 0..~1300, recommended filter
  threshold 30-35)`, `offset_time (f32, per-point time offset for motion
  compensation)`.  Only the derived SLAM cloud (xyzrgba) is stored.
- **device status (`LIDAR_DT_DEV_STATUS`)** - SoC/CPU/GPU/NPU temps, CPU & RAM
  use, rgb/dtof/imu sensor ODR, SLAM cloud tx rate.  Logged to
  `log/Driver_*/Conn_*/dev_status.csv` on the driver host (`devstatuslog: 1`),
  not in the dataset.
- **AE / AWB state (camera exposure)** - `exposure_time`, `gain`, `iso`,
  `brightness`, `rgain/bgain/cct`, via `/odin1/get_ae` / `/odin1/get_awb`.  Not
  recorded per frame; the images carry no EXIF, so per-frame exposure is not
  recoverable from the data.
- **image mask** - `sendimagemask` / `image_mask_abs_path`; off in the recorded
  runs.
- **NTP / PTP sync (`LIDAR_DT_NTP`)**, **high-frequency odometry** and
  **map-odom TF** - present as stream types but not written by the logger.

## The alpha channel

In `MT<run>.olx` the 4th colour field is labelled `alpha` but is **not
transparency**; it is a point-reliability / confidence-like value:

- Strongly negatively correlated with range (`corr(alpha, range) ~ -0.8`).
- Bimodal: high values (~234) mark trusted near points; a low tail marks
  far/noisy returns.
- Filter out low-alpha points before fusion / rendering.

## Getting the missing data (future recordings)

- Persist raw DTOF (intensity + confidence + offset_time) by logging
  `cloud_raw`; the current `BinaryDataLogger` does not (requires a
  driver/logger extension).
- `devstatuslog: 1` writes `dev_status.csv` on the driver host; copy that folder
  back if needed.
- Set `sendimagemask: 1` + a mask path to record an image mask.
- Per-frame AE/AWB requires subscribing to `/odin1/get_ae` / `/odin1/get_awb`.
