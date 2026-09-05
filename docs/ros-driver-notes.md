# ROS driver reference

Reference for the Odin1 ROS driver repository (`odin_ros_driver`), which is the
authoritative source for the binary formats and transforms used by this
package.

## Key source files

| File | What it defines |
|------|-----------------|
| `include/data_logger.h` | `BinaryDataLogger` - writers, file names, folder layout. |
| `include/host_sdk_sample.h` | The frame blobs written to each stream; pose/cloud/image math; TF trees. |
| `include/lidar_api_type.h` | `slam_cloud_point_t`, `ros_odom_convert_complete_t`, `imu_convert_data_t`, `capture_Image_List_t`. |
| `include/polynomial_camera.hpp` | Polynomial equidistant camera model (`world2cam`, `cam2world`). |
| `include/cloud_reprojector.hpp` / `.cpp` | Cloud-to-image reprojection; `calculateTic`. |
| `src/cloud_reprojection_ros.cpp` | Loads `cam_in_ex.txt`, subscribes to cloud/odom/wiwc, reprojects. |

## Data logger

- Root dir: `$ODIN_DATA_DIR` else `$HOME/OdinData`; subdir `YYYYMMDD_HHMMSS`.
- Files: `OdinPose.bin`, `MT<ts>.olx`, `OdinImage.bin`, `OdinRotate.bin`,
  `OdinIMU.bin`, `image/info.txt`, `image/cam_in_ex.txt`, `calib_online.yaml`.
- `info.txt` keys: `device`, `pointcloud=xyzrgbi`, optional `device_id`,
  `firmware_version`, `algorithm_version`, `created_at`.
- Writes are batched (default 20 frames).

## Frame blob construction (`host_sdk_sample.h`)

- **Image** (`publishRgb`): `[u32 idx][f64 ts][u32 jpeg_size][jpeg]`.
- **Cloud** (`publishPC2XYZRGBA`): `[u32 idx][f64 ts][u32 npts]` then per point
  `[f32 x][f32 y][f32 z][u8 r][u8 g][u8 b][u8 a]`.  `xyz` are the fixed-point
  int32 values / `1e4` (`SLAM_CLOUD_XYZ_TO_M = 1e-4`); colours are the low byte
  of each int32.
- **Pose** (`publishOdometry` STANDARD): `[u32 idx][f64 ts][7 x f32]`
  `(x,y,z,qx,qy,qz,qw)`.  Position = int64 µm / 1e6; quaternion = int64 / 1e6.
- **Rotate** (`recordrotate` on WIWC): `[u32 idx][f64 ts][4 x f32]` quaternion
  from `ros_odom_convert_complete_t::orient`.
- **IMU** (`publishImu`): `[f64 ts][6 x f32] (ax,ay,az,wx,wy,wz)` with the axis
  swap `(ax,ay,az)=(accel_y,-accel_x,accel_z)`,
  `(wx,wy,wz)=(gyro_y,-gyro_x,gyro_z)`.

## WIWC extrinsics

`LIDAR_DT_SLAM_WIWC` messages carry online extrinsic updates inside a
`ros_odom_convert_complete_t`:

- `pose.covariance[0:16]`  -> `Tcl_0` (4x4, row-major)
- `twist.covariance[0:16]` -> `body_T_lidar` (4x4, row-major)

The logger writes these to `calib_online.yaml` per WIWC frame, with `index` and
`timestamp`.  `frame.index` matches the odometry/cloud index.

## Reprojection chain (`cloud_reprojector`)

```
T_odom_imu = pose (odom->imu)
T_cam_odom = inv(Tic) * inv(T_odom_imu)      # odom -> camera
p_cam      = T_cam_odom * p_odom
(u,v)      = PolynomialCamera::world2cam(p_cam)   # distorted
or pinhole = fx*X/Z+cx, fy*Y/Z+cy                # undistorted (default)
```

## TF tree (host_sdk_sample, odom->camera)

```
odom ->(T_wi) imu ->(T_base_lidar = inv(T_il)) lidar ->(T_cl) camera
```

`host_sdk_sample.h` uses `T_wc = T_wi * inv(T_il) * T_cl`; the
`cloud_reprojector` chain gives `T_wc = T_wi * T_il * inv(T_cl)`.  The latter
is the one used by this package and matches `calculateTic`
(see `coordinates-and-calibration.md`).

## Timing / QoS notes

- `g_use_host_ros_time`: 0 = raw device time, 1 = host clock, 2 = PTP-aligned.
- 400 Hz topics (`/odin1/imu`, `/odin1/odometry_highfreq`) can drop under
  `ros2 bag record` because the recorder's `keep_last` queue overflows on slow
  disks; use a QoS override.
- The cloud device timestamp is typically ~100 ms ahead of odom; the driver
  burst-publishes TF samples so `tf2` can interpolate.
