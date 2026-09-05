# Binary data format

Binary layout of every stream in a raw Odin1 recording, taken from the Odin1
ROS driver (`include/data_logger.h`, `include/host_sdk_sample.h`,
`include/lidar_api_type.h`).

All multi-byte fields are **little-endian**.  Frames are written back-to-back
with no file-level header.  Streams are recorded independently, so **indices do
not match across files** - always align by timestamp.

| Stream file            | Variable length? | Frame header size |
|------------------------|------------------|-------------------|
| `OdinImage.bin`        | yes (JPEG)       | 16 bytes          |
| `OdinPose.bin`         | no (fixed 40 B)  | -                 |
| `OdinRotate.bin`       | no (fixed 28 B)  | -                 |
| `OdinIMU.bin`          | no (fixed 32 B)  | -                 |
| `MT<run>.olx` (cloud)  | yes (points)     | 16 bytes          |

## `OdinImage.bin` - JPEG image stream

```
offset  size  type    field
0       4     uint32  frame_index
4       8     float64 timestamp_sec
12      4     uint32  jpeg_size
16      jpeg_size     JPEG payload
```

JPEG decodes to an RGB image (the recorded dimensions are in
`cam_in_ex.txt`).  Written by `host_sdk_sample.cpp::publishRgb`.

## `OdinPose.bin` - odometry (7-DOF)

```
offset  size  type    field
0       4     uint32  frame_index
4       8     float64 timestamp_sec
12      4     float32 x   (metres)
16      4     float32 y
20      4     float32 z
24      16    4xfloat32 qx,qy,qz,qw  (unit quaternion, world->imu)
```

From `publishOdometry` (`LIDAR_DT_SLAM_ODOMETRY`).  The logged stream already
converts the device's int64 µm / 1e-6 quaternion to float32 metres / unit quat
(`ros_odom_convert_complete_t`).

## `OdinRotate.bin` - WIWC rotation only

```
offset  size  type    field
0       4     uint32  frame_index
4       8     float64 timestamp_sec
12      16    4xfloat32 qx,qy,qz,qw
```

From `recordrotate` on `LIDAR_DT_SLAM_WIWC`.  It is the WIWC orientation only.

## `OdinIMU.bin` - IMU stream

```
offset  size  type    field
0       8     float64 timestamp_sec
8       24    6xfloat32 ax, ay, az, wx, wy, wz
```

- `ax,ay,az` in m/s^2, `wx,wy,wz` in rad/s, already axis-swapped by the driver:
  `(ax,ay,az)=(accel_y,-accel_x,accel_z)`,
  `(wx,wy,wz)=(gyro_y,-gyro_x,gyro_z)`.
- The IMU timestamp base differs from the other streams; do not assume it shares
  the odometry time base.  It is not used in the standard output.

## `MT<run>.olx` - SLAM point cloud stream

```
offset  size  type    field
0       4     uint32  frame_index
4       8     float64 timestamp_sec
12      4     uint32  num_points
16      num_points * 16 bytes  points
```

Each 16-byte point:

```
offset  size  type       field
0       12    3x float32 x, y, z   (metres)
12      4     4x uint8   r, g, b, a
```

- Points are already in the SLAM odometry (world) frame (the driver publishes
  the same data on `/odin1/cloud_slam` with `frame_id="odom"`).
- The device wire format is `7 x int32` (`slam_cloud_point_t`: `xyz` in 0.1 mm,
  `rgba` in the low byte of each int32); the logger writes float32 xyz + uint8
  rgba.
- `a` (alpha) is a per-point reliability/confidence-like value, not
  transparency (see `raw-data-analysis.md`).

## Timestamps

`aligned_stamp_ns()` returns:
- `g_use_host_ros_time == 0` : raw device boot time,
- `== 1` : host wall clock,
- `== 2` : device timestamp aligned by smoothed PTP offset.

The cloud / pose / rotate / WIWC timestamps are index-aligned.  The image
stream leads the cloud stream by a constant number of frames (7-9 at ~10 fps,
depending on the run); image frame `j` matches cloud frame `j - offset`.
