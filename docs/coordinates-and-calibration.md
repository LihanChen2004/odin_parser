# Coordinates, calibration and camera model

How points and poses are represented, and the camera projection model used for
reprojection.  These definitions are the ground truth for everything the
package emits.

## Frames

| Frame  | Description                                                       |
|--------|-------------------------------------------------------------------|
| `odom` | SLAM odometry frame; equals the map frame at recording start. World frame of the output. |
| `imu`  | IMU / body frame.                                                 |
| `lidar`| dTOF LiDAR frame.                                                 |
| `cam`  | front camera frame (optical: +X right, +Y down, +Z forward).      |

## Transform convention

`T_a_b` maps points from frame `b` into frame `a`: `p_a = T_a_b @ [p_b; 1]`.

- `T_wi` : `imu -> odom` (world-from-imu), from `OdinPose.bin` (position in m,
  unit quaternion).
- `T_il` : `lidar -> imu` = `body_T_lidar` in `calib_online.yaml`.
- `T_cl` : `lidar -> camera` = `Tcl_0` in `calib_online.yaml`
  (i.e. the camera pose expressed in the lidar frame, `p_cam = T_cl @ p_lidar`).

### Poses emitted by the pipeline

```
camera pose in world:  T_wc = T_wi * T_il * inv(T_cl)
lidar  pose in world:  T_wl = T_wi * T_il
```

`T_wc` from the chain:
`p_cam = T_cl @ p_lidar`, `p_lidar = inv(T_il) @ p_imu`,
`p_imu = inv(T_wi) @ p_odom`  =>  `p_odom = T_wi @ T_il @ inv(T_cl) @ p_cam`.

Point clouds are already in the `odom` (world) frame; no transform is applied.

The driver repo contains two inconsistent chains (`host_sdk_sample.h` uses
`T_wc = T_wi * inv(T_il) * T_cl`); the formula above is the one the pipeline
uses and matches `cloud_reprojector::calculateTic`.

## Camera intrinsics

From `image/cam_in_ex.txt` (`cam_0`):

```
K = [[A11, A12, u0],
     [  0, A22, v0],
     [  0,   0,  1]]           # fx=A11, fy=A22, skew=A12, cx=u0, cy=v0
```

Distortion is the **polynomial / equidistant** model
(`polynomial_camera.hpp`):

```
theta_d = theta + k2*theta^2 + k3*theta^3 + ... + k7*theta^7
```

`k1` is not reported (treated as 0).  Projection of a camera-frame point to a
raw (distorted) pixel:

```
r  = sqrt(X^2 + Y^2)
th = atan2(r, Z)
thd = theta_d(th)
u = fx * X * thd/r + skew * Y * thd/r + cx
v = fy * Y * thd/r + cy
```

The recorded JPEGs are the raw distorted frames; the pinhole (undistorted)
variant is `u = fx*X/Z + cx`, `v = fy*Y/Z + cy` and is provided for comparison.

## Online vs static extrinsics

- `image/cam_in_ex.txt` holds the **static** initial `Tcl_0`.
- `calib_online.yaml` holds a **per-frame** `Tcl_0` and `body_T_lidar` that the
  device refines online.  The pipeline uses the per-frame values
  (`frame.index` == cloud/pose index).

## Output conventions

- `poses.json` `matrix` is the 4x4 world-from-sensor pose (row-major);
  `position`/`quaternion` are the same pose as `(x,y,z)` / `(x,y,z,w)`.
- `intrinsics.json` stores `K`, `A11/A12/A22/u0/v0`, `distortion_coefficients`,
  image size and the static `Tcl_0_static`.
- `tcl_frames.json` stores each online `Tcl_0` and `body_T_lidar` matrix.
