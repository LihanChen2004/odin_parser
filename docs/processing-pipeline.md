# Processing pipeline

Design of `odin_parser.process.process_dataset`.

## Architecture

The pipeline is split into three layers:

- `odin_parser.recording.OdinRecording` - random-access view of a raw
  recording.  It lazily builds **one** header index (TOC) per stream and reuses
  it for time alignment, random access, pose interpolation and frame iteration,
  so a stream's header is never scanned more than once.
- `odin_parser.sync` - cross-sensor time-offset estimator (colour coherence).
- `odin_parser.output.write_dataset` - the PandaSet-style writer and on-disk
  schema (read back by `odin_parser.dataset.OdinDataset`).

`process_dataset` only wires these together.

## Goals

- Stream the whole dataset with bounded memory (streams are ~2 GB each).
- Emit only synchronised frames (image + cloud + poses together).
- Keep the output lossless (positions in m, colours in `[0,255]`).

## Inputs

A raw recording directory (see `binary-format.md`).  Inputs are read-only.

## Output schema

See `output-dataset.md` for the full schema.  In short: images copied verbatim,
clouds converted to `(N,7)` float32 and gzipped per frame, plus JSON metadata
(`poses.json`, `intrinsics.json`, `tcl_frames.json`, `dataset_info.json`).
Per-frame `masks/` (LiDAR/RGB validity, PNG) and `confidence/` (alpha
reliability, float16 `.npy`) are also written when masks are enabled
(`--generate-masks`).

## Steps

1. **Discover** streams (glob `Odin*.bin`, `MT<run>.olx`) and build one TOC per
   stream.
2. **Static calib** - parse `image/cam_in_ex.txt`, `image/info.txt`.
3. **Online calib** - parse `calib_online.yaml` into `{index: CalibFrame}`.
   (The file repeats top-level `frame:` keys; blocks are split before YAML
   parsing to avoid duplicate-key collapse.)
4. **Time alignment** - `align_cloud_to_image` bisects for the nearest image per
   cloud frame and keeps matches within tolerance (1 ms).  The image stream
   leads the cloud stream by a constant number of frames; the offset differs per
   recording (observed 7-9 at ~10 fps), so alignment is by timestamp rather than
   a hard-coded index.
5. **Pose time offset** - estimate the odometry pose lag (in frames) by
   scanning interpolated poses and scoring reprojection colour coherence
   (`sync.estimate_pose_offset`); it is data-driven, not hard-coded.
6. **Write outputs** - frames are independent, so per-frame file writes run on
   a pool of `--workers` (default 8) processes while the input decode loop
   stays sequential: copy images; convert clouds to float32, drop non-finite
   `xyz`, gzip per frame; write confidence maps as float16 `.npy` and masks as
   PNG; write JSON metadata last.
7. **Report** - `ProcessSummary` with frame count / points / image bytes.

## Memory / performance notes

- One streaming pass per payload file; the header index is built once.
- Header-only alignment/TOC passes `f.seek(payload, 1)` and never load payloads.
- Per-frame numpy arrays are transient (~34k x 16 bytes raw); peak RSS stays in
  the tens of MB per worker.
- The write stage is embarrassingly parallel (frames are independent); the
  dominant per-frame costs are the confidence Gaussian smoothing and PNG
  compression, which parallelise across `--workers` processes.  A ~3900-frame
  recording exports in a few minutes on an 8-worker setup (NAS-bound at
  ~15-20 MB/s).

## Extending

- Emit `OdinIMU.bin` samples by streaming `read_imu_stream` alongside the main
  pass and matching by timestamp (note: the IMU time base differs - see
  `binary-format.md`).
- Add the `OdinRotate.bin` (WIWC) orientation to a pose file by using
  `read_rotate_stream` (index-aligned with odometry).
- Produce undistorted images with `PolynomialCamera`'s inverse distortion.
