# Processing pipeline

Design of `odin_parser.process.process_dataset`.

## Architecture

`process_dataset` only wires the pieces together; the work lives in:

| module | responsibility |
|--------|----------------|
| `streams` | binary readers.  `iter_frame_spans` frames any header-prefixed stream and backs both the one-pass generators (`read_*_stream`) and the random-access TOCs (`build_*_toc` + `*Index`). |
| `recording` | `OdinRecording`, a random-access view of a raw recording.  Builds one header index (TOC) per stream and reuses it for alignment, random access, pose interpolation and frame iteration, so a stream's header is scanned once. |
| `calib` | parsers for `image/cam_in_ex.txt`, `image/info.txt` and `calib_online.yaml` (intrinsics and static/online extrinsics). |
| `camera` | `PolynomialCamera`, the equidistant projection model. |
| `geometry` | SE(3) and quaternion helpers. |
| `align` | timestamp-based cloud<->image matching. |
| `sync` | cross-sensor pose time-offset estimator (colour coherence). |
| `masks` | per-frame LiDAR/RGB validity mask and alpha confidence map. |
| `output` | `write_dataset`, the PandaSet-style writer and on-disk schema. |
| `dataset` | `OdinDataset`, which reads a processed tree back. |
| `visualize` | point-cloud projection and overlay rendering. |
| `cli` | the `odin-parser info\|process\|preview` entry point. |
| `process` | `ProcessOptions` and `process_dataset`, which wire the above. |

`docs/output-dataset.md` is the schema reference and
`docs/coordinates-and-calibration.md` the frame/transform reference.

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
(`--masks`, the default).

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
- `iter_frame_spans` walks a stream header-only (seeking past each payload) and
  backs both the TOC and the streaming readers, so payloads are never loaded
  just to index a stream.
- Per-frame numpy arrays are transient (~34k x 16 bytes raw); peak RSS stays in
  the tens of MB per worker.

### Cost profile

The write stage is embarrassingly parallel (frames are independent) and is
**CPU-bound, not IO-bound**.  A representative 1005-frame recording
(1600x1296, ~37k points/frame) costs ~98 s wall on 8 workers at ~590% CPU,
nearly all of it in the write stage (TOC + align ~0.03 s, pose-offset estimate
~2.8 s).  The dominant per-frame costs are:

| step | ms/frame | note |
|------|----------|------|
| `confidence_map` (Gaussian, sigma=40) | ~310 | dominant; kernel radius ~160 px |
| `gzip` pickled cloud (level 9, the default) | ~180 | level 1 ~13 ms, level 6 ~32 ms |
| `fov_mask` | ~9 | |
| `np.save` float16 confidence | ~5 | 4.15 MB/frame -> ~4 GB per 1000 frames |
| cloud read + decode | ~0.6 | |

Output volume is ~4.9 GB per 1000 frames, ~80% of it float16 confidence maps.
The NFS mount sustains ~240 MB/s read and ~113 MB/s write while the pipeline
reaches ~70 MB/s of writes, so storage bandwidth is not the limiter.
`--no-masks` skips the Gaussian and the confidence maps, removing the two
largest costs.

## Extending

- Emit `OdinIMU.bin` samples by streaming `read_imu_stream` alongside the main
  pass and matching by timestamp (note: the IMU time base differs - see
  `binary-format.md`).
- Add the `OdinRotate.bin` (WIWC) orientation to a pose file by using
  `read_rotate_stream` (index-aligned with odometry).
- Add an inverse of `PolynomialCamera.world_to_pixel` (an iterative `cam2world`)
  to undistort images; the forward projection model exists, the inverse does
  not yet.
