# Validation

How to check that a processed dataset is correct.  These procedures are
dataset-agnostic and are re-run whenever a new recording or firmware needs
verifying.

## Time alignment

The cloud / pose / rotate / WIWC streams share identical timestamps and
indices.  The image stream leads the cloud stream by a constant number of
frames (7-9 at ~10 fps; the offset differs per run).

To check:

1. Header-only pass over each stream (seek over payloads) to build
   `[(index, timestamp)]` lists.
2. For each cloud frame, bisect the image timestamps for the nearest frame and
   require `|dt| <= 1 ms` (`align.align_cloud_to_image`).
3. Confirm the map is monotonic and the offset is constant.  The pipeline
   aligns by timestamp rather than a hard-coded index, so an offset that drifts
   would still be handled.

## Extrinsic / pose correctness

The driver repo contains two mutually inconsistent transform chains; the one
that matches `cloud_reprojector::calculateTic` is used.  Verify it on the
processed data:

**Colour-coherence test.**  For a frame, project the cloud (world frame) into
the image with a candidate pose and the camera model, then score

```
score = mean || rgb_cloud(point) - rgb_image(project(point)) ||
```

The correct pose minimises the score and maximises in-view point count.  Compare
`T_wc = T_wi * T_il * inv(T_cl)` against the alternative `T_wi * inv(T_il) *
T_cl`: the former wins by a large margin and by in-view count.

**Perturbation test.**  Perturb the camera pose and re-score.  A correct pose is
a sharp local optimum: a 1-2 deg rotation or a ~10 cm translation raises the
mean colour distance substantially.

**Sanity.**  At frame 0 the camera and lidar sit near the odom origin, inside
the cloud bounding box, and the camera's optical axis points into the cloud.

## Regression tests

`tests/test_process.py` builds a synthetic recording with a known image-lead
offset and asserts the alignment map, the output layout and the camera / lidar
pose values (including the `T_cl` inversion).  `tests/test_streams.py`,
`test_calib.py` and `test_geometry.py` cover the binary readers, the YAML
parsers and the SE(3) helpers.

```bash
uv run pytest
```
