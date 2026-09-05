"""odin_parser - Parse ManifoldTech Odin1 private binary datasets.

The package reads the raw streams recorded by the Odin1 device and the
ManifoldTech ROS driver (see ``docs/binary-format.md`` for the authoritative
binary layout, which was reverse-engineered from ``odin_ros_driver``) and
emits a PandaSet-style processed dataset that is convenient to feed into
gsplat:

    processed/<run>/
    ├── camera/front_camera/
    │   ├── intrinsics.json     # intrinsics + distortion + static Tcl_0
    │   ├── tcl_frames.json     # per-frame online lidar<->camera extrinsics
    │   ├── poses.json          # camera poses in the world frame
    │   └── 00.jpg .. N.jpg     # RGB images
    └── lidar/
        ├── poses.json          # lidar poses in the world frame
        └── 00.pkl.gz .. N.pkl.gz  # point clouds (N,7) float32 x,y,z,r,g,b,a

Public API
----------
- :func:`odin_parser.process.process_dataset` - run the full pipeline.
- :class:`odin_parser.dataset.OdinDataset` - read a processed dataset back.
- Command line: ``odin-parser info|process|preview ...`` (see ``cli.py``).
"""

from . import align, calib, camera, geometry, streams, sync, visualize
from .dataset import FrameRecord, OdinDataset
from .process import ProcessOptions, process_dataset
from .recording import OdinRecording
from .version import __version__

__all__ = [
    "FrameRecord",
    "OdinDataset",
    "OdinRecording",
    "ProcessOptions",
    "__version__",
    "align",
    "calib",
    "camera",
    "geometry",
    "process_dataset",
    "streams",
    "sync",
    "visualize",
]
