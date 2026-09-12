"""Parsers for the Odin1 calibration files.

Two calibration files ship with each recording:

``image/cam_in_ex.txt`` (static, written by the device)
    ``Tcl_0``     - 4x4 extrinsic matrix, **lidar -> camera** (camera pose
                    expressed in the lidar frame).
    ``cam_0``     - polynomial (equidistant) camera intrinsics:
                    ``image_width``, ``image_height``, ``A11``/``A12``/``A22``
                    (K = [[A11, A12, u0], [0, A22, v0], [0, 0, 1]]), ``u0``,
                    ``v0`` and distortion coefficients ``k2..k7``.

``calib_online.yaml`` (per-frame online extrinsics, ~10 Hz)
    One ``frame:`` block per WIWC update:
        ``index``       - frame index (== SLAM pose / cloud index)
        ``timestamp``   - seconds
        ``Tcl_0``       - 4x4 **lidar -> camera** matrix (online value)
        ``body_T_lidar`` - 4x4 **lidar -> imu** matrix (online value)

Coordinate semantics were validated empirically against the recorded data
(reprojection + color-coherence test) and the ROS driver sources - see
``docs/coordinates-and-calibration.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .geometry import Mat4

Mat4x4 = Mat4

# Polynomial distortion coefficients (k2..k7) used when a calibration file does
# not provide them.
DEFAULT_DISTORTION: tuple[float, ...] = (0.0,) * 6


@dataclass(frozen=True)
class Intrinsics:
    """Static polynomial (equidistant) camera intrinsics.

    ``k`` holds the six polynomial coefficients ``[k2, k3, k4, k5, k6, k7]``
    (the device does not report ``k1``), applied to increasing powers of the
    polar angle::

        theta_d = theta + k2*theta^2 + k3*theta^3 + ... + k7*theta^7

    The driver uses ``k2..k7`` exactly as listed here.
    """

    name: str = "front_camera"
    image_width: int = 1600
    image_height: int = 1296
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    skew: float = 0.0
    k: tuple[float, ...] = DEFAULT_DISTORTION
    Tcl_0_static: np.ndarray | None = None

    @property
    def A(self) -> np.ndarray:
        """K matrix in driver notation: [[A11, A12, u0],[0, A22, v0],[0,0,1]]."""
        return np.array(
            [[self.fx, self.skew, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": "polynomial_equidistant",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "A11": self.fx,
            "A12": self.skew,
            "A22": self.fy,
            "u0": self.cx,
            "v0": self.cy,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_coefficients": list(self.k),
            "K": self.A.tolist(),
            "Tcl_0_static": (
                None if self.Tcl_0_static is None else self.Tcl_0_static.tolist()
            ),
        }


@dataclass(frozen=True)
class CalibFrame:
    """One online extrinsics update."""

    index: int
    timestamp: float
    Tcl_0: np.ndarray  # lidar -> camera
    body_T_lidar: np.ndarray  # lidar -> imu

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "Tcl_0": self.Tcl_0.tolist(),
            "body_T_lidar": self.body_T_lidar.tolist(),
        }


def _opencv_matrix(loader: yaml.SafeLoader, node) -> np.ndarray:
    """Handle ``!!opencv-matrix`` nodes (rows/cols/dt/data)."""
    value = loader.construct_mapping(node, deep=True)
    rows, cols = int(value["rows"]), int(value["cols"])
    return np.asarray(value["data"], dtype=np.float64).reshape(rows, cols)


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix)


def parse_intrinsics(path: str | Path) -> Intrinsics:
    """Parse ``cam_in_ex.txt`` (static intrinsics + static Tcl_0)."""
    text = Path(path).read_text()
    doc = yaml.safe_load(text)
    cam = doc["cam_0"]
    tcl = np.asarray(doc["Tcl_0"], dtype=np.float64).reshape(4, 4)
    k = tuple(float(cam.get(key, 0.0)) for key in ("k2", "k3", "k4", "k5", "k6", "k7"))
    return Intrinsics(
        image_width=int(cam["image_width"]),
        image_height=int(cam["image_height"]),
        fx=float(cam["A11"]),
        fy=float(cam["A22"]),
        cx=float(cam["u0"]),
        cy=float(cam["v0"]),
        skew=float(cam["A12"]),
        k=k,
        Tcl_0_static=tcl,
    )


def parse_online_calib(path: str | Path) -> dict[int, CalibFrame]:
    """Parse ``calib_online.yaml`` into {frame_index: CalibFrame}.

    The file repeats top-level ``frame:`` keys which plain YAML would collapse,
    so blocks are split manually before parsing.  ``frame.index`` matches the
    SLAM pose / cloud index 1:1 (verified: identical timestamps).
    """
    text = Path(path).read_text()
    # Normalise line endings so the block split below also matches CRLF files.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on lines that begin a new frame block.  Re-indent each block under
    # a synthetic "frame:" key so its nested fields stay valid YAML.
    blocks = re.split(r"^frame:\n", text, flags=re.MULTILINE)
    frames: dict[int, CalibFrame] = {}
    for block in blocks:
        block = block.rstrip()
        if not block.strip():
            continue
        doc = yaml.safe_load("frame:\n" + block)
        if not isinstance(doc, dict) or "frame" not in doc:
            continue
        doc = doc["frame"]
        if "index" not in doc:
            continue
        idx = int(doc["index"])
        tcl = np.asarray(doc["Tcl_0"], dtype=np.float64).reshape(4, 4)
        til = np.asarray(doc["body_T_lidar"], dtype=np.float64).reshape(4, 4)
        frames[idx] = CalibFrame(
            index=idx,
            timestamp=float(doc["timestamp"]),
            Tcl_0=tcl,
            body_T_lidar=til,
        )
    return frames


def parse_info(path: str | Path) -> dict:
    """Parse ``image/info.txt`` (simple key=value lines)."""
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out
