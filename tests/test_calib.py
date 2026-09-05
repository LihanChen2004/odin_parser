"""Unit tests for the calibration file parsers."""

import numpy as np

from odin_parser.calib import (
    parse_info,
    parse_intrinsics,
    parse_online_calib,
)


def test_parse_intrinsics(tmp_path):
    txt = """
Tcl_0: [0.012810, -0.999810, 0.014700, -0.007140,
        -0.004270, -0.014760, -0.999880, 0.000070,
        0.999910, 0.012750, -0.004460, -0.000840,
        0.000000, 0.000000, 0.000000, 1.000000]
cam_0:
   image_width: 1600
   image_height: 1296
   k2: 0.005754
   k3: -0.034050
   k4: 0.008453
   k5: 0.010385
   k6: -0.011936
   k7: 0.001663
   p1: 0.000000
   p2: 0.000000
   A11: 732.462635
   A12: -0.804103
   A22: 730.267891
   u0: 818.996746
   v0: 678.971288
"""
    p = tmp_path / "cam_in_ex.txt"
    p.write_text(txt)
    intr = parse_intrinsics(p)
    assert intr.image_width == 1600
    assert intr.fx == 732.462635
    assert intr.skew == -0.804103
    assert intr.k == (0.005754, -0.034050, 0.008453, 0.010385, -0.011936, 0.001663)
    assert intr.Tcl_0_static.shape == (4, 4)
    assert np.allclose(intr.Tcl_0_static[3], [0, 0, 0, 1])


def test_parse_online_calib(tmp_path):
    yaml = """frame:
    index: 0
    timestamp: 147.9978623360
    cam_num: 1
    Tcl_0: [
        0.0129156837, -0.9998075020, 0.0147696983, -0.0071421617,
        -0.0043803903, -0.0148273629, -0.9998804736, 0.0000487236,
        0.9999069943, 0.0128494429, -0.0045710526, -0.0009146357,
        0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000
    ]

    body_T_lidar: !!opencv-matrix
       rows: 4
       cols: 4
       dt: d
       data: [ 0.9999476881,-0.0093301376,0.0041916017,-0.0281796210,
              0.0093983204,0.9998188265,-0.0165525138,0.0240821385,
              -0.0040364050,0.0165910419,0.9998542118,0.0345644405,
              0.0000000000,0.0000000000,0.0000000000,1.0000000000]

frame:
    index: 1
    timestamp: 148.0953380550
    cam_num: 1
    Tcl_0: [
        0.0128446011, -0.9998072227, 0.0148503726, -0.0071482028,
        -0.0044424456, -0.0149085111, -0.9998789932, 0.0000078129,
        0.9999076362, 0.0127770748, -0.0046330830, -0.0008783638,
        0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000
    ]

    body_T_lidar: !!opencv-matrix
       rows: 4
       cols: 4
       dt: d
       data: [ 1.0,0.0,0.0,0.0,
              0.0,1.0,0.0,0.0,
              0.0,0.0,1.0,0.0,
              0.0,0.0,0.0,1.0]
"""
    p = tmp_path / "calib_online.yaml"
    p.write_text(yaml)
    frames = parse_online_calib(p)
    assert set(frames.keys()) == {0, 1}
    f0 = frames[0]
    assert f0.timestamp == 147.9978623360
    assert f0.Tcl_0.shape == (4, 4)
    assert np.allclose(
        f0.body_T_lidar[:3, 3], [-0.0281796210, 0.0240821385, 0.0345644405]
    )
    assert np.allclose(frames[1].body_T_lidar, np.eye(4))


def test_parse_info(tmp_path):
    p = tmp_path / "info.txt"
    p.write_text("device=OdinOne\npointcloud=xyzrgbi\ncreated_at=20260424_231347\n")
    info = parse_info(p)
    assert info["device"] == "OdinOne"
    assert info["created_at"] == "20260424_231347"
