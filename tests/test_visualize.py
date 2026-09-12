"""Smoke tests for the projection and rendering helpers."""

import numpy as np

from odin_parser.camera import PolynomialCamera
from odin_parser.visualize import project_world_points, render_depth, render_overlay

CAM = PolynomialCamera(fx=500.0, fy=500.0, cx=100.0, cy=100.0)
IMG = np.full((200, 200, 3), 128, dtype=np.uint8)
# One point 3 m in front of the camera, already in the world frame.
WORLD = np.array([[0.0, 0.0, 3.0, 255.0, 0.0, 0.0, 200.0]], dtype=np.float32)
T = np.eye(4)  # camera at the world origin, looking down +Z


def test_project_world_points_hits_principal_point():
    proj = project_world_points(WORLD, T, CAM)
    assert proj.u.size == 1
    assert np.isclose(proj.u[0], CAM.cx)
    assert np.isclose(proj.v[0], CAM.cy)


def test_render_overlay_and_depth_shapes():
    overlay = render_overlay(IMG, WORLD, T, CAM)
    depth = render_depth(IMG, WORLD, T, CAM)
    assert overlay.shape == IMG.shape and overlay.dtype == np.uint8
    assert depth.shape == IMG.shape and depth.dtype == np.uint8


def test_render_with_no_points_returns_original_image():
    empty = np.zeros((0, 7), dtype=np.float32)
    np.testing.assert_array_equal(render_overlay(IMG, empty, T, CAM), IMG)
    np.testing.assert_array_equal(render_depth(IMG, empty, T, CAM), IMG)
