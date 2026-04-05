"""Tests for utils/geometry.py -- ray-geometry and spatial utilities."""

import numpy as np
import pytest

from ms.utils.geometry import (
    bbox_center,
    extend_ray,
    pitch_yaw_to_2d,
    ray_hits_box,
    ray_hits_cone,
)

# ── pitch_yaw_to_2d ─────────────────────────────────────────────────────────

class TestPitchYawTo2D:
    """Tests for pitch_yaw_to_2d conversion."""

    def test_zero_angles_returns_zero_vector(self):
        """pitch=0, yaw=0 produces a near-zero vector (norm < epsilon)."""
        d = pitch_yaw_to_2d(0.0, 0.0)
        assert d.shape == (2,)
        # Both sin(0)*cos(0)=0 and sin(0)=0, so d is zero; function returns d as-is
        np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-9)

    def test_pitch_only(self):
        """Non-zero pitch with yaw=0 gives a direction along the first axis."""
        d = pitch_yaw_to_2d(np.pi / 4, 0.0)
        assert d.shape == (2,)
        # yaw=0 -> d[1] = -sin(0) = 0, d[0] = -sin(pi/4)*cos(0) < 0
        assert d[1] == pytest.approx(0.0, abs=1e-9)
        assert d[0] < 0  # -sin(pi/4) is negative

    def test_yaw_only(self):
        """Non-zero yaw with pitch=0 gives a direction along the second axis."""
        d = pitch_yaw_to_2d(0.0, np.pi / 4)
        # pitch=0 -> d[0] = -sin(0)*cos(pi/4) = 0
        # d[1] = -sin(pi/4) < 0
        assert d[0] == pytest.approx(0.0, abs=1e-9)
        assert d[1] < 0

    def test_normalized(self):
        """Result is unit-length for non-degenerate inputs."""
        d = pitch_yaw_to_2d(0.3, 0.5)
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)

    def test_negative_angles(self):
        """Negative angles are handled correctly."""
        d = pitch_yaw_to_2d(-np.pi / 6, -np.pi / 3)
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)


# ── ray_hits_box ─────────────────────────────────────────────────────────────

class TestRayHitsBox:
    """Tests for Liang-Barsky segment-AABB intersection."""

    def test_segment_through_box(self):
        """Segment that clearly crosses the box returns True."""
        assert ray_hits_box(
            np.array([0.0, 5.0]), np.array([20.0, 5.0]),
            5, 0, 15, 10,
        )

    def test_segment_misses_box(self):
        """Segment entirely above the box returns False."""
        assert not ray_hits_box(
            np.array([0.0, 20.0]), np.array([20.0, 20.0]),
            5, 0, 15, 10,
        )

    def test_segment_starts_inside_box(self):
        """Segment starting inside the box always hits."""
        assert ray_hits_box(
            np.array([10.0, 5.0]), np.array([30.0, 5.0]),
            5, 0, 15, 10,
        )

    def test_segment_ends_inside_box(self):
        """Segment ending inside the box always hits."""
        assert ray_hits_box(
            np.array([0.0, 5.0]), np.array([10.0, 5.0]),
            5, 0, 15, 10,
        )

    def test_segment_entirely_inside(self):
        """Segment fully contained in the box."""
        assert ray_hits_box(
            np.array([6.0, 1.0]), np.array([14.0, 9.0]),
            5, 0, 15, 10,
        )

    def test_zero_length_segment_inside(self):
        """Degenerate (zero-length) segment inside the box."""
        assert ray_hits_box(
            np.array([10.0, 5.0]), np.array([10.0, 5.0]),
            5, 0, 15, 10,
        )

    def test_zero_length_segment_outside(self):
        """Degenerate (zero-length) segment outside the box."""
        assert not ray_hits_box(
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            5, 0, 15, 10,
        )

    def test_diagonal_through_box(self):
        """Diagonal segment crossing the box."""
        assert ray_hits_box(
            np.array([0.0, 0.0]), np.array([20.0, 20.0]),
            5, 5, 15, 15,
        )

    def test_segment_just_touches_corner(self):
        """Segment ending exactly at box corner."""
        assert ray_hits_box(
            np.array([0.0, 0.0]), np.array([5.0, 0.0]),
            5, 0, 15, 10,
        )

    def test_segment_too_short(self):
        """Segment pointing at box but ending before reaching it."""
        assert not ray_hits_box(
            np.array([0.0, 5.0]), np.array([3.0, 5.0]),
            5, 0, 15, 10,
        )

    def test_vertical_segment_through_box(self):
        """Purely vertical segment crossing the box."""
        assert ray_hits_box(
            np.array([10.0, -5.0]), np.array([10.0, 15.0]),
            5, 0, 15, 10,
        )

    def test_vertical_segment_parallel_outside(self):
        """Vertical segment parallel to box side but outside."""
        assert not ray_hits_box(
            np.array([0.0, -5.0]), np.array([0.0, 15.0]),
            5, 0, 15, 10,
        )


# ── ray_hits_cone ────────────────────────────────────────────────────────────

class TestRayHitsCone:
    """Tests for gaze-cone / bounding-box intersection."""

    def test_origin_inside_box(self):
        """Origin inside the box always returns True."""
        origin = np.array([10.0, 10.0])
        direction = np.array([1.0, 0.0])
        assert ray_hits_cone(origin, direction, 5, 5, 15, 15, 10.0)

    def test_box_directly_ahead_narrow_cone(self):
        """Box centred on the gaze direction, narrow cone."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        assert ray_hits_cone(origin, direction, 50, -5, 60, 5, 5.0)

    def test_box_behind_origin(self):
        """Box behind the origin should not be hit (dot < 0)."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        assert not ray_hits_cone(origin, direction, -60, -5, -50, 5, 10.0)

    def test_box_far_off_axis(self):
        """Box at 90 degrees off-axis with a narrow cone."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        assert not ray_hits_cone(origin, direction, -5, 50, 5, 60, 5.0)

    def test_wide_cone_catches_off_axis_box(self):
        """Wide cone (90 deg half-angle) catches a box off to the side."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        assert ray_hits_cone(origin, direction, 10, 8, 20, 12, 89.0)

    def test_zero_half_angle_degenerates_to_ray(self):
        """Half-angle of 0 should act like a thin ray (cos_thresh=1)."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        # Box directly on the axis
        assert ray_hits_cone(origin, direction, 5, -1, 10, 1, 0.0)


# ── extend_ray ───────────────────────────────────────────────────────────────

class TestExtendRay:
    """Tests for extending a ray to a given length."""

    def test_basic_extension(self):
        """Ray extended to 100 px has the correct norm."""
        result = extend_ray([0, 0], [3, 4], length=100.0)
        np.testing.assert_allclose(np.linalg.norm(result), 100.0, atol=1e-6)

    def test_direction_preserved(self):
        """Extended ray preserves the original direction."""
        origin = np.array([10.0, 20.0])
        end = np.array([13.0, 24.0])
        result = extend_ray(origin, end, length=50.0)
        d_original = end - origin
        d_result = result - origin
        cos_angle = np.dot(d_original, d_result) / (
            np.linalg.norm(d_original) * np.linalg.norm(d_result)
        )
        assert cos_angle == pytest.approx(1.0, abs=1e-6)

    def test_zero_length_ray(self):
        """Zero-length ray (origin==end) returns end unchanged."""
        origin = np.array([5.0, 5.0])
        result = extend_ray(origin, origin, length=100.0)
        np.testing.assert_array_equal(result, origin)

    def test_very_short_ray(self):
        """Very short ray is properly extended."""
        result = extend_ray([0, 0], [1e-8, 1e-8], length=100.0)
        # Below the 1e-6 threshold, so returns end as-is
        np.testing.assert_allclose(result, [1e-8, 1e-8], atol=1e-12)

    def test_default_length_uses_constant(self):
        """Calling without explicit length uses the constant from constants.py."""
        from ms.constants import RAY_EXT_LENGTH
        result = extend_ray([0, 0], [1, 0])
        expected_len = np.linalg.norm(result)
        assert expected_len == pytest.approx(RAY_EXT_LENGTH, rel=1e-6)

    def test_negative_direction(self):
        """Negative direction components work correctly."""
        result = extend_ray([10, 10], [7, 6], length=50.0)
        d = result - np.array([10.0, 10.0])
        assert d[0] < 0
        assert d[1] < 0
        assert np.linalg.norm(d) == pytest.approx(50.0, abs=1e-6)


# ── bbox_center ──────────────────────────────────────────────────────────────

class TestBboxCenter:
    """Tests for computing bounding-box centres."""

    def test_simple_bbox(self):
        """Centre of a simple bbox."""
        obj = {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 20}
        c = bbox_center(obj)
        np.testing.assert_array_equal(c, [5.0, 10.0])

    def test_negative_coords(self):
        """Bbox with negative coordinates."""
        obj = {'x1': -10, 'y1': -20, 'x2': 10, 'y2': 0}
        c = bbox_center(obj)
        np.testing.assert_array_equal(c, [0.0, -10.0])

    def test_zero_area_bbox(self):
        """Degenerate bbox (point) returns the point itself."""
        obj = {'x1': 5, 'y1': 5, 'x2': 5, 'y2': 5}
        c = bbox_center(obj)
        np.testing.assert_array_equal(c, [5.0, 5.0])

    def test_float_coords(self):
        """Float coordinates produce correct centre."""
        obj = {'x1': 1.5, 'y1': 2.5, 'x2': 3.5, 'y2': 4.5}
        c = bbox_center(obj)
        np.testing.assert_allclose(c, [2.5, 3.5])

    def test_return_type(self):
        """Result is a float numpy array."""
        obj = {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}
        c = bbox_center(obj)
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float64
