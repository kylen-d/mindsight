"""Tests for depth estimation integration -- sample_depth_patch, depth-aware
snap_score, and hit_event depth enrichment."""

import numpy as np
import pytest

from mindsight.utils.geometry import sample_depth_patch
from mindsight.GazeTracking.gaze_processing import (
    compute_ray_intersections,
    snap_score,
)
from mindsight.pipeline_config import GazeConfig


# ── sample_depth_patch ──────────────────────────────────────────────────────


class TestSampleDepthPatch:

    def test_center_returns_median(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        # Place a known pattern: 5x5 patch around (50, 50)
        depth[48:53, 48:53] = 0.8
        depth[50, 50] = 0.9  # outlier in center
        result = sample_depth_patch(depth, 50, 50, radius=2)
        # Median of 25 values: 24 at 0.8, 1 at 0.9 -> median = 0.8
        assert result == pytest.approx(0.8, abs=1e-4)

    def test_edge_clamps_to_bounds(self):
        depth = np.full((10, 10), 0.5, dtype=np.float32)
        # Sample at corner (0, 0) with radius=2 -> clamped to (0:3, 0:3)
        result = sample_depth_patch(depth, 0, 0, radius=2)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_out_of_bounds_returns_fallback(self):
        depth = np.full((10, 10), 0.3, dtype=np.float32)
        # Completely out of bounds
        result = sample_depth_patch(depth, -100, -100, radius=2)
        assert result == pytest.approx(0.5)  # fallback

    def test_single_pixel_radius_zero(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        depth[5, 5] = 0.7
        result = sample_depth_patch(depth, 5, 5, radius=0)
        assert result == pytest.approx(0.7, abs=1e-4)


# ── snap_score with depth ──────────────────────────────────────────────────


def _make_obj(x1, y1, x2, y2, cls='cup', depth_median=None):
    """Helper to create a dict-like detection object."""
    d = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
         'class_name': cls, 'conf': 0.9}
    if depth_median is not None:
        d['depth_median'] = depth_median
    return d


class TestSnapScoreDepth:

    def test_depth_prefers_matching_depth_far(self):
        """When gaze endpoint depth is far, the far object should win."""
        # Two overlapping objects at different depths
        near_obj = _make_obj(90, 90, 110, 110, depth_median=0.2)  # near
        far_obj = _make_obj(95, 95, 115, 115, depth_median=0.8)   # far

        # Depth map: gaze endpoint region is far (0.8), with variation
        # so depth_range > epsilon
        depth_map = np.full((200, 200), 0.8, dtype=np.float32)
        depth_map[0:50, :] = 0.1  # near region to create depth range

        origin = np.array([50.0, 50.0])
        direction = np.array([1.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        fallback = origin + direction * 100

        # Without depth: both objects are similarly close to the ray
        _, found_no_depth, obj_no_depth, _ = snap_score(
            origin, direction, [near_obj, far_obj], fallback,
            snap_dist=500, w_dist=1.0, w_angle=0.0, w_depth=0.0,
            gate_angle_deg=180)

        # With depth: far object should score better (lower penalty)
        _, found_depth, obj_depth, _ = snap_score(
            origin, direction, [near_obj, far_obj], fallback,
            snap_dist=500, w_dist=0.2, w_angle=0.0,
            w_depth=1.0, depth_map=depth_map,
            gaze_endpoint=fallback,
            gate_angle_deg=180)

        assert found_depth
        assert obj_depth['depth_median'] == pytest.approx(0.8)

    def test_depth_prefers_matching_depth_near(self):
        """When gaze endpoint depth is near, the near object should win."""
        near_obj = _make_obj(90, 90, 110, 110, depth_median=0.2)
        far_obj = _make_obj(95, 95, 115, 115, depth_median=0.8)

        # Depth map: gaze endpoint region is near (0.2)
        depth_map = np.full((200, 200), 0.2, dtype=np.float32)
        # But make the rest of the map varied so depth_range > epsilon
        depth_map[0:50, :] = 0.9

        origin = np.array([50.0, 50.0])
        direction = np.array([1.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        fallback = origin + direction * 100

        _, found, obj, _ = snap_score(
            origin, direction, [near_obj, far_obj], fallback,
            snap_dist=500, w_dist=0.2, w_angle=0.0,
            w_depth=1.0, depth_map=depth_map,
            gaze_endpoint=fallback,
            gate_angle_deg=180)

        assert found
        assert obj['depth_median'] == pytest.approx(0.2)

    def test_flat_depth_falls_back_to_2d(self):
        """When depth range < epsilon, depth_factor should be 0 for all."""
        obj_a = _make_obj(90, 90, 110, 110, depth_median=0.5)
        obj_b = _make_obj(95, 95, 115, 115, depth_median=0.5)

        # Flat depth map
        depth_map = np.full((200, 200), 0.5, dtype=np.float32)

        origin = np.array([50.0, 50.0])
        direction = np.array([1.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        fallback = origin + direction * 100

        # With depth weight active but flat depth map, result should be
        # the same as without depth
        _, found_depth, _, score_depth = snap_score(
            origin, direction, [obj_a, obj_b], fallback,
            snap_dist=500, w_dist=1.0, w_angle=0.0,
            w_depth=1.0, depth_map=depth_map,
            gaze_endpoint=fallback,
            gate_angle_deg=180)

        _, found_no, _, score_no = snap_score(
            origin, direction, [obj_a, obj_b], fallback,
            snap_dist=500, w_dist=1.0, w_angle=0.0,
            w_depth=0.0,
            gate_angle_deg=180)

        # Scores should be identical since depth_factor = 0
        assert score_depth == pytest.approx(score_no, abs=1e-4)

    def test_depth_zero_weight_no_effect(self):
        """w_depth=0 should produce identical results to no depth."""
        obj = _make_obj(90, 90, 110, 110, depth_median=0.2)
        depth_map = np.full((200, 200), 0.8, dtype=np.float32)

        origin = np.array([50.0, 50.0])
        direction = np.array([1.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        fallback = origin + direction * 100

        _, _, _, score_with = snap_score(
            origin, direction, [obj], fallback,
            snap_dist=500, w_dist=1.0, w_angle=0.0,
            w_depth=0.0, depth_map=depth_map,
            gaze_endpoint=fallback,
            gate_angle_deg=180)

        _, _, _, score_without = snap_score(
            origin, direction, [obj], fallback,
            snap_dist=500, w_dist=1.0, w_angle=0.0,
            gate_angle_deg=180)

        assert score_with == pytest.approx(score_without, abs=1e-6)


# ── compute_ray_intersections with depth ───────────────────────────────────


class TestHitEventsDepthEnrichment:

    def test_depth_fields_present_when_depth_map_provided(self):
        """hit_events should contain depth_median and depth_at_gaze."""
        gaze_cfg = GazeConfig(gaze_cone_angle=0.0, hit_conf_gate=0.0,
                              detect_extend=0.0, forward_gaze_threshold=0.0)

        # Object in the ray path
        obj = _make_obj(80, 80, 120, 120, depth_median=0.6)
        objects = [obj]

        # Ray that hits the object
        origin = np.array([50.0, 100.0])
        ray_end = np.array([150.0, 100.0])
        persons_gaze = [(origin, ray_end, (0.1, 0.1))]
        face_confs = [0.9]
        face_track_ids = [0]
        face_objs = []

        depth_map = np.full((200, 200), 0.5, dtype=np.float32)
        depth_map[80:120, 80:120] = 0.6  # object region

        _, hits, hit_events = compute_ray_intersections(
            persons_gaze, face_confs, face_track_ids,
            face_objs, objects, gaze_cfg,
            depth_map=depth_map)

        assert len(hit_events) > 0
        ev = hit_events[0]
        assert 'depth_median' in ev
        assert 'depth_at_gaze' in ev
        assert ev['depth_median'] == pytest.approx(0.6, abs=0.1)

    def test_no_depth_fields_without_depth_map(self):
        """hit_events should NOT contain depth fields when no depth_map."""
        gaze_cfg = GazeConfig(gaze_cone_angle=0.0, hit_conf_gate=0.0,
                              detect_extend=0.0, forward_gaze_threshold=0.0)

        obj = _make_obj(80, 80, 120, 120)
        origin = np.array([50.0, 100.0])
        ray_end = np.array([150.0, 100.0])

        _, _, hit_events = compute_ray_intersections(
            [(origin, ray_end, (0.1, 0.1))], [0.9], [0],
            [], [obj], gaze_cfg,
            depth_map=None)

        assert len(hit_events) > 0
        assert 'depth_median' not in hit_events[0]
        assert 'depth_at_gaze' not in hit_events[0]
