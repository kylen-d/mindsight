"""Tests for GazeTracking/gaze_processing.py -- smoother, snap, lock-on."""

import numpy as np
import pytest

from mindsight.GazeTracking.gaze_processing import (
    GazeLockTracker,
    GazeSmootherReID,
    SnapTemporalState,
    snap_score,
)


# ── GazeSmootherReID ────────────────────────────────────────────────────────


class TestGazeSmootherReID:

    def test_single_face_returns_smoothed(self):
        s = GazeSmootherReID(alpha=0.5)
        result = s.update([(np.array([100.0, 200.0]), 0.3, -0.2, None)])
        assert len(result) == 1
        pitch, yaw, tid = result[0]
        assert isinstance(tid, int)
        # First frame: smoothed == raw (no prior state)
        assert pitch == pytest.approx(0.3, abs=1e-4)
        assert yaw == pytest.approx(-0.2, abs=1e-4)

    def test_smoothing_averages_over_frames(self):
        s = GazeSmootherReID(alpha=0.5)
        # Frame 1: pitch=1.0, yaw=0.0
        s.update([(np.array([100.0, 100.0]), 1.0, 0.0, None)])
        # Frame 2: pitch=0.0, yaw=0.0 -- smoothed should be ~0.5
        result = s.update([(np.array([100.0, 100.0]), 0.0, 0.0, None)])
        pitch, yaw, tid = result[0]
        assert pitch == pytest.approx(0.5, abs=0.1)

    def test_stable_track_id_across_frames(self):
        s = GazeSmootherReID(alpha=0.3)
        r1 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        r2 = s.update([(np.array([102.0, 101.0]), 0.1, 0.1, None)])
        assert r1[0][2] == r2[0][2]  # same track ID

    def test_new_face_gets_new_id(self):
        s = GazeSmootherReID(alpha=0.3, max_dist=50)
        r1 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        # Face at very different location
        r2 = s.update([(np.array([500.0, 500.0]), 0.2, 0.2, None)])
        assert r1[0][2] != r2[0][2]

    def test_empty_input_returns_empty(self):
        s = GazeSmootherReID()
        assert s.update([]) == []

    def test_grace_frames_revives_lost_track(self):
        s = GazeSmootherReID(alpha=0.5, grace_frames=5)
        r1 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        tid1 = r1[0][2]
        # Disappear for 1 frame
        s.update([])
        # Reappear at same location
        r3 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        assert r3[0][2] == tid1  # same track revived

    def test_grace_frames_zero_no_revival(self):
        s = GazeSmootherReID(alpha=0.5, grace_frames=0)
        r1 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        tid1 = r1[0][2]
        s.update([])
        r3 = s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        # With grace=0, new track ID is assigned
        assert r3[0][2] != tid1

    def test_multiple_faces_tracked_independently(self):
        s = GazeSmootherReID(alpha=0.5, max_dist=200)
        faces = [
            (np.array([100.0, 100.0]), 0.1, 0.2, None),
            (np.array([400.0, 400.0]), 0.3, 0.4, None),
        ]
        result = s.update(faces)
        assert len(result) == 2
        assert result[0][2] != result[1][2]  # different IDs


# ── snap_score ─────────────────────────────────────────────────────────────


class _FakeDetection(dict):
    """Lightweight Detection stand-in for unit tests.

    Supports both attribute and dict-style access (like the real Detection
    dataclass) without importing heavyweight YOLO dependencies.
    """
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _make_obj(x1, y1, x2, y2, cls="obj"):
    """Create a Detection-like object for scoring tests."""
    return _FakeDetection(
        class_name=cls, cls_id=0, conf=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2, ghost=False, _face_idx=None)


class TestSnapScore:

    def test_on_axis_intersection_scores_lowest(self):
        """Object directly on ray with bbox intersection gets a very low score."""
        origin = np.array([100.0, 100.0])
        direction = np.array([1.0, 0.0])  # pointing right
        obj = _make_obj(200, 80, 300, 120)  # directly ahead, ray intersects
        face_ctr = np.array([100.0, 100.0])

        ctr, found, _, score = snap_score(
            origin, direction, [obj], origin,
            snap_dist=200.0, face_center=face_ctr,
            frame_diag=1000.0, quality_thresh=2.0)
        assert found is True
        assert score < 0.0  # intersection bonus should push it negative

    def test_off_axis_penalized(self):
        """Object at 45 deg scores worse than one at 10 deg."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])

        obj_near = _make_obj(180, -20, 220, 20)   # ~0 deg off-axis
        obj_far = _make_obj(180, 160, 220, 200)   # ~45 deg off-axis

        _, _, _, score_near = snap_score(
            origin, direction, [obj_near], origin,
            snap_dist=300.0, face_center=face_ctr,
            frame_diag=1000.0, quality_thresh=2.0)
        _, _, _, score_far = snap_score(
            origin, direction, [obj_far], origin,
            snap_dist=300.0, face_center=face_ctr,
            frame_diag=1000.0, quality_thresh=2.0)
        assert score_near < score_far

    def test_beyond_gate_angle_excluded(self):
        """Object well beyond gate angle is not returned."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])
        # Object almost directly below: center at (10, 500) -> ~89 degrees
        obj = _make_obj(0, 480, 20, 520)

        _, found, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=600.0, gate_angle_deg=60.0,
            head_blend=0.0,  # pure gaze direction for clarity
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert found is False

    def test_no_objects_returns_fallback(self):
        origin = np.array([100.0, 100.0])
        direction = np.array([1.0, 0.0])
        fallback = np.array([500.0, 100.0])

        ctr, found, obj, _ = snap_score(
            origin, direction, [], fallback,
            snap_dist=200.0, quality_thresh=2.0)
        assert found is False
        assert obj is None
        assert np.allclose(ctr, fallback)

    def test_behind_face_excluded(self):
        """Object behind the face (t <= 0) never selected."""
        origin = np.array([300.0, 100.0])
        direction = np.array([1.0, 0.0])  # pointing right
        obj = _make_obj(50, 80, 150, 120)  # to the left = behind

        _, found, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=500.0, quality_thresh=2.0)
        assert found is False

    def test_temporal_bonus_creates_stickiness(self):
        """With prev_target_key matching object A, A wins even when B is slightly closer."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])

        obj_a = _make_obj(190, -20, 230, 20)  # slightly farther
        obj_b = _make_obj(180, -20, 220, 20)  # slightly closer

        tracker = SnapTemporalState(grid_px=40)
        key_a = tracker.key_for(np.array([210.0, 0.0]))

        # Without temporal bonus, B should win (closer)
        _, _, matched_no_t, _ = snap_score(
            origin, direction, [obj_a, obj_b], origin,
            snap_dist=300.0, w_temporal=0.0,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)

        # With temporal bonus for A, A should win
        _, _, matched_with_t, _ = snap_score(
            origin, direction, [obj_a, obj_b], origin,
            snap_dist=300.0, w_temporal=1.0,
            prev_target_key=key_a, _key_fn=tracker.key_for,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert matched_with_t is not None
        # A's center is around (210, 0)
        from mindsight.utils.geometry import bbox_center
        a_ctr = bbox_center(obj_a)
        assert np.allclose(bbox_center(matched_with_t), a_ctr)

    def test_quality_threshold_rejects_poor_match(self):
        """Single distant off-axis object exceeds quality threshold."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])
        # Object barely within snap_dist but significantly off-axis
        obj = _make_obj(180, 120, 220, 160)

        _, found, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=200.0, quality_thresh=0.1,
            face_center=face_ctr, frame_diag=1000.0)
        assert found is False

    def test_confidence_scales_snap_dist(self):
        """Low gaze_conf reduces effective snap distance."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])
        # Object at edge of normal snap_dist
        obj = _make_obj(180, 100, 220, 140)

        # High confidence: should find it
        _, found_hi, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=150.0, gaze_conf=1.0,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)

        # Low confidence: effective dist = 150 * 0.2 = 30, object too far
        _, found_lo, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=150.0, gaze_conf=0.0,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert found_hi is True
        assert found_lo is False

    def test_head_blend_zero_uses_pure_gaze(self):
        """With head_blend=0, angular penalty is purely gaze-based."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 500.0])  # face center far off
        obj = _make_obj(180, -20, 220, 20)  # on gaze axis

        _, found, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=300.0, head_blend=0.0,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert found is True

    def test_intersection_bonus_tips_preference(self):
        """Two equidistant objects, one intersected, the intersected one wins."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])

        obj_hit = _make_obj(180, -20, 220, 20)    # ray passes through
        obj_miss = _make_obj(180, 25, 220, 65)    # ray just misses

        ctr, found, matched, _ = snap_score(
            origin, direction, [obj_hit, obj_miss], origin,
            snap_dist=300.0, w_intersect=0.5,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert found is True
        from mindsight.utils.geometry import bbox_center
        assert np.allclose(ctr, bbox_center(obj_hit))

    def test_size_weight_prefers_larger(self):
        """With w_size > 0, larger object wins over equidistant smaller one."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        face_ctr = np.array([0.0, 0.0])

        obj_small = _make_obj(195, -5, 205, 5)     # 10x10
        obj_large = _make_obj(185, -25, 215, 25)   # 30x50

        _, _, matched, _ = snap_score(
            origin, direction, [obj_small, obj_large], origin,
            snap_dist=300.0, w_size=2.0,
            face_center=face_ctr, frame_diag=1000.0,
            quality_thresh=2.0)
        assert matched is not None
        from mindsight.utils.geometry import bbox_center
        assert np.allclose(bbox_center(matched), bbox_center(obj_large))


# ── SnapTemporalState ─────────────────────────────────────────────────────


class TestSnapTemporalState:

    def test_instant_engage_when_no_prior(self):
        t = SnapTemporalState(engage_frames=0)
        center = np.array([100.0, 200.0])
        result, did_snap = t.update(0, center, True)
        assert did_snap is True
        assert np.allclose(result, center)

    def test_release_after_n_frames(self):
        t = SnapTemporalState(release_frames=2)
        center = np.array([100.0, 200.0])
        t.update(0, center, True)
        # No match for release_frames
        t.update(0, None, False)
        result, did_snap = t.update(0, None, False)
        assert did_snap is False
        assert result is None

    def test_hold_during_release_window(self):
        t = SnapTemporalState(release_frames=3)
        center = np.array([100.0, 200.0])
        t.update(0, center, True)
        # First no-match frame: should still hold
        result, did_snap = t.update(0, None, False)
        assert did_snap is True
        assert np.allclose(result, center)

    def test_engage_delay(self):
        t = SnapTemporalState(engage_frames=3)
        center = np.array([100.0, 200.0])
        # First 2 frames: not yet engaged
        for _ in range(2):
            result, did_snap = t.update(0, center, True)
            assert did_snap is False
        # Third frame: engaged
        result, did_snap = t.update(0, center, True)
        assert did_snap is True
        assert np.allclose(result, center)

    def test_per_face_independence(self):
        t = SnapTemporalState()
        c1 = np.array([100.0, 100.0])
        c2 = np.array([500.0, 500.0])
        r1, s1 = t.update(0, c1, True)
        r2, s2 = t.update(1, c2, True)
        assert s1 is True
        assert s2 is True

    def test_prev_target_key(self):
        t = SnapTemporalState(grid_px=40)
        center = np.array([100.0, 200.0])
        assert t.prev_target_key(0) is None
        t.update(0, center, True)
        key = t.prev_target_key(0)
        assert key is not None
        assert key == t.key_for(center)


# ── Tip Snap Independence ─────────────────────────────────────────────────


class TestTipSnapIndependence:

    def test_tip_uses_own_distance(self):
        """Tip snap with snap_tip_dist=50 rejects objects that snap_dist=150 would accept."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        obj = _make_obj(180, 80, 220, 120)  # ~100px from ray

        # Object snap (dist=150): should find
        _, found_obj, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=150.0, quality_thresh=2.0)

        # Tip snap (dist=50): should NOT find
        _, found_tip, _, _ = snap_score(
            origin, direction, [obj], origin,
            snap_dist=50.0, quality_thresh=2.0)
        assert found_obj is True
        assert found_tip is False

    def test_tip_falls_back_to_shared(self):
        """When snap_tip_dist=-1, effective distance should equal snap_dist."""
        # This is tested at the config level -- snap_tip_dist=-1 resolves to
        # snap_dist before calling snap_score(). Just verify the resolution
        # logic matches.
        from mindsight.pipeline_config import GazeConfig
        cfg = GazeConfig(snap_dist=200.0, snap_tip_dist=-1.0,
                         snap_quality_thresh=0.8, snap_tip_quality=-1.0)
        tip_dist = cfg.snap_tip_dist if cfg.snap_tip_dist >= 0 else cfg.snap_dist
        tip_qual = cfg.snap_tip_quality if cfg.snap_tip_quality >= 0 else cfg.snap_quality_thresh
        assert tip_dist == 200.0
        assert tip_qual == 0.8


# ── GazeLockTracker ─────────────────────────────────────────────────────────


class TestGazeLockTracker:

    def _make_object(self, x1, y1, x2, y2, cls="obj"):
        """Helper to create object dicts with bbox keys."""
        return _FakeDetection(
            class_name=cls, cls_id=0, conf=0.9,
            x1=x1, y1=y1, x2=x2, y2=y2, ghost=False, _face_idx=None)

    def test_no_lock_before_dwell(self):
        t = GazeLockTracker(dwell_frames=10, lock_dist=200)
        obj = self._make_object(200, 200, 300, 300)
        # Gaze near the object but not enough frames
        origin = np.array([250.0, 250.0])
        ray_end = np.array([260.0, 260.0])
        for _ in range(5):  # less than dwell_frames
            results = t.update(
                [(origin, ray_end, (0.1, 0.1))],
                [obj])
        # Should not be locked yet
        _, obj_idx, dwell_frac = results[0]
        assert obj_idx is None or dwell_frac < 1.0

    def test_lock_engages_after_dwell(self):
        t = GazeLockTracker(dwell_frames=5, lock_dist=200)
        obj = self._make_object(200, 200, 300, 300)
        origin = np.array([250.0, 250.0])
        ray_end = np.array([260.0, 260.0])
        for _ in range(10):
            results = t.update(
                [(origin, ray_end, (0.1, 0.1))],
                [obj])
        _, obj_idx, dwell_frac = results[0]
        assert obj_idx is not None

    def test_empty_objects(self):
        t = GazeLockTracker(dwell_frames=5)
        origin = np.array([100.0, 100.0])
        ray_end = np.array([200.0, 200.0])
        results = t.update([(origin, ray_end, (0.1, 0.1))], [])
        assert len(results) == 1

    def test_empty_gaze(self):
        t = GazeLockTracker()
        results = t.update([], [])
        assert results == []


# ── Histogram scoring ───────────────────────────────────────────────────────


class TestHistogramScoring:

    def test_score_no_histogram(self):
        """With no crop, score should be purely positional distance."""
        s = GazeSmootherReID(hist_weight=0.5)
        # Need a track with a center
        s.update([(np.array([100.0, 100.0]), 0.1, 0.1, None)])
        # Access internal for unit testing
        track = list(s._tracks.values())[0]
        center = np.array([110.0, 110.0])
        score = s._score(center, None, track)
        expected_dist = np.linalg.norm(center - track['c'])
        # Without histogram, score == positional distance
        assert score == pytest.approx(expected_dist, abs=0.5)

    def test_bhattacharyya_identical_returns_zero(self):
        h = np.array([0.25, 0.25, 0.25, 0.25])
        dist = GazeSmootherReID._bhattacharyya(h, h)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_bhattacharyya_none_returns_zero(self):
        h = np.array([0.25, 0.25, 0.25, 0.25])
        assert GazeSmootherReID._bhattacharyya(None, h) == 0.0
        assert GazeSmootherReID._bhattacharyya(h, None) == 0.0
        assert GazeSmootherReID._bhattacharyya(None, None) == 0.0
