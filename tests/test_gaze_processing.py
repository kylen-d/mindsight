"""Tests for GazeTracking/gaze_processing.py -- smoother, snap, lock-on."""

import numpy as np
import pytest

from ms.GazeTracking.gaze_processing import (
    GazeLockTracker,
    GazeSmootherReID,
    SnapHysteresisTracker,
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


# ── SnapHysteresisTracker ───────────────────────────────────────────────────


class TestSnapHysteresisTracker:

    def test_snap_engages_immediately_when_no_current(self):
        t = SnapHysteresisTracker(switch_frames=3)
        center = np.array([100.0, 200.0])
        result, snapped = t.update(0, center, True)
        assert snapped is True
        assert result is not None
        assert np.allclose(result, center)

    def test_snap_holds_when_same_target(self):
        t = SnapHysteresisTracker(switch_frames=3, grid_px=40)
        center = np.array([100.0, 200.0])
        t.update(0, center, True)
        # Same grid cell
        center2 = np.array([105.0, 205.0])
        result, snapped = t.update(0, center2, True)
        assert snapped is True

    def test_snap_switches_after_switch_frames(self):
        t = SnapHysteresisTracker(switch_frames=3, grid_px=40)
        old_center = np.array([100.0, 200.0])
        new_center = np.array([500.0, 500.0])
        # Establish snap to old target
        t.update(0, old_center, True)
        # Present new candidate for switch_frames
        for _ in range(3):
            result, snapped = t.update(0, new_center, True)
        assert snapped is True
        assert np.allclose(result, new_center)

    def test_snap_releases_after_no_target(self):
        t = SnapHysteresisTracker(switch_frames=3, release_frames=2)
        center = np.array([100.0, 200.0])
        t.update(0, center, True)
        # No target for release_frames
        for _ in range(3):
            result, snapped = t.update(0, None, False)
        assert snapped is False
        assert result is None

    def test_independent_per_face(self):
        t = SnapHysteresisTracker(switch_frames=3)
        c1 = np.array([100.0, 100.0])
        c2 = np.array([500.0, 500.0])
        r1, s1 = t.update(0, c1, True)
        r2, s2 = t.update(1, c2, True)
        assert s1 is True
        assert s2 is True


# ── GazeLockTracker ─────────────────────────────────────────────────────────


class TestGazeLockTracker:

    def _make_object(self, x1, y1, x2, y2, cls="obj"):
        """Helper to create object dicts with bbox keys."""
        from ms.ObjectDetection.detection import Detection
        return Detection(
            class_name=cls, cls_id=0, conf=0.9,
            x1=x1, y1=y1, x2=x2, y2=y2,
        )

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
