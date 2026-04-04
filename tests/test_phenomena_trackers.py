"""Tests for phenomena trackers -- MutualGaze, GazeAversion, AttentionSpan."""

import numpy as np
import pytest

from Phenomena.Default.attention_span import AttentionSpanTracker
from Phenomena.Default.gaze_aversion import GazeAversionTracker
from Phenomena.Default.mutual_gaze import MutualGazeTracker

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_person_gaze(origin, direction, conf=0.9):
    """Build a (origin, ray_end, confidence) tuple matching persons_gaze format."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return (origin, direction, conf)


def _make_face_bbox(cx, cy, half_w=30, half_h=40):
    """Build a (x1, y1, x2, y2) face bbox centred at (cx, cy)."""
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _make_det(class_name, conf=0.9, x1=0, y1=0, x2=50, y2=50):
    """Build a detection dict matching the expected format."""
    return {'class_name': class_name, 'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


# ══════════════════════════════════════════════════════════════════════════════
# MutualGazeTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestMutualGazeTracker:
    """Tests for MutualGazeTracker."""

    def test_init(self):
        """Tracker initializes with empty state."""
        t = MutualGazeTracker()
        assert t.pair_counts == {}
        assert t._current_pairs == set()

    def test_two_people_looking_at_each_other(self):
        """Two people facing each other should produce a mutual gaze pair."""
        t = MutualGazeTracker()
        # Person 0 at (100, 200) looking right, Person 1 at (300, 200) looking left
        persons_gaze = [
            _make_person_gaze([100, 200], [300, 200]),   # origin, ray_end
            _make_person_gaze([300, 200], [100, 200]),
        ]
        face_bboxes = [
            _make_face_bbox(100, 200),  # Person 0's face
            _make_face_bbox(300, 200),  # Person 1's face
        ]
        result = t.update(persons_gaze=persons_gaze, face_bboxes=face_bboxes)
        assert (0, 1) in result['pairs']
        assert t.pair_counts[(0, 1)] == 1

    def test_two_people_not_looking_at_each_other(self):
        """Two people looking in the same direction should not produce mutual gaze."""
        t = MutualGazeTracker()
        # Both looking right (away from each other for person 1)
        persons_gaze = [
            _make_person_gaze([100, 200], [500, 200]),
            _make_person_gaze([300, 200], [500, 200]),
        ]
        face_bboxes = [
            _make_face_bbox(100, 200),
            _make_face_bbox(300, 200),
        ]
        result = t.update(persons_gaze=persons_gaze, face_bboxes=face_bboxes)
        assert len(result['pairs']) == 0

    def test_one_sided_gaze(self):
        """Only one person looking at the other is not mutual gaze."""
        t = MutualGazeTracker()
        persons_gaze = [
            _make_person_gaze([100, 200], [300, 200]),  # looking at person 1
            _make_person_gaze([300, 200], [500, 200]),  # looking away
        ]
        face_bboxes = [
            _make_face_bbox(100, 200),
            _make_face_bbox(300, 200),
        ]
        result = t.update(persons_gaze=persons_gaze, face_bboxes=face_bboxes)
        assert len(result['pairs']) == 0

    def test_accumulates_over_frames(self):
        """Pair counts accumulate across multiple update() calls."""
        t = MutualGazeTracker()
        persons_gaze = [
            _make_person_gaze([100, 200], [300, 200]),
            _make_person_gaze([300, 200], [100, 200]),
        ]
        face_bboxes = [
            _make_face_bbox(100, 200),
            _make_face_bbox(300, 200),
        ]
        for _ in range(5):
            t.update(persons_gaze=persons_gaze, face_bboxes=face_bboxes)
        assert t.pair_counts[(0, 1)] == 5

    def test_empty_input(self):
        """No persons produces no pairs."""
        t = MutualGazeTracker()
        result = t.update(persons_gaze=[], face_bboxes=[])
        assert result['pairs'] == set()

    def test_single_person(self):
        """Single person cannot have mutual gaze."""
        t = MutualGazeTracker()
        result = t.update(
            persons_gaze=[_make_person_gaze([100, 200], [300, 200])],
            face_bboxes=[_make_face_bbox(100, 200)],
        )
        assert result['pairs'] == set()

    def test_csv_rows_empty(self):
        """csv_rows returns empty list when no mutual gaze was detected."""
        t = MutualGazeTracker()
        assert t.csv_rows(100) == []

    def test_csv_rows_with_data(self):
        """csv_rows returns header + data rows after mutual gaze."""
        t = MutualGazeTracker()
        t.pair_counts[(0, 1)] = 10
        rows = t.csv_rows(100)
        assert len(rows) == 2  # header + 1 data row
        assert rows[0][0] == "category"
        assert rows[1][0] == "mutual_gaze"


# ══════════════════════════════════════════════════════════════════════════════
# GazeAversionTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestGazeAversionTracker:
    """Tests for GazeAversionTracker."""

    def test_init_defaults(self):
        """Default window is 60, default min_conf is 0.5."""
        t = GazeAversionTracker()
        assert t.window == 60
        assert t.min_conf == 0.5

    def test_init_custom(self):
        """Custom window and confidence."""
        t = GazeAversionTracker(window_frames=30, min_obj_conf=0.8)
        assert t.window == 30
        assert t.min_conf == 0.8

    def test_no_aversion_when_looking(self):
        """Person looking at the object does not trigger aversion."""
        t = GazeAversionTracker(window_frames=3)
        det = _make_det('cup', conf=0.9)
        for _ in range(10):
            t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=[det],
                hits={(0, 0)},  # person 0 is looking at object 0
            )
        result = t.update(
            persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
            dets=[det],
            hits={(0, 0)},
        )
        assert len(result['aversions']) == 0

    def test_aversion_triggered_after_window(self):
        """Aversion fires after window_frames of not looking."""
        window = 5
        t = GazeAversionTracker(window_frames=window)
        det = _make_det('cup', conf=0.9)
        result = None
        for _ in range(window + 1):
            result = t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=[det],
                hits=set(),  # not looking at anything
            )
        assert (0, 'cup') in result['aversions']

    def test_aversion_resets_when_looking(self):
        """Counter resets if the person looks at the object."""
        window = 5
        t = GazeAversionTracker(window_frames=window)
        det = _make_det('cup', conf=0.9)
        # Build up partial aversion
        for _ in range(window - 1):
            t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=[det],
                hits=set(),
            )
        # Now look at it -- resets counter
        t.update(
            persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
            dets=[det],
            hits={(0, 0)},
        )
        # Continue not looking -- should need full window again
        result = None
        for _ in range(window - 1):
            result = t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=[det],
                hits=set(),
            )
        assert (0, 'cup') not in result['aversions']

    def test_low_conf_objects_ignored(self):
        """Objects below min_conf are not tracked for aversion."""
        t = GazeAversionTracker(window_frames=3, min_obj_conf=0.8)
        det = _make_det('cup', conf=0.3)  # below threshold
        for _ in range(10):
            t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=[det],
                hits=set(),
            )
        result = t.update(
            persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
            dets=[det],
            hits=set(),
        )
        assert len(result['aversions']) == 0

    def test_empty_input(self):
        """No persons and no dets produces no aversions."""
        t = GazeAversionTracker()
        result = t.update(persons_gaze=[], dets=[], hits=set())
        assert result['aversions'] == set()

    def test_multiple_objects(self):
        """Aversion can fire for one object but not another."""
        window = 3
        t = GazeAversionTracker(window_frames=window)
        dets = [_make_det('cup', conf=0.9), _make_det('phone', conf=0.9)]
        result = None
        for _ in range(window + 1):
            result = t.update(
                persons_gaze=[_make_person_gaze([0, 0], [50, 50])],
                dets=dets,
                hits={(0, 0)},  # looking at cup (idx 0), not phone (idx 1)
            )
        assert (0, 'cup') not in result['aversions']
        assert (0, 'phone') in result['aversions']


# ══════════════════════════════════════════════════════════════════════════════
# AttentionSpanTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestAttentionSpanTracker:
    """Tests for AttentionSpanTracker."""

    def test_init(self):
        """Tracker initializes with empty state."""
        t = AttentionSpanTracker()
        assert t._active == {}
        assert t._durations == {}

    def test_single_glance(self):
        """A completed glance is recorded with correct duration."""
        t = AttentionSpanTracker()
        det = _make_det('cup')
        # Frames 0-4: looking at cup (5 frames of looking)
        for frame_no in range(5):
            t.update(frame_no=frame_no, dets=[det], hits={(0, 0)})
        # Frame 5: stop looking -- closes the glance
        t.update(frame_no=5, dets=[det], hits=set())
        assert t.avg_glance_duration(0, 'cup') == 5.0

    def test_no_completed_glance(self):
        """Active (unclosed) glance is not included in averages."""
        t = AttentionSpanTracker()
        det = _make_det('cup')
        for frame_no in range(5):
            t.update(frame_no=frame_no, dets=[det], hits={(0, 0)})
        # Glance still active -- not yet closed
        assert t.avg_glance_duration(0, 'cup') == 0.0

    def test_multiple_glances_averaged(self):
        """Multiple completed glances produce the correct average."""
        t = AttentionSpanTracker()
        det = _make_det('cup')
        # Glance 1: frames 0-2 (duration 3)
        for fn in range(3):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        t.update(frame_no=3, dets=[det], hits=set())  # close glance 1
        # Glance 2: frames 4-10 (duration 7)
        for fn in range(4, 11):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        t.update(frame_no=11, dets=[det], hits=set())  # close glance 2
        assert t.avg_glance_duration(0, 'cup') == pytest.approx(5.0)  # (3+7)/2

    def test_all_averages(self):
        """all_averages returns per-class averages for a participant."""
        t = AttentionSpanTracker()
        dets = [_make_det('cup'), _make_det('phone')]
        # Look at cup for 4 frames, then stop
        for fn in range(4):
            t.update(frame_no=fn, dets=dets, hits={(0, 0)})
        t.update(frame_no=4, dets=dets, hits=set())
        # Look at phone for 2 frames, then stop
        for fn in range(5, 7):
            t.update(frame_no=fn, dets=dets, hits={(0, 1)})
        t.update(frame_no=7, dets=dets, hits=set())
        avgs = t.all_averages(0)
        assert avgs['cup'] == pytest.approx(4.0)
        assert avgs['phone'] == pytest.approx(2.0)

    def test_most_salient(self):
        """most_salient returns the object with the longest average glance."""
        t = AttentionSpanTracker()
        dets = [_make_det('cup'), _make_det('phone')]
        # Cup glance: 10 frames
        for fn in range(10):
            t.update(frame_no=fn, dets=dets, hits={(0, 0)})
        t.update(frame_no=10, dets=dets, hits=set())
        # Phone glance: 3 frames
        for fn in range(11, 14):
            t.update(frame_no=fn, dets=dets, hits={(0, 1)})
        t.update(frame_no=14, dets=dets, hits=set())
        result = t.most_salient(0)
        assert result is not None
        assert result[0] == 'cup'
        assert result[1] == pytest.approx(10.0)

    def test_most_salient_no_data(self):
        """most_salient returns None when no glances completed."""
        t = AttentionSpanTracker()
        assert t.most_salient(0) is None

    def test_all_participants(self):
        """all_participants returns face indices with completed glances."""
        t = AttentionSpanTracker()
        det = _make_det('cup')
        # Person 0 looks then stops
        t.update(frame_no=0, dets=[det], hits={(0, 0)})
        t.update(frame_no=1, dets=[det], hits=set())
        # Person 1 looks then stops
        t.update(frame_no=2, dets=[det], hits={(1, 0)})
        t.update(frame_no=3, dets=[det], hits=set())
        assert t.all_participants() == {0, 1}

    def test_all_participants_empty(self):
        """all_participants returns empty set with no data."""
        t = AttentionSpanTracker()
        assert t.all_participants() == set()

    def test_empty_input(self):
        """Empty input does not crash."""
        t = AttentionSpanTracker()
        result = t.update(frame_no=0, dets=[], hits=set())
        assert result == {}

    def test_kwargs_interface(self):
        """Tracker works when called via **kwargs unpacking."""
        t = AttentionSpanTracker()
        kwargs = {
            'frame_no': 0,
            'dets': [_make_det('cup')],
            'hits': {(0, 0)},
            'extra_unused_key': 'ignored',
        }
        result = t.update(**kwargs)
        assert isinstance(result, dict)
