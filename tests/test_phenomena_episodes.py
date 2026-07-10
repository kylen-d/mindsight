"""Tests for episodic phenomena recording (batch 2+3, B7 + B5).

Covers:
* the ``EpisodeLog`` helper;
* the ``finalize()`` run-end hook wiring;
* per-tracker episode open/close/finalize + the JA / tip / aversion / attention
  span / gaze-following behaviour fixes;
* the merged ``{stem}_phenomena_events.csv`` writer and its global aggregation
  suffix;
* the ``--gaze-leader-tips`` without ``--gaze-tips`` warning.
"""

import csv
from types import SimpleNamespace

import numpy as np

from mindsight.Phenomena.Default.attention_span import AttentionSpanTracker
from mindsight.Phenomena.Default.gaze_aversion import GazeAversionTracker
from mindsight.Phenomena.Default.gaze_following import GazeFollowingTracker
from mindsight.Phenomena.Default.joint_attention import JointAttentionTracker
from mindsight.Phenomena.Default.mutual_gaze import MutualGazeTracker
from mindsight.Phenomena.helpers import EpisodeLog
from mindsight.Phenomena.phenomena_pipeline import (
    finalize_trackers,
    warn_leader_tips_without_tips,
)
from mindsight.outputs.csv_output import write_summary_tables
from Plugins import PhenomenaPlugin

# ── Helpers ──────────────────────────────────────────────────────────────────


def _person(origin, ray_end, conf=0.9):
    return (np.asarray(origin, float), np.asarray(ray_end, float), conf)


def _bbox(cx, cy, hw=30, hh=40):
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _det(class_name, conf=0.9, x1=0, y1=0, x2=50, y2=50):
    return {'class_name': class_name, 'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


# ══════════════════════════════════════════════════════════════════════════════
# EpisodeLog
# ══════════════════════════════════════════════════════════════════════════════

class TestEpisodeLog:

    def test_open_close_records_row(self):
        log = EpisodeLog()
        log.open("k", phenomenon="p", participant=0, partner=1,
                 object="cup", frame_start=5)
        assert log.is_open("k")
        log.close("k", 12)
        assert not log.is_open("k")
        assert log.rows == [{
            "phenomenon": "p", "participant": 0, "partner": 1,
            "object": "cup", "frame_start": 5, "frame_end": 12}]

    def test_double_open_is_noop(self):
        log = EpisodeLog()
        log.open("k", phenomenon="p", participant=0, partner="",
                 object="", frame_start=5)
        log.open("k", phenomenon="p", participant=0, partner="",
                 object="", frame_start=99)  # ignored
        log.close("k", 10)
        assert log.rows[0]["frame_start"] == 5

    def test_close_unknown_is_noop(self):
        log = EpisodeLog()
        log.close("missing", 10)
        assert log.rows == []

    def test_close_all(self):
        log = EpisodeLog()
        log.open("a", phenomenon="p", participant=0, partner="",
                 object="", frame_start=1)
        log.open("b", phenomenon="p", participant=1, partner="",
                 object="", frame_start=2)
        log.close_all(20)
        assert {r["participant"] for r in log.rows} == {0, 1}
        assert all(r["frame_end"] == 20 for r in log.rows)


# ══════════════════════════════════════════════════════════════════════════════
# finalize() hook wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestFinalizeHook:

    def test_base_finalize_is_noop(self):
        # Base default must not raise and must return None.
        assert PhenomenaPlugin.finalize(MutualGazeTracker(), 5) is None

    def test_finalize_trackers_calls_each(self):
        calls = []

        class Stub(PhenomenaPlugin):
            name = "stub"

            def finalize(self, frame_no, **kwargs):
                calls.append(frame_no)

        finalize_trackers([Stub(), Stub()], 869)
        assert calls == [869, 869]


# ══════════════════════════════════════════════════════════════════════════════
# JointAttentionTracker: quorum math, temporal filter, tip separation
# ══════════════════════════════════════════════════════════════════════════════

class TestJointAttentionQuorum:

    def test_compute_raw_ja_all_faces(self):
        # 2 faces both look at object 0 -> JA; object 1 only one face -> not JA.
        hits = {(0, 0), (1, 0), (0, 1)}
        assert JointAttentionTracker._compute_raw_ja(hits, 2, 1.0) == {0}

    def test_compute_raw_ja_needs_two_faces(self):
        assert JointAttentionTracker._compute_raw_ja({(0, 0)}, 1, 1.0) == set()

    def test_compute_raw_ja_soft_quorum(self):
        # 4 faces, quorum 0.75 -> ceil(3) watchers needed.
        hits = {(0, 0), (1, 0), (2, 0)}          # 3/4 look at obj 0
        assert JointAttentionTracker._compute_raw_ja(hits, 4, 0.75) == {0}
        hits2 = {(0, 0), (1, 0)}                  # only 2/4
        assert JointAttentionTracker._compute_raw_ja(hits2, 4, 0.75) == set()

    def test_temporal_filter_confirms_after_threshold(self):
        t = JointAttentionTracker(window=4, threshold=0.75)
        # Raw JA present 3 of 4 frames -> 0.75 >= threshold -> confirmed.
        t._temporal_filter({0})
        t._temporal_filter({0})
        t._temporal_filter(set())
        assert t._temporal_filter({0}) == {0}

    def test_tip_counts_as_joint_attention(self):
        """Tip convergence IS joint attention (union, user ruling 2026-07-09):
        tip-only frames count toward the JA percentage, and the tip breakdown
        counter tracks them separately."""
        t = JointAttentionTracker(window=0)
        # No object JA, but a tip convergence is present.
        faces = frozenset({0, 1})
        for fn in range(5):
            t.update(frame_no=fn, hits=set(), n_faces=2,
                     tip_convergences=[(faces, np.array([10.0, 10.0]))],
                     dets=[], pid_map=None)
        assert t.joint_pct == 100.0          # tip-only frames are JA frames
        assert t._confirmed_frames == 5
        assert t._tip_frames == 5            # breakdown counter

    def test_multiple_ja_objects_count_one_frame(self):
        """A frame where SEVERAL objects hold JA simultaneously is one JA
        frame (user ruling 2026-07-09): the percentage is per-frame, not
        per-object."""
        t = JointAttentionTracker(window=0)
        dets = [_det("plate"), _det("cup")]
        # Both faces look at BOTH objects -> two simultaneous JA objects.
        t.update(frame_no=0, hits={(0, 0), (1, 0), (0, 1), (1, 1)}, n_faces=2,
                 tip_convergences=[], dets=dets, pid_map=None)
        assert t._joint_frames == 1
        assert t._confirmed_frames == 1
        # Both objects still get their own episode records.
        t.finalize(1)
        ja_eps = [e for e in t._episodes.rows
                  if e["phenomenon"] == "joint_attention"]
        assert {e["object"] for e in ja_eps} == {"plate", "cup"}

    def test_tip_and_object_ja_never_double_count(self):
        """A frame with BOTH object JA and tip convergence counts once."""
        t = JointAttentionTracker(window=0)
        det = _det("plate")
        faces = frozenset({0, 1})
        # Both modes active on the same frame.
        t.update(frame_no=0, hits={(0, 0), (1, 0)}, n_faces=2,
                 tip_convergences=[(faces, np.array([10.0, 10.0]))],
                 dets=[det], pid_map=None)
        # Tip only.
        t.update(frame_no=1, hits=set(), n_faces=2,
                 tip_convergences=[(faces, np.array([10.0, 10.0]))],
                 dets=[det], pid_map=None)
        # Neither.
        t.update(frame_no=2, hits=set(), n_faces=2, tip_convergences=[],
                 dets=[det], pid_map=None)
        assert t._confirmed_frames == 2      # union: 1 + 1, not 2 + 1
        assert t._tip_frames == 2

    def test_tip_episode_and_counter_separation(self):
        t = JointAttentionTracker(window=0)
        faces = frozenset({0, 1})
        # Tip present frames 0-2, gone frame 3.
        for fn in range(3):
            t.update(frame_no=fn, hits=set(), n_faces=2,
                     tip_convergences=[(faces, np.array([10.0, 10.0]))],
                     dets=[], pid_map=None)
        t.update(frame_no=3, hits=set(), n_faces=2, tip_convergences=[],
                 dets=[], pid_map=None)
        t.finalize(4)
        tip_eps = [e for e in t._episodes.rows
                   if e["phenomenon"] == "tip_convergence"]
        assert len(tip_eps) == 1
        assert tip_eps[0]["participant"] == "P0+P1"
        assert tip_eps[0]["frame_start"] == 0
        assert tip_eps[0]["frame_end"] == 3
        # summary_metrics emits tip rows only when tip frames occurred.
        rows = t.summary_metrics(10, 30.0)
        tip_rows = [r for r in rows if r.get("phenomenon") == "tip_convergence"]
        assert {r["metric"] for r in tip_rows} == {
            "frames_active", "seconds_active", "pct_of_video"}

    def test_no_tip_rows_when_no_tips(self):
        t = JointAttentionTracker(window=0)
        for fn in range(3):
            t.update(frame_no=fn, hits=set(), n_faces=2, tip_convergences=[],
                     dets=[], pid_map=None)
        rows = t.summary_metrics(10, 30.0)
        assert all(r.get("phenomenon") != "tip_convergence" for r in rows)

    def test_object_ja_episode(self):
        t = JointAttentionTracker(window=0)
        det = _det("plate")
        # Both faces look at plate (obj 0) frames 0-2, then stop.
        for fn in range(3):
            t.update(frame_no=fn, hits={(0, 0), (1, 0)}, n_faces=2,
                     dets=[det], tip_convergences=[], pid_map=None)
        t.update(frame_no=3, hits=set(), n_faces=2, dets=[det],
                 tip_convergences=[], pid_map=None)
        t.finalize(4)
        ja_eps = [e for e in t._episodes.rows
                  if e["phenomenon"] == "joint_attention"]
        assert len(ja_eps) == 1
        assert ja_eps[0]["participant"] == "all"
        assert ja_eps[0]["object"] == "plate"


# ══════════════════════════════════════════════════════════════════════════════
# MutualGazeTracker episodes
# ══════════════════════════════════════════════════════════════════════════════

class TestMutualGazeEpisodes:

    def _facing(self):
        pg = [_person([100, 200], [300, 200]),
              _person([300, 200], [100, 200])]
        bb = [_bbox(100, 200), _bbox(300, 200)]
        return pg, bb

    def test_episode_open_close(self):
        t = MutualGazeTracker()
        pg, bb = self._facing()
        for fn in range(4):
            t.update(frame_no=fn, persons_gaze=pg, face_bboxes=bb)
        # Break eye contact at frame 4.
        t.update(frame_no=4, persons_gaze=[_person([100, 200], [500, 200]),
                                           _person([300, 200], [500, 200])],
                 face_bboxes=bb)
        assert len(t._episodes.rows) == 1
        ep = t._episodes.rows[0]
        assert ep["frame_start"] == 0
        assert ep["frame_end"] == 4

    def test_finalize_closes_open_pair(self):
        t = MutualGazeTracker()
        pg, bb = self._facing()
        for fn in range(3):
            t.update(frame_no=fn, persons_gaze=pg, face_bboxes=bb)
        assert t._episodes.rows == []      # still active
        t.finalize(3)
        assert len(t._episodes.rows) == 1
        assert t._episodes.rows[0]["frame_end"] == 3

    def test_pair_counts_unchanged(self):
        """Episode bookkeeping must not perturb pair_counts / summary values."""
        t = MutualGazeTracker()
        pg, bb = self._facing()
        for fn in range(5):
            t.update(frame_no=fn, persons_gaze=pg, face_bboxes=bb)
        assert t.pair_counts[(0, 1)] == 5
        rows = t.summary_metrics(100, 100.0)
        metrics = {r["metric"]: r["value"] for r in rows}
        assert metrics["frames_active"] == 5

    def test_episode_rows_resolves_pids(self):
        t = MutualGazeTracker()
        pg, bb = self._facing()
        t.update(frame_no=0, persons_gaze=pg, face_bboxes=bb)
        t.finalize(1)
        rows = t.episode_rows(100, 30.0)
        assert rows[0]["participant"] == "P0"
        assert rows[0]["partner"] == "P1"
        assert rows[0]["phenomenon"] == "mutual_gaze"


# ══════════════════════════════════════════════════════════════════════════════
# GazeAversionTracker resolved-episode accounting
# ══════════════════════════════════════════════════════════════════════════════

class TestGazeAversionEpisodes:

    def test_mid_run_episode_reported(self):
        """A streak that resolves mid-run must still appear as an episode."""
        t = GazeAversionTracker(window_frames=3)
        det = _det("plate")
        # Frames 0-4: not looking (streak begins frame 0, crosses at frame 2).
        for fn in range(5):
            t.update(frame_no=fn, persons_gaze=[_person([0, 0], [1, 1])],
                     dets=[det], hits=set())
        # Frame 5: looks -> resolves the streak, closing the episode.
        t.update(frame_no=5, persons_gaze=[_person([0, 0], [1, 1])],
                 dets=[det], hits={(0, 0)})
        assert len(t._episodes.rows) == 1
        ep = t._episodes.rows[0]
        assert ep["frame_start"] == 0     # streak began frame 0
        assert ep["frame_end"] == 5
        assert ep["participant"] == 0
        assert ep["object"] == "plate"

    def test_reset_clears_streak_start(self):
        t = GazeAversionTracker(window_frames=3)
        det = _det("plate")
        # Partial streak, then look resets it before the window is reached.
        t.update(frame_no=0, persons_gaze=[_person([0, 0], [1, 1])],
                 dets=[det], hits=set())
        t.update(frame_no=1, persons_gaze=[_person([0, 0], [1, 1])],
                 dets=[det], hits={(0, 0)})   # reset
        assert (0, "plate") not in t._streak_start
        assert t._no_look[(0, "plate")] == 0

    def test_end_of_run_streak_closed_by_finalize(self):
        t = GazeAversionTracker(window_frames=3)
        det = _det("plate")
        for fn in range(6):
            t.update(frame_no=fn, persons_gaze=[_person([0, 0], [1, 1])],
                     dets=[det], hits=set())
        assert t._episodes.rows == []      # unresolved streak still open
        t.finalize(6)
        assert len(t._episodes.rows) == 1
        assert t._episodes.rows[0]["frame_end"] == 6

    def test_summary_metrics_episode_accounting(self):
        t = GazeAversionTracker(window_frames=3)
        det = _det("plate")
        for fn in range(6):
            t.update(frame_no=fn, persons_gaze=[_person([0, 0], [1, 1])],
                     dets=[det], hits=set())
        t.finalize(6)
        rows = t.summary_metrics(100, 30.0)
        metrics = {r["metric"]: r["value"] for r in rows}
        assert metrics["episode_count"] == 1
        assert metrics["frames_active"] == 6      # frame_end - frame_start
        assert "seconds_active" in metrics
        assert "pct_of_video" in metrics


# ══════════════════════════════════════════════════════════════════════════════
# AttentionSpanTracker episodes + new summary rows + finalize flush
# ══════════════════════════════════════════════════════════════════════════════

class TestAttentionSpanEpisodes:

    def test_completed_glance_episode(self):
        t = AttentionSpanTracker()
        det = _det("cup")
        for fn in range(5):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        t.update(frame_no=5, dets=[det], hits=set())   # close glance
        assert len(t._episodes.rows) == 1
        ep = t._episodes.rows[0]
        assert ep["phenomenon"] == "attention_span"
        assert ep["frame_start"] == 0
        assert ep["frame_end"] == 5
        assert ep["object"] == "cup"

    def test_finalize_flushes_in_flight_glance(self):
        t = AttentionSpanTracker()
        det = _det("cup")
        for fn in range(5):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        assert t.avg_glance_duration(0, "cup") == 0.0   # still open
        t.finalize(5)
        assert t.avg_glance_duration(0, "cup") == 5.0   # flushed
        assert len(t._episodes.rows) == 1

    def test_summary_metrics_new_rows(self):
        t = AttentionSpanTracker()
        det = _det("cup")
        # Glance 1: 4 frames.
        for fn in range(4):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        t.update(frame_no=4, dets=[det], hits=set())
        # Glance 2: 6 frames.
        for fn in range(5, 11):
            t.update(frame_no=fn, dets=[det], hits={(0, 0)})
        t.update(frame_no=11, dets=[det], hits=set())
        rows = t.summary_metrics(100, 10.0)
        metrics = {r["metric"]: r["value"] for r in rows}
        assert metrics["glance_count"] == 2
        # mean glance = (4+6)/2 = 5 frames / 10 fps = 0.500 s
        assert metrics["mean_glance_seconds"] == "0.500"
        # total = 10 frames / 10 fps = 1.000 s
        assert metrics["total_seconds"] == "1.000"


# ══════════════════════════════════════════════════════════════════════════════
# GazeFollowingTracker flicker double-award regression
# ══════════════════════════════════════════════════════════════════════════════

class TestGazeFollowingFlicker:

    def test_same_follower_reacquire_awards_once(self):
        """Hit flicker: same follower re-acquires within lag -> ONE event."""
        t = GazeFollowingTracker(lag_frames=30)
        # Frame 0: leader 0 acquires object 0 (a shift).
        t.update(frame_no=0, hits={(0, 0)})
        # Frame 1: follower 1 acquires object 0 -> follow event #1.
        t.update(frame_no=1, hits={(0, 0), (1, 0)})
        assert len(t.event_log) == 1
        # Frame 2: follower 1 loses object 0 (flicker off).
        t.update(frame_no=2, hits={(0, 0)})
        # Frame 3: follower 1 re-acquires object 0 within lag -> NO new event.
        t.update(frame_no=3, hits={(0, 0), (1, 0)})
        assert len(t.event_log) == 1

    def test_two_followers_of_one_shift_both_award(self):
        """The leader's single shift must credit BOTH distinct followers --
        the fix tracks awarded followers per shift rather than deleting the
        shift after the first award."""
        t = GazeFollowingTracker(lag_frames=30)
        # Frame 0: leader 0 acquires object 0 (the shift under test).
        t.update(frame_no=0, hits={(0, 0)})
        # Frame 1: followers 1 and 2 both acquire object 0.
        t.update(frame_no=1, hits={(0, 0), (1, 0), (2, 0)})
        leader0_followers = {ev["follower"] for ev in t.event_log
                             if ev["leader"] == 0}
        assert leader0_followers == {1, 2}

    def test_follow_episode_recorded(self):
        t = GazeFollowingTracker(lag_frames=30)
        t.update(frame_no=0, hits={(0, 0)})
        t.update(frame_no=1, hits={(0, 0), (1, 0)})
        assert len(t._episodes.rows) == 1
        ep = t._episodes.rows[0]
        assert ep["participant"] == 1     # follower
        assert ep["partner"] == 0         # leader
        assert ep["frame_start"] == ep["frame_end"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# --gaze-leader-tips without --gaze-tips warning
# ══════════════════════════════════════════════════════════════════════════════

class TestLeaderTipsWarning:

    def test_warns_when_tips_off(self, capsys):
        fired = warn_leader_tips_without_tips(
            SimpleNamespace(gaze_leader_tips=True),
            SimpleNamespace(gaze_tips=False))
        assert fired is True
        assert "--gaze-leader-tips has no effect without --gaze-tips" \
            in capsys.readouterr().out

    def test_silent_when_tips_on(self, capsys):
        fired = warn_leader_tips_without_tips(
            SimpleNamespace(gaze_leader_tips=True),
            SimpleNamespace(gaze_tips=True))
        assert fired is False
        assert capsys.readouterr().out == ""

    def test_silent_when_leader_tips_off(self, capsys):
        fired = warn_leader_tips_without_tips(
            SimpleNamespace(gaze_leader_tips=False),
            SimpleNamespace(gaze_tips=False))
        assert fired is False
        assert capsys.readouterr().out == ""


# ══════════════════════════════════════════════════════════════════════════════
# Merged phenomena_events writer
# ══════════════════════════════════════════════════════════════════════════════

_EPISODE_HEADER = ["video_name", "conditions", "phenomenon", "participant",
                   "partner", "object", "frame_start", "frame_end", "t_start",
                   "t_end", "duration_s"]


class _EpTracker(PhenomenaPlugin):
    """Minimal tracker exposing a pre-populated EpisodeLog."""

    name = "ep"

    def __init__(self, episodes):
        self._episodes = EpisodeLog()
        self._episodes.rows = episodes


class TestPhenomenaEventsWriter:

    def _tracker(self):
        return _EpTracker([
            {"phenomenon": "mutual_gaze", "participant": 0, "partner": 1,
             "object": "", "frame_start": 30, "frame_end": 60},
            {"phenomenon": "gaze_following", "participant": 1, "partner": 0,
             "object": "", "frame_start": 5, "frame_end": 5},
        ])

    def test_header_and_sorting(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {},
                             all_trackers=[self._tracker()])
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        assert rows[0] == _EPISODE_HEADER
        # Sorted by (frame_start, phenomenon, participant): follow(5) first.
        assert rows[1][2] == "gaze_following"
        assert rows[1][6] == "5"
        assert rows[2][2] == "mutual_gaze"
        assert rows[2][6] == "30"

    def test_timestamps_and_duration(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {},
                             all_trackers=[self._tracker()])
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        mg = [r for r in rows if r[2] == "mutual_gaze"][0]
        assert mg[3] == "P0" and mg[4] == "P1"
        assert mg[8] == f"{30 / 30.0:.3f}"      # t_start
        assert mg[9] == f"{60 / 30.0:.3f}"      # t_end
        assert mg[10] == f"{30 / 30.0:.3f}"     # duration_s

    def test_empty_when_no_episodes(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {},
                             all_trackers=[_EpTracker([])])
        assert not (tmp_path / "clip_phenomena_events.csv").exists()

    def test_no_fps_blanks_time_columns(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 0.0, {},
                             all_trackers=[self._tracker()])
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        assert rows[1][8] == "" and rows[1][9] == "" and rows[1][10] == ""

    def test_project_mode_prefix(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {},
                             all_trackers=[self._tracker()],
                             video_name="clip", conditions="A|B")
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        assert rows[1][0] == "clip" and rows[1][1] == "A|B"

    def test_single_mode_empty_prefix(self, tmp_path):
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {},
                             all_trackers=[self._tracker()])
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        assert rows[1][0] == "" and rows[1][1] == ""

    def test_multiple_trackers_merged_once(self, tmp_path):
        """One merged file across trackers -- no clobber."""
        a = _EpTracker([{"phenomenon": "mutual_gaze", "participant": 0,
                         "partner": 1, "object": "", "frame_start": 1,
                         "frame_end": 2}])
        b = _EpTracker([{"phenomenon": "gaze_following", "participant": 2,
                         "partner": 3, "object": "", "frame_start": 9,
                         "frame_end": 9}])
        summary = tmp_path / "clip_summary.csv"
        write_summary_tables(summary, 100, 30.0, {}, all_trackers=[a, b])
        rows = _read(tmp_path / "clip_phenomena_events.csv")
        assert len(rows) == 3      # header + 2 data rows from both trackers
