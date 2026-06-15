"""Tests for the event-log writer: header layout and the t_seconds column.

Covers io/writers.open_event_log (header) and
outputs/data_pipeline.collect_frame_data (per-frame row values).
"""

import csv

from mindsight.io.writers import open_event_log
from mindsight.outputs import data_pipeline
from mindsight.outputs.data_pipeline import collect_frame_data, finalize_run
from mindsight.pipeline_config import OutputConfig

# trimmed.mp4's real cv2 CAP_PROP_FPS -- NON-integer; never hand-compute with 30.0.
TRIMMED_FPS = 29.894330189751166


def _read(path):
    with open(path) as f:
        return list(csv.reader(f))


def _hit_event():
    return {"face_idx": 0, "object": "cat", "object_conf": 0.5,
            "bbox": [1, 2, 3, 4]}


class TestOpenEventLogHeader:

    def test_single_mode_header(self, tmp_path):
        log = tmp_path / "events.csv"
        fh, _ = open_event_log(OutputConfig(log_path=str(log)))
        fh.close()
        header = _read(log)[0]
        assert header[0] == "frame"
        assert header[1] == "t_seconds"
        assert header[2] == "face_idx"
        assert header == ["frame", "t_seconds", "face_idx", "object",
                          "object_conf", "bbox_x1", "bbox_y1", "bbox_x2",
                          "bbox_y2", "joint_attention",
                          "joint_attention_confirmed", "participant_label"]

    def test_project_mode_header(self, tmp_path):
        log = tmp_path / "events.csv"
        cfg = OutputConfig(log_path=str(log), video_name="clip",
                           conditions="a|b")
        fh, _ = open_event_log(cfg)
        fh.close()
        header = _read(log)[0]
        # project prefix stays in front; t_seconds is core column 3 (T5)
        assert header[0] == "video_name"
        assert header[1] == "conditions"
        assert header[2] == "frame"
        assert header[3] == "t_seconds"
        assert header[4] == "face_idx"


class TestCollectFrameDataTSeconds:

    def test_row_value_at_known_fps(self, tmp_path):
        log = tmp_path / "events.csv"
        fh, w = open_event_log(OutputConfig(log_path=str(log)))
        ctx = {"video_fps": TRIMMED_FPS}
        collect_frame_data(ctx, log_csv=w, frame_no=6,
                           hit_events=[_hit_event()],
                           face_track_ids=[], persons_gaze=[])
        fh.close()
        row = _read(log)[1]
        # frame 6 / 29.894330189751166 = 0.20071... -> "0.201"
        assert row[0] == "6"
        assert row[1] == f"{6 / TRIMMED_FPS:.3f}"
        assert row[1] == "0.201"
        assert row[2] == "0"          # face_idx
        assert row[3] == "cat"        # object

    def test_row_value_three_decimals(self, tmp_path):
        log = tmp_path / "events.csv"
        fh, w = open_event_log(OutputConfig(log_path=str(log)))
        ctx = {"video_fps": TRIMMED_FPS}
        collect_frame_data(ctx, log_csv=w, frame_no=100,
                           hit_events=[_hit_event()],
                           face_track_ids=[], persons_gaze=[])
        fh.close()
        row = _read(log)[1]
        assert row[1] == f"{100 / TRIMMED_FPS:.3f}"
        # exactly three decimal places
        assert len(row[1].split(".")[1]) == 3

    def test_fps_zero_fallback_empty_string(self, tmp_path):
        log = tmp_path / "events.csv"
        fh, w = open_event_log(OutputConfig(log_path=str(log)))
        ctx = {"video_fps": 0.0}
        collect_frame_data(ctx, log_csv=w, frame_no=5,
                           hit_events=[_hit_event()],
                           face_track_ids=[], persons_gaze=[])
        fh.close()
        row = _read(log)[1]
        assert row[1] == ""

    def test_fps_missing_fallback_empty_string(self, tmp_path):
        log = tmp_path / "events.csv"
        fh, w = open_event_log(OutputConfig(log_path=str(log)))
        ctx = {}  # no video_fps key at all
        collect_frame_data(ctx, log_csv=w, frame_no=5,
                           hit_events=[_hit_event()],
                           face_track_ids=[], persons_gaze=[])
        fh.close()
        row = _read(log)[1]
        assert row[1] == ""

    def test_project_mode_row_prefix(self, tmp_path):
        log = tmp_path / "events.csv"
        cfg = OutputConfig(log_path=str(log), video_name="clip",
                           conditions="a|b")
        fh, w = open_event_log(cfg)
        ctx = {"video_fps": TRIMMED_FPS, "video_name": "clip",
               "conditions": "a|b"}
        collect_frame_data(ctx, log_csv=w, frame_no=6,
                           hit_events=[_hit_event()],
                           face_track_ids=[], persons_gaze=[])
        fh.close()
        row = _read(log)[1]
        assert row[0] == "clip"
        assert row[1] == "a|b"
        assert row[2] == "6"          # frame
        assert row[3] == f"{6 / TRIMMED_FPS:.3f}"  # t_seconds core col 3


class TestFinalizeRunFps:
    """finalize_run must hand generate_run_charts the REAL video fps
    (ctx['video_fps']), not the hard-coded 30.0 default (latent bug fixed
    in SP2 Step 6)."""

    def _run(self, monkeypatch, ctx, tmp_path):
        captured = {}

        def fake_charts(output_path, all_trackers, total_frames, fps,
                        pid_map=None, data_plugins=None):
            captured["fps"] = fps
            return []

        monkeypatch.setattr(data_pipeline, "generate_run_charts", fake_charts)
        ctx.setdefault("charts_path", str(tmp_path / "charts.png"))
        ctx.setdefault("source", "clip.mp4")
        finalize_run(ctx)
        return captured

    def test_uses_real_video_fps(self, monkeypatch, tmp_path):
        captured = self._run(
            monkeypatch,
            {"video_fps": TRIMMED_FPS, "total_frames": 100}, tmp_path)
        assert captured["fps"] == TRIMMED_FPS

    def test_falls_back_to_thirty_when_no_fps(self, monkeypatch, tmp_path):
        captured = self._run(
            monkeypatch, {"total_frames": 100}, tmp_path)
        assert captured["fps"] == 30.0

    def test_video_fps_wins_over_stale_fps_key(self, monkeypatch, tmp_path):
        captured = self._run(
            monkeypatch,
            {"video_fps": TRIMMED_FPS, "fps": 30.0, "total_frames": 100},
            tmp_path)
        assert captured["fps"] == TRIMMED_FPS
