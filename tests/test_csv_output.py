"""Tests for outputs/csv_output.py -- tidy long-format summary writer."""

import csv

from mindsight.outputs.csv_output import (
    resolve_summary_path,
    write_summary_tables,
)
from Plugins import PhenomenaPlugin

_HEADER = ["video_name", "conditions", "phenomenon", "participant",
           "partner", "object", "metric", "value"]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


class TestResolveSummaryPath:

    def test_none_returns_none(self):
        assert resolve_summary_path(None, "video.mp4") is None

    def test_false_returns_none(self):
        assert resolve_summary_path(False, "video.mp4") is None

    def test_string_passed_through(self):
        assert resolve_summary_path("/tmp/out.csv", "video.mp4") == "/tmp/out.csv"

    def test_true_auto_generates_path(self):
        result = resolve_summary_path(True, "my_video.mp4")
        assert result is not None
        assert "my_video" in result
        assert result.endswith("_summary.csv")

    def test_true_with_webcam(self):
        result = resolve_summary_path(True, 0)
        assert "webcam" in result


class TestSummaryHeaderAndLookTime:

    def test_header_row(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), total_frames=100, fps=100.0,
                             look_counts={})
        rows = _read(out)
        assert rows[0] == _HEADER
        # No data rows when there is nothing to report.
        assert len(rows) == 1

    def test_header_identical_single_and_project(self, tmp_path):
        single = tmp_path / "s_summary.csv"
        proj = tmp_path / "p_summary.csv"
        write_summary_tables(str(single), 100, 100.0, {(0, "cat"): 10})
        write_summary_tables(str(proj), 100, 100.0, {(0, "cat"): 10},
                             video_name="clip", conditions="A|B")
        assert _read(single)[0] == _read(proj)[0] == _HEADER

    def test_object_look_time_three_metrics(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), total_frames=100, fps=100.0,
                             look_counts={(0, "ball"): 50})
        rows = _read(out)[1:]
        by_metric = {r[6]: r for r in rows}
        assert set(by_metric) == {"frames_active", "seconds_active",
                                  "pct_of_video"}
        assert by_metric["frames_active"][7] == "50"
        # 50 frames / 100 fps = 0.500 s
        assert by_metric["seconds_active"][7] == "0.500"
        # 50 / 100 * 100 = 50.0000 %
        assert by_metric["pct_of_video"][7] == "50.0000"
        # phenomenon / participant / object columns
        assert by_metric["frames_active"][2] == "object_look_time"
        assert by_metric["frames_active"][3] == "P0"
        assert by_metric["frames_active"][5] == "ball"

    def test_single_mode_prefix_empty(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {(0, "cat"): 10})
        data = _read(out)[1]
        assert data[0] == "" and data[1] == ""

    def test_project_mode_prefix_filled(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {(0, "cat"): 10},
                             video_name="myclip", conditions="cond_a|cond_b")
        data = _read(out)[1]
        assert data[0] == "myclip"
        assert data[1] == "cond_a|cond_b"

    def test_zero_fps_blank_seconds(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 0.0, {(0, "cat"): 10})
        rows = _read(out)[1:]
        sec = [r for r in rows if r[6] == "seconds_active"][0]
        assert sec[7] == ""

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "x_summary.csv"
        write_summary_tables(str(out), 10, 100.0, {(0, "x"): 5})
        assert out.exists()


class _FakeTracker(PhenomenaPlugin):
    name = "fake_metric"

    def summary_metrics(self, total_frames, fps, *, pid_map=None):
        return [
            {"participant": "P0", "partner": "P1", "object": "",
             "metric": "event_count", "value": 7},
        ]


class _FakeStreamTracker(PhenomenaPlugin):
    name = "fake_stream"

    def summary_tables(self, total_frames, fps, *, pid_map=None):
        return {"my_stream": (["participant", "idx"],
                              [["P0", 0], ["P0", 1]])}


class _LegacyTracker(PhenomenaPlugin):
    name = "legacy_thing"

    def csv_rows(self, total_frames, *, pid_map=None):
        return [["category", "value"], ["legacy_thing", "99"]]


class TestTrackerHooks:

    def test_summary_metrics_row_emitted(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_FakeTracker()])
        rows = _read(out)[1:]
        assert len(rows) == 1
        r = rows[0]
        # phenomenon defaults to tracker name
        assert r[2] == "fake_metric"
        assert r[3] == "P0" and r[4] == "P1"
        assert r[6] == "event_count" and r[7] == "7"

    def test_stream_table_written_to_own_file(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_FakeStreamTracker()],
                             video_name="vid", conditions="C")
        stream = tmp_path / "vid_my_stream.csv"
        assert stream.exists()
        rows = _read(stream)
        assert rows[0] == ["video_name", "conditions", "participant", "idx"]
        assert rows[1] == ["vid", "C", "P0", "0"]
        assert rows[2] == ["vid", "C", "P0", "1"]

    def test_empty_stream_not_written(self, tmp_path):
        class _Empty(PhenomenaPlugin):
            name = "empty_stream"

            def summary_tables(self, total_frames, fps, *, pid_map=None):
                return {"nothing": (["a"], [])}

        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_Empty()])
        assert not (tmp_path / "vid_nothing.csv").exists()

    def test_legacy_csv_rows_passthrough(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_LegacyTracker()])
        passthrough = tmp_path / "vid_plugin_legacy_thing.csv"
        assert passthrough.exists()
        rows = _read(passthrough)
        assert rows[0] == ["category", "value"]
        assert rows[1] == ["legacy_thing", "99"]

    def test_tidy_tracker_no_passthrough_file(self, tmp_path):
        # A tracker implementing tidy hooks must NOT get a passthrough file.
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_FakeTracker()])
        assert not (tmp_path / "vid_plugin_fake_metric.csv").exists()

    def test_deterministic_sort(self, tmp_path):
        class _Multi(PhenomenaPlugin):
            name = "z_pheno"

            def summary_metrics(self, total_frames, fps, *, pid_map=None):
                return [
                    {"participant": "P1", "partner": "", "object": "",
                     "metric": "b", "value": 1},
                    {"participant": "P0", "partner": "", "object": "",
                     "metric": "a", "value": 2},
                ]

        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {(0, "obj"): 5},
                             all_trackers=[_Multi()])
        rows = _read(out)[1:]
        keys = [(r[2], r[3], r[4], r[5], r[6]) for r in rows]
        assert keys == sorted(keys)


class _LabelledTracker(PhenomenaPlugin):
    name = "terse_name"
    summary_label = "pretty_label"

    def summary_metrics(self, total_frames, fps, *, pid_map=None):
        return [
            {"participant": "P0", "partner": "", "object": "",
             "metric": "event_count", "value": 3},
        ]


class TestSummaryLabel:

    def test_default_label_is_name(self):
        # Base property: summary_label defaults to name when not overridden.
        assert _FakeTracker().summary_label == "fake_metric"

    def test_override_label_is_used_as_phenomenon(self, tmp_path):
        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_LabelledTracker()])
        rows = _read(out)[1:]
        assert len(rows) == 1
        # phenomenon column takes the summary_label, NOT the terse name.
        assert rows[0][2] == "pretty_label"

    def test_explicit_phenomenon_still_wins_over_label(self, tmp_path):
        class _Explicit(PhenomenaPlugin):
            name = "terse_name"
            summary_label = "pretty_label"

            def summary_metrics(self, total_frames, fps, *, pid_map=None):
                return [{"phenomenon": "explicit", "participant": "P0",
                         "partner": "", "object": "", "metric": "m",
                         "value": 1}]

        out = tmp_path / "vid_summary.csv"
        write_summary_tables(str(out), 100, 100.0, {},
                             all_trackers=[_Explicit()])
        assert _read(out)[1][2] == "explicit"
