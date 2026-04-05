"""Tests for DataCollection/csv_output.py -- summary CSV writing."""

import csv

import pytest

from ms.DataCollection.csv_output import resolve_summary_path, write_summary_csv


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
        assert result.endswith(".csv")

    def test_true_with_webcam(self):
        result = resolve_summary_path(True, 0)
        assert "webcam" in result


class TestWriteSummaryCsv:

    def test_basic_write(self, tmp_path):
        out = tmp_path / "summary.csv"
        look_counts = {(0, "person"): 50, (0, "cat"): 20}
        write_summary_csv(str(out), total_frames=100,
                          look_counts=look_counts)
        assert out.exists()
        content = out.read_text()
        assert "Object Look Time" in content
        assert "person" in content
        assert "cat" in content

    def test_correct_column_count(self, tmp_path):
        out = tmp_path / "summary.csv"
        look_counts = {(0, "person"): 30}
        write_summary_csv(str(out), total_frames=100,
                          look_counts=look_counts)
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header row
        header = rows[1]
        assert "category" in header
        assert "participant" in header
        assert "object" in header
        # Data row should have same number of columns as header
        data = rows[2]
        assert len(data) == len(header)

    def test_percentage_calculation(self, tmp_path):
        out = tmp_path / "summary.csv"
        look_counts = {(0, "ball"): 25}
        write_summary_csv(str(out), total_frames=100,
                          look_counts=look_counts)
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        data_row = rows[2]
        pct = float(data_row[-1])
        assert pct == pytest.approx(25.0, abs=0.1)

    def test_empty_look_counts(self, tmp_path):
        out = tmp_path / "summary.csv"
        write_summary_csv(str(out), total_frames=100,
                          look_counts={})
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "summary.csv"
        write_summary_csv(str(out), total_frames=10,
                          look_counts={(0, "x"): 5})
        assert out.exists()

    def test_with_tracker(self, tmp_path):
        """A tracker that provides csv_rows should appear in output."""

        class FakeTracker:
            name = "mutual_gaze"

            def csv_rows(self, total_frames, pid_map=None):
                return [
                    ["category", "value"],
                    ["mutual_gaze", "42"],
                ]

        out = tmp_path / "summary.csv"
        write_summary_csv(str(out), total_frames=100,
                          look_counts={(0, "obj"): 10},
                          all_trackers=[FakeTracker()])
        content = out.read_text()
        assert "Mutual Gaze" in content
        assert "42" in content

    def test_project_mode_columns(self, tmp_path):
        out = tmp_path / "summary.csv"
        look_counts = {(0, "person"): 10}
        write_summary_csv(
            str(out), total_frames=50,
            look_counts=look_counts,
            video_name="test_video",
            conditions="cond_a|cond_b")
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[1]
        assert header[0] == "video_name"
        assert header[1] == "conditions"
        data = rows[2]
        assert data[0] == "test_video"
        assert data[1] == "cond_a|cond_b"
