"""Tests for the opt-in detections side stream (v1.1 W4B, validation phase 2).

Default-off contract is the load-bearing part: with the flag off the
collector list is None, nothing accumulates, no file is written — every
existing golden stays byte-identical.  With it on, one row per detection
per frame lands in ``{stem}_detections.csv`` (header-only when a run
finds nothing, so IoU scoring can tell "found nothing" from "not run").
"""

import csv

from mindsight.outputs.csv_output import (
    DETECTIONS_STREAM_HEADER,
    write_summary_tables,
)
from mindsight.outputs.data_pipeline import collect_frame_data
from mindsight.pipeline import RunOptions

_DETS = [
    {"class_name": "plate", "conf": 0.91, "x1": 10, "y1": 20, "x2": 60, "y2": 80},
    {"class_name": "cup", "conf": 0.5, "x1": 100, "y1": 110, "x2": 130, "y2": 150},
]


def _collect(ctx):
    collect_frame_data(ctx, log_csv=None, frame_no=7, hit_events=[],
                       face_track_ids=[], persons_gaze=[])


def test_collector_appends_rows_when_enabled():
    ctx = {"detections_stream_rows": [], "all_dets": _DETS, "video_fps": 30.0}
    _collect(ctx)
    rows = ctx["detections_stream_rows"]
    assert rows == [
        [7, "0.233", "plate", "0.910", 10, 20, 60, 80],
        [7, "0.233", "cup", "0.500", 100, 110, 130, 150],
    ]


def test_collector_inert_when_disabled():
    ctx = {"detections_stream_rows": None, "all_dets": _DETS, "video_fps": 30.0}
    _collect(ctx)                                    # must not raise
    assert ctx["detections_stream_rows"] is None


def test_stream_file_written_and_header_only_when_empty(tmp_path):
    summary = tmp_path / "clip_summary.csv"
    write_summary_tables(summary, total_frames=10, fps=30.0, look_counts={},
                         detections_stream=[[7, "0.233", "plate", "0.910",
                                             10, 20, 60, 80]])
    out = tmp_path / "clip_detections.csv"
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["video_name", "conditions"] + DETECTIONS_STREAM_HEADER
    assert rows[1][2:] == ["7", "0.233", "plate", "0.910",
                           "10", "20", "60", "80"]

    # Enabled but zero detections: header-only file, not absence.
    write_summary_tables(tmp_path / "empty_summary.csv", total_frames=1,
                         fps=30.0, look_counts={}, detections_stream=[])
    assert list(csv.reader((tmp_path / "empty_detections.csv").open())) == [
        ["video_name", "conditions"] + DETECTIONS_STREAM_HEADER]

    # Default (None): no file at all.
    write_summary_tables(tmp_path / "off_summary.csv", total_frames=1,
                         fps=30.0, look_counts={}, detections_stream=None)
    assert not (tmp_path / "off_detections.csv").exists()


def test_run_options_and_flag_default():
    # Excluded run-loop toggle (like --lite-overlay): lives on RunOptions,
    # deliberately NOT in the schema/OutputConfig, so canonical_hash and
    # the resume ledger never move because of it.
    assert RunOptions().save_detections is False
    assert RunOptions(save_detections=True).save_detections is True

    from mindsight.cli_flags import build_parser
    ns = build_parser().parse_args([])
    assert ns.save_detections is False
    ns = build_parser().parse_args(["--save-detections"])
    assert ns.save_detections is True
