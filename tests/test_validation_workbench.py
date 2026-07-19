"""Offscreen tests for the validation workbench pane (W4B phase 3).

A fake worker replaces GazeWorker: on start it writes a synthetic gaze
stream into the allocated run dir (proving the pane handed it a
correctly prepared namespace) and pushes the completion sentinel.  What
is under test: the validate flow end-to-end (allocate -> prepare ->
run -> score -> table), the previous-run comparison column, and set
management plumbing.
"""

import csv
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

import cv2  # noqa: E402

from mindsight.validation import (  # noqa: E402
    ValidationSet,
    ValidationStore,
)

GAZE_HEADER = ["video_name", "conditions", "frame", "t_seconds", "face_idx",
               "participant_label", "gaze_conf", "gaze_pitch", "gaze_yaw",
               "origin_x", "origin_y", "ray_end_x", "ray_end_y",
               "ray_snapped", "ray_extended", "trust", "accepted_inference",
               "inout_score", "depth_at_end", "hit_objects"]


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _FakeWorker(threading.Thread):
    """Writes a gaze stream where the prepared namespace points, then
    signals completion the way GazeWorker does (None sentinel)."""

    end_xy = (110, 110)

    def __init__(self, ns, frame_q, log_q):
        super().__init__(daemon=True)
        self.ns = ns
        self.frame_q = frame_q

    def run(self):
        summary = Path(self.ns.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        gaze = summary.parent / f"{summary.stem.replace('_summary', '')}_gaze.csv"
        with open(gaze, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(GAZE_HEADER)
            w.writerow(["", "", 10, "", 0, "P0", "1.0", "0", "0",
                        100, 100, *type(self).end_xy, 0, 0, "", "", "", "", ""])
        self.frame_q.put(None)


def _make(qapp, tmp_path):
    from mindsight.GUI.validation_workbench import ValidationWorkbench
    store = ValidationStore(tmp_path / "validation")
    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (32, 32))
    vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    vset = ValidationSet(name="s", video=str(video))
    vset.set_label(10, "0", {"x": 110, "y": 100})
    store.save(vset)

    ns_holder = {"ns": None}

    def provider():
        from argparse import Namespace
        return Namespace(source="0", summary=None, log=None, save=None,
                         heatmap=None, charts=None, no_dashboard=False,
                         save_detections=False)

    wb = ValidationWorkbench(namespace_provider=provider, store=store,
                             worker_factory=_FakeWorker)
    return wb, store, ns_holder


def _drain(wb):
    # Deterministic completion: the fake worker finishes quickly; poll
    # until the sentinel is consumed.
    for _ in range(200):
        wb._on_poll()
        if wb._worker is None or not wb._worker.is_alive():
            wb._on_poll()
            if wb._worker is None:
                return
    raise AssertionError("workbench never finished")


def test_validate_flow_scores_and_fills_table(qapp, tmp_path):
    wb, store, _ = _make(qapp, tmp_path)
    assert wb._set_combo.currentData() == "s"
    wb._on_validate()
    assert wb._worker is not None
    wb._worker.join(timeout=5)
    _drain(wb)

    runs = sorted((store.root / ".runs" / "s").glob("run-*"))
    assert len(runs) == 1
    score = json.loads((runs[0] / "score.json").read_text())
    assert score["scored_points"] == 1
    assert score["endpoint_px_mean"] == pytest.approx(10.0)
    assert (runs[0] / "settings.json").is_file()
    # Table: run column filled, prev column empty (first run).
    assert wb._table.item(0, 0).text() == "10.0"
    assert wb._table.item(0, 1).text() == "—"
    assert wb._validate_btn.isEnabled()

    # Second run with a worse fake result: prev column = first run.
    _FakeWorker.end_xy = (130, 100)
    try:
        wb._on_validate()
        wb._worker.join(timeout=5)
        _drain(wb)
    finally:
        _FakeWorker.end_xy = (110, 110)
    assert wb._table.item(0, 0).text() == "20.0"
    assert wb._table.item(0, 1).text() == "10.0"


def test_validate_without_sets_is_graceful(qapp, tmp_path):
    from argparse import Namespace

    from mindsight.GUI.validation_workbench import ValidationWorkbench
    wb = ValidationWorkbench(
        namespace_provider=lambda: Namespace(),
        store=ValidationStore(tmp_path / "empty"),
        worker_factory=_FakeWorker)
    wb._on_validate()
    assert wb._worker is None
    assert "Create a validation set" in wb._status.text()


def test_delete_keeps_run_history(qapp, tmp_path):
    wb, store, _ = _make(qapp, tmp_path)
    wb._on_validate()
    wb._worker.join(timeout=5)
    _drain(wb)
    runs_dir = store.root / ".runs" / "s"
    assert runs_dir.is_dir()
    store.delete("s")
    wb.refresh_sets()
    assert wb._set_combo.count() == 0
    assert runs_dir.is_dir()          # results survive set deletion
