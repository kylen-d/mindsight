"""Offscreen tests for the Auto-tune sweep dialog (W4C item 1).

The fake worker writes a gaze stream whose error DEPENDS on the swept
knob value it finds in its prepared namespace, so combos score
differently and the winner pick is meaningful.  Under test: the
sequential sweep loop (allocate -> prepare+override -> run -> score ->
manifest), cancel-keeps-completed-scores, the Apply-best round trip,
and reopening the last sweep.
"""

import csv
import json
import os
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

import cv2  # noqa: E402

from mindsight.validation import (  # noqa: E402
    ValidationSet,
    ValidationStore,
    allocate_sweep_path,
    new_sweep_manifest,
    save_sweep,
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


class _KnobWorker(threading.Thread):
    """Error tracks the knob: ray end lands rf_len_gain*20 px right of
    the label at (100, 100), so gain 0.5 -> 10px, gain 1.5 -> 30px."""

    def __init__(self, ns, frame_q, log_q):
        super().__init__(daemon=True)
        self.ns = ns
        self.frame_q = frame_q
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        summary = Path(self.ns.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        gaze = summary.parent / \
            f"{summary.stem.replace('_summary', '')}_gaze.csv"
        end_x = 100 + 20 * float(self.ns.rf_len_gain)
        with open(gaze, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(GAZE_HEADER)
            w.writerow(["", "", 10, "", 0, "P0", "1.0", "0", "0",
                        60, 100, end_x, 100, 0, 0, "", "", "", "", ""])
        for _ in range(3):
            self.frame_q.put(np.zeros((4, 4, 3), np.uint8))
        self.frame_q.put(None)


def _setup(tmp_path):
    store = ValidationStore(tmp_path / "validation")
    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (32, 32))
    vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    vset = ValidationSet(name="s", video=str(video))
    vset.set_label(10, "0", {"x": 100, "y": 100})
    store.save(vset)
    return store


def _provider():
    return Namespace(source="0", summary=None, log=None, save=None,
                     heatmap=None, charts=None, no_dashboard=False,
                     save_detections=False, rf_len_gain=1.0,
                     min_call_gap=30)


def _make(store, applied, worker=_KnobWorker):
    from mindsight.GUI.validation_autotune import AutoTuneDialog
    return AutoTuneDialog(store, "s", _provider, applied.append,
                          worker_factory=worker)


def _set_knob(dlg, row, label_dest, values_text):
    combo, values = dlg._knob_rows[row]
    idx = combo.findData(label_dest)
    assert idx >= 0
    combo.setCurrentIndex(idx)
    values.setText(values_text)


def _drain(dlg):
    for _ in range(500):
        if dlg._worker is not None:
            dlg._worker.join(timeout=5)
        dlg._on_poll()
        if dlg._worker is None and dlg._pending is None:
            if dlg._start_btn.isEnabled():
                return
    raise AssertionError("sweep never finished")


def test_sweep_runs_all_combos_and_picks_winner(qapp, tmp_path):
    store = _setup(tmp_path)
    applied = []
    dlg = _make(store, applied)
    _set_knob(dlg, 0, "rf_len_gain", "1.5, 0.5, 1.0")
    assert "3 combination" in dlg._estimate.text()
    dlg._on_start()
    _drain(dlg)

    runs = sorted((store.root / ".runs" / "s").glob("run-*"))
    assert len(runs) == 3
    manifest = json.loads(
        (store.root / ".runs" / "s" / "sweep-001.json").read_text())
    assert [r["overrides"]["rf_len_gain"] for r in manifest["results"]] \
        == [1.5, 0.5, 1.0]
    means = [r["score"]["endpoint_px_mean"] for r in manifest["results"]]
    assert means == pytest.approx([30.0, 10.0, 20.0])
    assert manifest["winner"] == 1                    # gain 0.5, not first
    # Every combo persisted an ordinary run History understands.
    for r in manifest["results"]:
        assert (store.root / ".runs" / "s" / r["run"]
                / "settings.json").is_file()
    # Table sorted by mean px: winner's gain first.
    assert dlg._table.rowCount() == 3
    assert dlg._table.item(0, 0).text() == "0.5"
    assert dlg._apply_btn.isEnabled()

    # Apply best: full tab namespace with ONLY the winning dest changed.
    dlg._on_apply_best()
    assert len(applied) == 1
    assert applied[0].rf_len_gain == 0.5
    assert applied[0].min_call_gap == 30
    assert applied[0].source == "0"                   # tab ns, not run ns


def test_cancel_mid_sweep_keeps_completed_scores(qapp, tmp_path):
    store = _setup(tmp_path)
    dlg = _make(store, [])
    _set_knob(dlg, 0, "rf_len_gain", "1.5, 0.5, 1.0")
    dlg._on_start()
    assert dlg._worker is not None
    dlg._on_cancel()                    # while combo 1 is in flight
    _drain(dlg)

    manifest = json.loads(
        (store.root / ".runs" / "s" / "sweep-001.json").read_text())
    assert len(manifest["results"]) == 1              # combos 2-3 never ran
    assert manifest["results"][0]["score"]["endpoint_px_mean"] \
        == pytest.approx(30.0)                        # combo 1 still scored
    assert manifest["winner"] == 0
    assert "cancelled" in dlg._status.text()
    assert len(sorted((store.root / ".runs" / "s").glob("run-*"))) == 1


def test_two_knob_sweep_applies_both_dests(qapp, tmp_path):
    store = _setup(tmp_path)
    applied = []
    dlg = _make(store, applied)
    _set_knob(dlg, 0, "rf_len_gain", "1.5, 0.5")
    _set_knob(dlg, 1, "min_call_gap", "15, 45")
    dlg._on_start()
    _drain(dlg)
    manifest = json.loads(
        (store.root / ".runs" / "s" / "sweep-001.json").read_text())
    assert len(manifest["results"]) == 4
    winner = manifest["results"][manifest["winner"]]["overrides"]
    assert winner["rf_len_gain"] == 0.5
    dlg._on_apply_best()
    assert applied[0].rf_len_gain == 0.5
    assert applied[0].min_call_gap == winner["min_call_gap"]


def test_bad_values_block_start(qapp, tmp_path):
    store = _setup(tmp_path)
    dlg = _make(store, [])
    _set_knob(dlg, 0, "min_call_gap", "30, x")
    dlg._on_start()
    assert dlg._worker is None
    assert "Bad value" in dlg._status.text()
    assert not list((store.root / ".runs").glob("**/run-*"))


def test_over_cap_blocks_start(qapp, tmp_path):
    store = _setup(tmp_path)
    dlg = _make(store, [])
    _set_knob(dlg, 0, "rf_len_gain", "1,2,3,4")
    _set_knob(dlg, 1, "min_call_gap", "10,20,30,40")
    assert "cap" in dlg._estimate.text()
    dlg._on_start()
    assert dlg._worker is None


def test_reopen_shows_last_sweep(qapp, tmp_path):
    store = _setup(tmp_path)
    path = allocate_sweep_path(store.root, "s")
    manifest = new_sweep_manifest("s", [("rf_len_gain", [1.0, 1.2])])
    manifest["results"] = [
        {"overrides": {"rf_len_gain": 1.0}, "run": "run-001",
         "score": {"endpoint_px_mean": 12.0}, "error": None},
        {"overrides": {"rf_len_gain": 1.2}, "run": "run-002",
         "score": {"endpoint_px_mean": 9.0}, "error": None},
    ]
    manifest["winner"] = 1
    save_sweep(path, manifest)

    applied = []
    dlg = _make(store, applied)
    assert dlg._table.rowCount() == 2
    assert dlg._table.item(0, 0).text() == "1.2"      # sorted, winner first
    assert dlg._apply_btn.isEnabled()
    dlg._on_apply_best()
    assert applied[0].rf_len_gain == 1.2
