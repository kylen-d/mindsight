"""Tests for validation run history + compare (W4B phase 4)."""

import json
import os

import pytest

from mindsight.validation import (
    allocate_run_dir,
    run_history,
    settings_diff,
)


def _seed_run(root, name, score, settings):
    run_dir = allocate_run_dir(root, name)
    (run_dir / "score.json").write_text(json.dumps(score))
    if settings is not None:
        (run_dir / "settings.json").write_text(json.dumps(settings))
    return run_dir


def test_settings_diff_ignores_run_targets():
    old = {"conf": 0.35, "snap_dist": 150, "source": "a.mp4",
           "summary": "x", "save_detections": False}
    new = {"conf": 0.5, "snap_dist": 150, "source": "b.mp4",
           "summary": "y", "save_detections": True, "ray_length": 1.2}
    diff = settings_diff(old, new)
    assert diff == {"conf": (0.35, 0.5), "ray_length": (None, 1.2)}


def test_run_history_diffs_against_previous_scored_run(tmp_path):
    _seed_run(tmp_path, "s", {"hit_rate": 0.5}, {"conf": 0.35})
    _seed_run(tmp_path, "s", {"hit_rate": 0.6}, {"conf": 0.5})
    # A run that never scored (crashed mid-run) is skipped entirely.
    unscored = allocate_run_dir(tmp_path, "s")
    assert unscored.name == "run-003"
    _seed_run(tmp_path, "s", {"hit_rate": 0.7}, {"conf": 0.5, "ray_length": 2})

    hist = run_history(tmp_path, "s")
    assert [h["run"] for h in hist] == ["run-001", "run-002", "run-004"]
    assert hist[0]["changed"] == {}
    assert hist[1]["changed"] == {"conf": (0.35, 0.5)}
    assert hist[2]["changed"] == {"ray_length": (None, 2)}


def test_run_history_tolerates_missing_settings(tmp_path):
    _seed_run(tmp_path, "s", {"hit_rate": 0.5}, {"conf": 0.35})
    _seed_run(tmp_path, "s", {"hit_rate": 0.6}, None)      # no snapshot
    _seed_run(tmp_path, "s", {"hit_rate": 0.7}, {"conf": 0.9})
    hist = run_history(tmp_path, "s")
    assert hist[1]["changed"] == {}                        # can't diff
    # Diff resumes against the last run that HAD a snapshot.
    assert hist[2]["changed"] == {"conf": (0.35, 0.9)}


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_history_dialog_renders(qapp, tmp_path):
    from mindsight.GUI.validation_history import ValidationHistoryDialog
    from mindsight.validation import ValidationStore

    store = ValidationStore(tmp_path)
    _seed_run(tmp_path, "s", {"endpoint_px_mean": 70.25, "hit_rate": 0.66},
              {"conf": 0.35})
    _seed_run(tmp_path, "s", {"endpoint_px_mean": 65.0, "hit_rate": 0.7},
              {"conf": 0.5})
    dlg = ValidationHistoryDialog(store, "s")
    assert dlg._table.rowCount() == 2
    # Newest first.
    assert dlg._table.item(0, 0).text() == "run-002"
    assert dlg._table.item(0, 1).text() == "65.0"
    assert "conf" in dlg._table.item(0, 6).text()
    assert dlg._table.item(1, 6).text() == "—"
    # Selection shows the full diff.
    dlg._table.selectRow(0)
    assert "0.35" in dlg._diff_box.toPlainText()
    dlg.reject()
