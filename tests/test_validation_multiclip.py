"""Multi-video / whole-project validation sets (W4C user request).

Store format 2 (clips array) + format-1 byte-stability for single-clip
sets, project discovery, pooled multi-clip scoring with per-video
breakdown, and the workbench/auto-tune sequential clip chains.
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
    ValidationSetError,
    ValidationStore,
    clips_from_project,
    prepare_clip_namespace,
    score_run,
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


def _write_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         30, (32, 32))
    vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    return path


def _write_gaze(run_dir: Path, stem: str, end_xy):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{stem}_gaze.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(GAZE_HEADER)
        w.writerow(["", "", 10, "", 0, "P0", "1.0", "0", "0",
                    100, 100, *end_xy, 0, 0, "", "", "", "", ""])


# ── Store: formats ───────────────────────────────────────────────────────────

def test_single_clip_saves_format1_byte_stable(tmp_path):
    """A pre-W4C format-1 file loads and re-saves byte-identically (the
    set file stays a valid eval-harness labels file)."""
    store = ValidationStore(tmp_path)
    old = {
        "format": 1,
        "name": "s",
        "video": "/tmp/a.mp4",
        "every": 10,
        "note": "",
        "labels": {"10": {"0": {"x": 1, "y": 2}, "1": "offscreen"}},
        "objects": {"10": [{"name": "cup", "x1": 1, "y1": 1,
                            "x2": 5, "y2": 5}]},
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(old, indent=2) + "\n")
    vset = store.load("s")
    assert len(vset.clips) == 1 and vset.video == "/tmp/a.mp4"
    store.save(vset)
    assert json.loads(path.read_text()) == old


def test_multi_clip_saves_format2_round_trip(tmp_path):
    store = ValidationStore(tmp_path)
    vset = ValidationSet(name="multi", participants=["S70", "S71"])
    a = vset.add_clip("/tmp/a.mp4", every=10)
    b = vset.add_clip("/tmp/b.mp4")
    a.set_label(10, "0", {"x": 1, "y": 2})
    b.set_label(20, "S71", "offscreen")
    store.save(vset)

    raw = json.loads((tmp_path / "multi.json").read_text())
    assert raw["format"] == 2 and len(raw["clips"]) == 2
    assert raw["participants"] == ["S70", "S71"]

    back = store.load("multi")
    assert [c.video for c in back.clips] == ["/tmp/a.mp4", "/tmp/b.mp4"]
    assert back.participants == ["S70", "S71"]
    assert back.clips[0].labels[10]["0"] == {"x": 1, "y": 2}
    assert back.clips[1].labels[20]["S71"] == "offscreen"
    assert back.total_frames() == 2 and back.point_label_count() == 1
    infos = store.list_sets()
    assert infos[0]["videos"] == 2 and infos[0]["frames"] == 2


def test_clip_stems_deduplicate_same_filename(tmp_path):
    vset = ValidationSet(name="dup")
    vset.add_clip("/tmp/one/clip.mp4")
    vset.add_clip("/tmp/two/clip.mp4")
    assert vset.clip_stems() == ["clip", "clip-1"]
    ns_a = prepare_clip_namespace(
        Namespace(source="0", summary=None, log=None, save=None,
                  heatmap=None, charts=None, no_dashboard=False,
                  save_detections=False),
        str(_write_video(tmp_path / "one" / "clip.mp4")),
        tmp_path / "run", "clip")
    assert "clip_summary.csv" in ns_a.summary


def test_clips_from_project_runs_layout(tmp_path):
    proj = tmp_path / "proj"
    for rid, labels in (("run1", {0: "S70", 1: "S71"}), ("run2", None)):
        folder = proj / "Inputs" / "Runs" / rid
        _write_video(folder / f"{rid}.mp4")
        if labels:
            folder.joinpath("run.yaml").write_text(
                "participants:\n" + "".join(
                    f"  {k}: {v}\n" for k, v in labels.items()))
    clips, participants = clips_from_project(proj)
    assert [c["run_id"] for c in clips] == ["run1", "run2"]
    assert clips[0]["pid_map"] == {0: "S70", 1: "S71"}
    assert participants == ["S70", "S71"]


def test_clips_from_project_legacy_flat_and_empty(tmp_path):
    proj = tmp_path / "flat"
    _write_video(proj / "Inputs" / "Videos" / "b.mp4")
    _write_video(proj / "Inputs" / "Videos" / "a.mp4")
    clips, participants = clips_from_project(proj)
    assert [Path(c["video"]).name for c in clips] == ["a.mp4", "b.mp4"]
    assert participants == []
    with pytest.raises(ValidationSetError, match="No staged videos"):
        clips_from_project(tmp_path / "nothing")


# ── Scoring: pooled + per-video ──────────────────────────────────────────────

def _two_clip_set(tmp_path):
    vset = ValidationSet(name="mc")
    a = vset.add_clip(str(_write_video(tmp_path / "a.mp4")))
    b = vset.add_clip(str(_write_video(tmp_path / "b.mp4")))
    a.set_label(10, "0", {"x": 110, "y": 100})   # fake end (110,100) -> 0px
    b.set_label(10, "0", {"x": 140, "y": 100})
    return vset


def test_score_run_pools_and_breaks_down_per_video(tmp_path):
    vset = _two_clip_set(tmp_path)
    run = tmp_path / "run-001"
    _write_gaze(run, "a", (110, 100))            # err 0 vs a's label
    _write_gaze(run, "b", (110, 100))            # err 30 vs b's label
    result = score_run(vset, run)
    assert result["scored_points"] == 2
    assert result["endpoint_px_mean"] == pytest.approx(15.0)
    assert result["videos_scored"] == 2
    assert result["per_video"]["a"]["endpoint_px_mean"] == pytest.approx(0.0)
    assert result["per_video"]["b"]["endpoint_px_mean"] == pytest.approx(30.0)
    assert "skipped_videos" not in result


def test_score_run_skips_missing_clip_stream(tmp_path):
    vset = _two_clip_set(tmp_path)
    run = tmp_path / "run-001"
    _write_gaze(run, "a", (110, 100))            # b never ran (cancel)
    result = score_run(vset, run)
    assert result["scored_points"] == 1
    assert result["skipped_videos"] == ["b"]
    assert result["videos_scored"] == 1


def test_score_run_all_missing_raises(tmp_path):
    vset = _two_clip_set(tmp_path)
    with pytest.raises(ValidationSetError, match="no gaze stream"):
        score_run(vset, tmp_path / "empty-run")


# ── Workbench: sequential clips ──────────────────────────────────────────────

class _StemWorker(threading.Thread):
    """Writes a gaze row whose endpoint depends on the STEM it was
    pointed at, proving each clip ran with its own namespace."""

    ends = {"a": (110, 100), "b": (110, 100)}
    started: list = []

    def __init__(self, ns, frame_q, log_q):
        super().__init__(daemon=True)
        self.ns = ns
        self.frame_q = frame_q
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        stem = Path(self.ns.summary).stem.replace("_summary", "")
        type(self).started.append(stem)
        _write_gaze(Path(self.ns.summary).parent, stem,
                    type(self).ends[stem])
        for _ in range(2):
            self.frame_q.put(np.zeros((4, 4, 3), np.uint8))
        self.frame_q.put(None)


def _make_workbench(tmp_path, vset):
    from mindsight.GUI.validation_workbench import ValidationWorkbench
    store = ValidationStore(tmp_path / "validation")
    store.save(vset)

    def provider():
        return Namespace(source="0", summary=None, log=None, save=None,
                         heatmap=None, charts=None, no_dashboard=False,
                         save_detections=False)

    return ValidationWorkbench(namespace_provider=provider, store=store,
                               worker_factory=_StemWorker), store


def _drain_wb(wb):
    for _ in range(500):
        if wb._worker is not None:
            wb._worker.join(timeout=5)
        wb._on_poll()
        if wb._worker is None and wb._pending is None:
            return
    raise AssertionError("workbench never finished")


def test_workbench_validates_all_clips_into_one_run(qapp, tmp_path):
    _StemWorker.started = []
    vset = _two_clip_set(tmp_path)
    wb, store = _make_workbench(tmp_path, vset)
    wb._on_validate()
    _drain_wb(wb)
    assert _StemWorker.started == ["a", "b"]
    runs = sorted((store.root / ".runs" / "mc").glob("run-*"))
    assert len(runs) == 1
    score = json.loads((runs[0] / "score.json").read_text())
    assert score["scored_points"] == 2
    assert score["per_video"]["b"]["endpoint_px_mean"] == pytest.approx(30.0)


def test_workbench_cancel_skips_remaining_clips(qapp, tmp_path):
    _StemWorker.started = []
    vset = _two_clip_set(tmp_path)
    wb, store = _make_workbench(tmp_path, vset)
    wb._on_validate()
    wb.stop()                       # cancel while clip 'a' is in flight
    _drain_wb(wb)
    assert _StemWorker.started == ["a"]          # 'b' never started
    runs = sorted((store.root / ".runs" / "mc").glob("run-*"))
    score = json.loads((runs[0] / "score.json").read_text())
    assert score["scored_points"] == 1
    assert score["skipped_videos"] == ["b"]
    assert "skipped" in wb._status.text()


# ── Auto-tune: every combo runs every clip ───────────────────────────────────

def test_autotune_combo_runs_all_clips(qapp, tmp_path):
    from mindsight.GUI.validation_autotune import AutoTuneDialog
    _StemWorker.started = []
    vset = _two_clip_set(tmp_path)
    store = ValidationStore(tmp_path / "validation")
    store.save(vset)

    def provider():
        return Namespace(source="0", summary=None, log=None, save=None,
                         heatmap=None, charts=None, no_dashboard=False,
                         save_detections=False, rf_len_gain=1.0)

    dlg = AutoTuneDialog(store, "mc", provider, lambda ns: None,
                         worker_factory=_StemWorker)
    combo, values = dlg._knob_rows[0]
    combo.setCurrentIndex(combo.findData("rf_len_gain"))
    values.setText("1.0, 1.2")
    dlg._on_start()
    for _ in range(500):
        if dlg._worker is not None:
            dlg._worker.join(timeout=5)
        dlg._on_poll()
        if dlg._worker is None and dlg._start_btn.isEnabled():
            break
    else:
        raise AssertionError("sweep never finished")

    assert _StemWorker.started == ["a", "b", "a", "b"]   # 2 combos x 2 clips
    manifest = json.loads(
        (store.root / ".runs" / "mc" / "sweep-001.json").read_text())
    assert len(manifest["results"]) == 2
    for entry in manifest["results"]:
        assert entry["score"]["scored_points"] == 2
        run = store.root / ".runs" / "mc" / entry["run"]
        assert (run / "a_gaze.csv").is_file() and (run / "b_gaze.csv").is_file()
