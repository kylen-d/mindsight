"""Tests for validation scoring + run plumbing (W4B phase 3).

The scorer is pinned against hand-computed values on a tiny synthetic
run (gaze + detections streams written directly), including the two
suite-only metrics (MAE degrees, object IoU) and the eval-harness
matching semantics (digit keys -> face_idx, custom keys ->
participant_label).
"""

import csv
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest

from mindsight.validation import (
    ValidationSet,
    ValidationSetError,
    allocate_run_dir,
    latest_score,
    list_run_dirs,
    prepare_validation_namespace,
    score_and_persist,
    score_run,
)

GAZE_HEADER = ["video_name", "conditions", "frame", "t_seconds", "face_idx",
               "participant_label", "gaze_conf", "gaze_pitch", "gaze_yaw",
               "origin_x", "origin_y", "ray_end_x", "ray_end_y",
               "ray_snapped", "ray_extended", "trust", "accepted_inference",
               "inout_score", "depth_at_end", "hit_objects"]


def _write_gaze(run_dir: Path, rows):
    with open(run_dir / "clip_gaze.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(GAZE_HEADER)
        for frame, tid, plabel, ox, oy, ex, ey, inout in rows:
            w.writerow(["", "", frame, "", tid, plabel, "1.0", "0", "0",
                        ox, oy, ex, ey, 0, 0, "", "", inout, "", ""])


def _write_dets(run_dir: Path, rows):
    with open(run_dir / "clip_detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_name", "conditions", "frame", "t_seconds",
                    "class", "conf", "x1", "y1", "x2", "y2"])
        for frame, x1, y1, x2, y2 in rows:
            w.writerow(["", "", frame, "", "thing", "0.9", x1, y1, x2, y2])


def _vset() -> ValidationSet:
    vset = ValidationSet(name="s", video="/tmp/clip.mp4")
    vset.set_label(10, "0", {"x": 110, "y": 100})   # digit -> face_idx
    vset.set_label(10, "S9", {"x": 200, "y": 250})  # custom -> label col
    vset.set_label(20, "0", "offscreen")
    vset.set_label(20, "1", "uncertain")            # excluded
    vset.add_object(10, "plate", (0, 0, 100, 100))
    return vset


def test_score_run_pinned(tmp_path):
    _write_gaze(tmp_path, [
        # frame 10, tid 0: origin (100,100), end (110,110); label (110,100)
        # -> err 10; MAE = angle between (10,10) and (10,0) = 45 deg.
        (10, 0, "P0", 100, 100, 110, 110, "0.9"),
        # frame 10, custom S9: end (200,240) vs label (200,250) -> err 10;
        # vectors (0,-10) vs (0,0)->... origin (200,200): vp=(0,40) wait
        (10, 5, "S9", 200, 200, 200, 240, "0.8"),
        # frame 20 tid 0: offscreen label -> inout goes to the AUC pool.
        (20, 0, "P0", 0, 0, 5, 5, "0.2"),
    ])
    _write_dets(tmp_path, [
        (10, 50, 0, 150, 100),     # IoU vs (0,0,100,100): 50x100/(150x100+... )
        (10, 300, 300, 320, 320),  # irrelevant box
    ])
    r = score_run(_vset(), tmp_path, "clip")
    assert r["scored_points"] == 2
    assert r["endpoint_px_mean"] == pytest.approx(10.0)
    assert r["hit_rate"] == 1.0
    # MAE: 45 deg for row 1; row 2: vp=(0,40), vt=(0,50) -> 0 deg.
    assert r["mae_deg_mean"] == pytest.approx(22.5)
    # IoU: inter 50*100=5000; union 100*100 + 100*100 - 5000 = 15000.
    assert r["object_iou_mean"] == pytest.approx(5000 / 15000)
    assert r["object_boxes_scored"] == 1
    # AUC: offscreen inout 0.2 vs onscreen [0.9, 0.8] -> perfectly ranked.
    assert r["offscreen_auc"] == pytest.approx(1.0)
    assert r["per_participant_mean_px"] == {
        "0": pytest.approx(10.0), "S9": pytest.approx(10.0)}
    assert math.isclose(r["endpoint_px_median"], 10.0)


def test_score_run_errors(tmp_path):
    with pytest.raises(ValidationSetError, match="no gaze stream"):
        score_run(_vset(), tmp_path, "clip")
    _write_gaze(tmp_path, [(99, 0, "P0", 0, 0, 1, 1, "")])
    with pytest.raises(ValidationSetError, match="No scorable"):
        score_run(_vset(), tmp_path, "clip")


def test_run_dir_allocation_and_latest_score(tmp_path):
    r1 = allocate_run_dir(tmp_path, "My Set")
    r2 = allocate_run_dir(tmp_path, "My Set")
    assert [r1.name, r2.name] == ["run-001", "run-002"]
    assert list_run_dirs(tmp_path, "My Set") == [r1, r2]
    assert latest_score(tmp_path, "My Set") is None
    (r1 / "score.json").write_text(json.dumps({"hit_rate": 0.5}))
    assert latest_score(tmp_path, "My Set") == {"hit_rate": 0.5}
    (r2 / "score.json").write_text(json.dumps({"hit_rate": 0.7}))
    assert latest_score(tmp_path, "My Set") == {"hit_rate": 0.7}


def test_prepare_validation_namespace(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    vset = ValidationSet(name="s", video=str(video))
    ns = Namespace(source="0", summary=None, log=None, save="out.mp4",
                   heatmap=True, charts=True, no_dashboard=False,
                   save_detections=False, conf=0.42)
    run_dir = tmp_path / "run-001"
    ns2 = prepare_validation_namespace(ns, vset, run_dir)
    assert ns2.source == str(video)
    assert ns2.summary == str(run_dir / "clip_summary.csv")
    assert ns2.log == str(run_dir / "clip_events.csv")
    assert ns2.save is None and ns2.heatmap is None and ns2.charts is None
    assert ns2.no_dashboard is True and ns2.save_detections is True
    assert ns2.conf == 0.42                    # user settings preserved
    assert ns.save == "out.mp4"                # original untouched
    with pytest.raises(ValidationSetError, match="not found"):
        prepare_validation_namespace(
            ns, ValidationSet(name="x", video="/nope.mp4"), run_dir)


def test_score_and_persist_writes_artifacts(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    vset = _vset()
    vset.video = str(video)
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    _write_gaze(run_dir, [(10, 0, "P0", 100, 100, 110, 110, "")])
    ns = Namespace(conf=0.35, _private="drop", weird=object())
    result = score_and_persist(vset, run_dir, ns=ns)
    saved = json.loads((run_dir / "score.json").read_text())
    assert saved["scored_points"] == result["scored_points"] == 1
    settings = json.loads((run_dir / "settings.json").read_text())
    assert settings["conf"] == 0.35
    assert "_private" not in settings
    assert isinstance(settings["weird"], str)   # stringified, not dropped
