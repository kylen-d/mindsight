"""SP2.1 Batch E: run_project resume bookkeeping (slow).

Drives ``mindsight.project.runner.run_project`` end-to-end with MONKEYPATCHED
build/run functions (no real models) over a tiny cv2-generated clip, asserting
the ledger's skip / reprocess / archive decisions.  The REAL end-to-end with
models is the G-LEDGER gate; this pins the orchestration bookkeeping cheaply.
"""
from __future__ import annotations

import json

import pytest

from mindsight.cli_flags import parse_cli
from mindsight.project.runner import run_project

pytestmark = pytest.mark.slow


def _make_clip(path, frames=15, w=64, h=48):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
    vw.release()


def _project(tmp_path):
    proj = tmp_path / "proj"
    _make_clip(proj / "Inputs" / "Videos" / "clip.mp4")
    return proj


def _build(ns):
    """Stand-in for factory.build_from_namespace -> the 14-tuple run_project unpacks."""
    return tuple([None] * 14)


def _make_run(calls):
    """A run_fn stub that records calls and writes minimal tidy outputs."""
    def _run(source, *args, **kwargs):
        from pathlib import Path
        run_output = args[6]          # (yolo..tracker_cfg) then run_output
        calls.append(source)
        summ = Path(run_output.summary_path)
        summ.parent.mkdir(parents=True, exist_ok=True)
        summ.write_text("video_name,conditions,phenomenon,participant,"
                        "partner,object,metric,value\n")
        Path(run_output.log_path).write_text("frame,t_seconds\n")
    return _run


def _ledger_data(proj):
    path = proj / "Outputs" / "_run" / "ledger.json"
    return json.loads(path.read_text())


def test_first_run_marks_done(tmp_path):
    proj = _project(tmp_path)
    calls: list = []
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    assert calls == [str(proj / "Inputs" / "Videos" / "clip.mp4")]
    data = _ledger_data(proj)
    rec = data["videos"]["clip.mp4"]
    assert data["ledger_version"] == 1
    assert rec["status"] == "done"
    assert rec["manifest"].endswith("clip_manifest.json")


def test_second_run_skips_unchanged(tmp_path, capsys):
    proj = _project(tmp_path)
    calls: list = []
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    assert len(calls) == 1
    # Second run, identical config -> skip.
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    assert len(calls) == 1                      # run_fn NOT called again
    assert "Skipping clip.mp4 (done, config unchanged)" in capsys.readouterr().out
    assert _ledger_data(proj)["videos"]["clip.mp4"]["status"] == "done"


def test_in_progress_is_reprocessed(tmp_path):
    proj = _project(tmp_path)
    calls: list = []
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    # Simulate a kill -9 mid-run: force the record back to in_progress.
    path = proj / "Outputs" / "_run" / "ledger.json"
    data = json.loads(path.read_text())
    data["videos"]["clip.mp4"]["status"] = "in_progress"
    path.write_text(json.dumps(data))
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    assert len(calls) == 2                       # reprocessed in place
    assert _ledger_data(proj)["videos"]["clip.mp4"]["status"] == "done"


def test_config_change_archives_and_reprocesses(tmp_path):
    proj = _project(tmp_path)
    calls: list = []
    run_project(proj, _make_run(calls), _build, parse_cli([]))
    # A processing-config change flips config_hash -> redo_archive.
    run_project(proj, _make_run(calls), _build, parse_cli(["--conf", "0.30"]))
    assert len(calls) == 2
    superseded = proj / "Outputs" / "_run" / "superseded"
    assert superseded.is_dir()
    archived = list(superseded.iterdir())
    assert len(archived) == 1 and archived[0].name.endswith("_clip")
    assert (archived[0] / "clip_summary.csv").exists()
