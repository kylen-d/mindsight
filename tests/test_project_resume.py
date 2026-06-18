"""SP3.1 Batch B: iter_project_runs resume bookkeeping (slow).

Drives ``mindsight.project.runner.iter_project_runs`` end-to-end with the model
build + per-video ``Pipeline`` MONKEYPATCHED (no real models) over a tiny
cv2-generated clip, exhausting the event stream and asserting the ledger's
skip / reprocess / archive decisions plus the emitted events.  The REAL
end-to-end with models is the G-LEDGER gate; this pins the orchestration
bookkeeping cheaply.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import mindsight.pipeline as _pipeline_mod
from mindsight.cli_flags import parse_cli
from mindsight.project.events import (
    BatchDone,
    BatchStarted,
    VideoArchived,
    VideoDone,
    VideoSkipped,
    VideoStarted,
)
from mindsight.project.runner import iter_project_runs

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


@pytest.fixture
def fake_models(monkeypatch):
    """Patch build_from_namespace -> the 14-tuple and Pipeline -> a stub that
    records each source and writes minimal tidy per-video outputs (no models).

    Returns the shared ``calls`` list (one entry per Pipeline.run invocation).
    """
    calls: list = []

    def _build(_ns):
        return tuple([None] * 14)

    class _StubPipeline:
        def __init__(self, *, output_cfg, **_kw):
            self._out = output_cfg

        def run(self, source, *, options=None, cancel=None):
            calls.append(source)
            summ = Path(self._out.summary_path)
            summ.parent.mkdir(parents=True, exist_ok=True)
            summ.write_text("video_name,conditions,phenomenon,participant,"
                            "partner,object,metric,value\n")
            Path(self._out.log_path).write_text("frame,t_seconds\n")
            return iter(())          # a video with no yielded frames

    monkeypatch.setattr(_pipeline_mod, "build_from_namespace", _build)
    monkeypatch.setattr(_pipeline_mod, "Pipeline", _StubPipeline)
    return calls


def _run(proj, ns, *, resume=True):
    """Exhaust the event stream, returning the list of emitted events."""
    return list(iter_project_runs(proj, ns, resume=resume))


def _ledger_data(proj):
    path = proj / "Outputs" / "_run" / "ledger.json"
    return json.loads(path.read_text())


def test_first_run_marks_done(tmp_path, fake_models):
    proj = _project(tmp_path)
    events = _run(proj, parse_cli([]))
    assert fake_models == [str(proj / "Inputs" / "Videos" / "clip.mp4")]
    # Event stream shape: BatchStarted, VideoStarted, VideoDone, BatchDone.
    assert isinstance(events[0], BatchStarted) and events[0].total == 1
    assert any(isinstance(e, VideoStarted) and e.run_id == "clip.mp4"
               for e in events)
    assert any(isinstance(e, VideoDone) and e.run_id == "clip.mp4"
               for e in events)
    assert isinstance(events[-1], BatchDone)
    data = _ledger_data(proj)
    rec = data["videos"]["clip.mp4"]
    assert data["ledger_version"] == 1
    assert rec["status"] == "done"
    assert rec["manifest"].endswith("clip_manifest.json")


def test_second_run_skips_unchanged(tmp_path, capsys, fake_models):
    proj = _project(tmp_path)
    _run(proj, parse_cli([]))
    assert len(fake_models) == 1
    capsys.readouterr()
    # Second run, identical config -> skip.
    events = _run(proj, parse_cli([]))
    assert len(fake_models) == 1                 # Pipeline.run NOT called again
    assert "Skipping clip.mp4 (done, config unchanged)" in capsys.readouterr().out
    assert any(isinstance(e, VideoSkipped) and e.run_id == "clip.mp4"
               for e in events)
    assert _ledger_data(proj)["videos"]["clip.mp4"]["status"] == "done"


def test_in_progress_is_reprocessed(tmp_path, fake_models):
    proj = _project(tmp_path)
    _run(proj, parse_cli([]))
    # Simulate a kill -9 mid-run: force the record back to in_progress.
    path = proj / "Outputs" / "_run" / "ledger.json"
    data = json.loads(path.read_text())
    data["videos"]["clip.mp4"]["status"] = "in_progress"
    path.write_text(json.dumps(data))
    _run(proj, parse_cli([]))
    assert len(fake_models) == 2                  # reprocessed in place
    assert _ledger_data(proj)["videos"]["clip.mp4"]["status"] == "done"


def test_config_change_archives_and_reprocesses(tmp_path, fake_models):
    proj = _project(tmp_path)
    _run(proj, parse_cli([]))
    # A processing-config change flips config_hash -> redo_archive.
    events = _run(proj, parse_cli(["--conf", "0.30"]))
    assert len(fake_models) == 2
    assert any(isinstance(e, VideoArchived) and e.run_id == "clip.mp4"
               for e in events)
    superseded = proj / "Outputs" / "_run" / "superseded"
    assert superseded.is_dir()
    archived = list(superseded.iterdir())
    assert len(archived) == 1 and archived[0].name.endswith("_clip")
    assert (archived[0] / "clip_summary.csv").exists()


def test_no_resume_reprocesses_without_archive(tmp_path, fake_models):
    proj = _project(tmp_path)
    _run(proj, parse_cli([]))
    _run(proj, parse_cli(["--no-resume"]), resume=False)
    assert len(fake_models) == 2                  # everything reprocessed
    assert not (proj / "Outputs" / "_run" / "superseded").exists()
