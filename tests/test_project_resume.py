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


# ── SP3.1 Batch E Step 10: per-run output mirroring (Q3) ─────────────────────

@pytest.fixture
def fake_models_rows(monkeypatch):
    """Like ``fake_models`` but the stub writes one tidy DATA row per run so the
    global aggregation + By Condition split have content to work with."""
    calls: list = []

    def _build(_ns):
        return tuple([None] * 14)

    class _StubPipeline:
        def __init__(self, *, output_cfg, **_kw):
            self._out = output_cfg

        def run(self, source, *, options=None, cancel=None):
            calls.append(source)
            vn, cond = self._out.video_name, self._out.conditions or ""
            summ = Path(self._out.summary_path)
            summ.parent.mkdir(parents=True, exist_ok=True)
            summ.write_text("video_name,conditions,metric,value\n"
                            f"{vn},{cond},frames,10\n")
            Path(self._out.log_path).write_text(
                "video_name,conditions,frame\n" f"{vn},{cond},0\n")
            return iter(())

    monkeypatch.setattr(_pipeline_mod, "build_from_namespace", _build)
    monkeypatch.setattr(_pipeline_mod, "Pipeline", _StubPipeline)
    return calls


def _run_folder_project(tmp_path):
    proj = tmp_path / "rfproj"
    for rid, cond, pid in (("dyad07_collab", "collab", "S70"),
                           ("dyad07_solo", "solo", "S80")):
        folder = proj / "Inputs" / "Runs" / rid
        _make_clip(folder / "clip.mp4")
        (folder / "run.yaml").write_text(
            f"participants: {{0: {pid}}}\nconditions: [{cond}]\n")
    return proj


def test_run_folder_outputs_mirror_and_aggregate(tmp_path, fake_models_rows):
    proj = _run_folder_project(tmp_path)
    events = _run(proj, parse_cli([]))
    out = proj / "Outputs"

    # Per-run outputs mirror under Outputs/Runs/<run_id>/ (Q3).
    for rid in ("dyad07_collab", "dyad07_solo"):
        run_dir = out / "Runs" / rid
        assert (run_dir / f"{rid}_summary.csv").exists()
        assert (run_dir / f"{rid}_Events.csv").exists()
        assert (run_dir / f"{rid}_manifest.json").exists()

    # Nothing leaks into the flat legacy per-video location.
    flat = out / "CSV Files"
    assert not (flat / "dyad07_collab_summary.csv").exists()

    # Globals aggregate into Outputs/CSV Files/ from the per-run dirs.
    gsum = (flat / "Global_summary.csv").read_text().splitlines()
    assert gsum[0] == "video_name,conditions,metric,value"
    assert any(r.startswith("dyad07_collab,collab,") for r in gsum[1:])
    assert any(r.startswith("dyad07_solo,solo,") for r in gsum[1:])

    # By Condition split lands in Outputs/By Condition/ (unchanged location).
    assert (out / "By Condition" / "collab_summary.csv").exists()
    assert (out / "By Condition" / "solo_summary.csv").exists()

    # Ledger is keyed by run_id at the unchanged Outputs/_run/ location.
    led = _ledger_data(proj)["videos"]
    assert set(led) == {"dyad07_collab", "dyad07_solo"}
    assert all(v["status"] == "done" for v in led.values())
    assert any(isinstance(e, VideoDone) and e.run_id == "dyad07_collab"
               for e in events)


def test_run_folder_config_change_archives_mirrored_outputs(tmp_path,
                                                            fake_models_rows):
    """A config change on a run-folder project archives the MIRRORED per-run
    outputs (Outputs/Runs/<run_id>/*) into _run/superseded/, then reprocesses."""
    proj = _run_folder_project(tmp_path)
    _run(proj, parse_cli([]))
    run_dir = proj / "Outputs" / "Runs" / "dyad07_collab"
    assert (run_dir / "dyad07_collab_summary.csv").exists()

    events = _run(proj, parse_cli(["--conf", "0.30"]))
    assert any(isinstance(e, VideoArchived) and e.run_id == "dyad07_collab"
               for e in events)
    superseded = proj / "Outputs" / "_run" / "superseded"
    archived = sorted(superseded.iterdir())
    # Both runs archived; each archive dir holds the mirrored per-run files.
    assert any(d.name.endswith("_dyad07_collab") for d in archived)
    collab_arch = next(d for d in archived if d.name.endswith("_dyad07_collab"))
    assert (collab_arch / "dyad07_collab_summary.csv").exists()
    assert (collab_arch / "dyad07_collab_manifest.json").exists()
    # Fresh outputs were written back to the mirrored location.
    assert (run_dir / "dyad07_collab_summary.csv").exists()


def test_run_folder_resume_skips_unchanged(tmp_path, fake_models_rows):
    proj = _run_folder_project(tmp_path)
    _run(proj, parse_cli([]))
    assert len(fake_models_rows) == 2
    events = _run(proj, parse_cli([]))          # resume -> both skip
    assert len(fake_models_rows) == 2           # nothing reprocessed
    skipped = {e.run_id for e in events if isinstance(e, VideoSkipped)}
    assert skipped == {"dyad07_collab", "dyad07_solo"}


def test_run_folder_manifest_carries_run_meta(tmp_path, fake_models_rows):
    proj = tmp_path / "mproj"
    folder = proj / "Inputs" / "Runs" / "run01"
    _make_clip(folder / "clip.mp4")
    (folder / "run.yaml").write_text(
        "conditions: [c]\ndate: 2026-07-02\nnotes: hi\n")
    _run(proj, parse_cli([]))
    manifest = json.loads(
        (proj / "Outputs" / "Runs" / "run01" / "run01_manifest.json").read_text())
    assert manifest["run_meta"] == {"date": "2026-07-02", "notes": "hi"}


# ── G-DEFER-3: anonymize request reaches the per-video OutputConfig ───────────

def _capture_output_cfgs(monkeypatch):
    seen: list = []

    def _build(_ns):
        return tuple([None] * 14)

    class _Writer:
        def __init__(self, *, output_cfg, **_kw):
            seen.append(output_cfg)

        def run(self, source, *, options=None, cancel=None):
            summ = Path(seen[-1].summary_path)
            summ.parent.mkdir(parents=True, exist_ok=True)
            summ.write_text("video_name,conditions,phenomenon,participant,"
                            "partner,object,metric,value\n")
            Path(seen[-1].log_path).write_text("frame,t_seconds\n")
            return iter(())

    monkeypatch.setattr(_pipeline_mod, "build_from_namespace", _build)
    monkeypatch.setattr(_pipeline_mod, "Pipeline", _Writer)
    return seen


def test_anonymize_ns_reaches_output_cfg(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    seen = _capture_output_cfgs(monkeypatch)
    ns = parse_cli(["--project", str(proj)])
    ns.anonymize = "blur"
    list(iter_project_runs(proj, ns, resume=False))
    assert seen and seen[0].anonymize == "blur"
    assert seen[0].anonymize_padding == 0.3


def test_no_anonymize_defaults_to_none(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    seen = _capture_output_cfgs(monkeypatch)
    ns = parse_cli(["--project", str(proj)])
    list(iter_project_runs(proj, ns, resume=False))
    assert seen and seen[0].anonymize is None
