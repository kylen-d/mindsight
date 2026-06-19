"""Fast tests for the Project facade + Ledger.invalidate (SP3.1 D3/D10).

No models built: Project.open / .runs / .status / .preflight are exercised over
tmp projects with hand-written ledger records; Ledger.invalidate is checked
against decide().
"""
from __future__ import annotations

import pytest

from mindsight.project.ledger import Ledger
from mindsight.project.project import Project, VideoStatus


def _project(tmp_path, videos=("a.mp4", "b.mp4")):
    proj = tmp_path / "proj"
    vids = proj / "Inputs" / "Videos"
    vids.mkdir(parents=True)
    for name in videos:
        (vids / name).write_bytes(b"\x00\x00")   # discover_sources needs files
    return proj


# ── Project.open / accessors ────────────────────────────────────────────────

def test_open_validates_and_resolves(tmp_path):
    proj = _project(tmp_path)
    p = Project.open(proj)
    assert p.path == proj.resolve()
    # No project.yaml -> config is None; output dirs were created by validate.
    assert p.config is None
    assert (proj / "Outputs" / "CSV Files").is_dir()


def test_open_missing_inputs_raises(tmp_path):
    bare = tmp_path / "empty"
    bare.mkdir()
    with pytest.raises(ValueError):
        Project.open(bare)


def test_runs_lists_discovered_sources(tmp_path):
    proj = _project(tmp_path, videos=("b.mp4", "a.mp4"))
    p = Project.open(proj)
    names = [s.name for s in p.runs()]
    assert names == ["a.mp4", "b.mp4"]          # sorted


def test_run_returns_lazy_iterator(tmp_path):
    proj = _project(tmp_path)
    p = Project.open(proj)
    from argparse import Namespace
    gen = p.run(Namespace(), resume=False)
    # Generator: nothing executes until iteration -> no model build here.
    assert hasattr(gen, "__next__")
    gen.close()


def test_preflight_is_stub(tmp_path):
    p = Project.open(_project(tmp_path))
    with pytest.raises(NotImplementedError):
        p.preflight()


# ── Project.status matrix ───────────────────────────────────────────────────

def _seed_ledger(proj, records):
    out_root = proj / "Outputs"
    ledger = Ledger.load(out_root)
    for run_id, (cfg_h, vid_h, paths) in records.items():
        ledger.mark_started(run_id, (cfg_h, vid_h), paths)
    return ledger


def test_status_reflects_ledger_matrix(tmp_path):
    proj = _project(tmp_path)
    ledger = _seed_ledger(proj, {
        "a.mp4": ("cfg1", "vid_a", {"summary": "x"}),
        "b.mp4": ("cfg1", "vid_b", {"summary": "y"}),
    })
    ledger.mark_done("a.mp4", "a_manifest.json")
    ledger.mark_error("b.mp4", "boom")

    statuses = Project.open(proj).status()
    assert [s.run_id for s in statuses] == ["a.mp4", "b.mp4"]   # sorted
    by_id = {s.run_id: s for s in statuses}
    assert isinstance(statuses[0], VideoStatus)
    assert by_id["a.mp4"].status == "done"
    assert by_id["a.mp4"].error is None
    assert by_id["a.mp4"].config_hash == "cfg1"
    assert by_id["b.mp4"].status == "error"
    assert by_id["b.mp4"].error == "boom"


def test_status_empty_when_no_ledger(tmp_path):
    assert Project.open(_project(tmp_path)).status() == []


# ── Ledger.invalidate ───────────────────────────────────────────────────────

def test_invalidate_drops_record_so_decide_redoes(tmp_path):
    proj = _project(tmp_path)
    out_root = proj / "Outputs"
    ledger = Ledger.load(out_root)
    hashes = ("cfg1", "vid_a")
    ledger.mark_started("a.mp4", hashes, {"summary": "x"})
    ledger.mark_done("a.mp4", "a_manifest.json")
    assert ledger.decide("a.mp4", hashes) == "skip"

    assert ledger.invalidate("a.mp4") is True
    assert ledger.decide("a.mp4", hashes) == "redo"      # absent -> redo
    assert ledger.record("a.mp4") is None

    # Idempotent: invalidating an absent record is a no-op.
    assert ledger.invalidate("a.mp4") is False

    # Persisted: a fresh load sees the drop.
    assert Ledger.load(out_root).decide("a.mp4", hashes) == "redo"


def test_invalidate_then_rerun_does_not_archive(tmp_path):
    """A dropped record yields redo (not redo_archive) -- reprocess in place."""
    proj = _project(tmp_path)
    ledger = Ledger.load(proj / "Outputs")
    hashes = ("cfg1", "vid_a")
    ledger.mark_started("a.mp4", hashes, {"summary": "x"})
    ledger.mark_done("a.mp4", "m.json")
    ledger.invalidate("a.mp4")
    # Even with a *changed* config the decision is redo (no stored done-state).
    assert ledger.decide("a.mp4", ("cfg2", "vid_a")) == "redo"
