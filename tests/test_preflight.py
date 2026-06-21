"""SP3.1 Batch D Step 7: PreflightReport checks + registry load-error capture.

Fast (no models, no video decode): synthetic tmp projects drive every check's
ok/warn/fail branch, device/disk are injected, and plugin load errors are
exercised through a THROWAWAY PluginRegistry -- the real Plugins/ tree is never
touched (T5).  Also pins the Q5 Option A data_collection wiring (registry state
+ from_args build) and the registry ``load_errors`` seam (D5).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from Plugins import DataCollectionPlugin, PluginRegistry, data_collection_registry
from mindsight.project.preflight import (
    CheckResult,
    PreflightReport,
    format_report,
    run_preflight,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _by_id(report, check_id) -> CheckResult:
    for c in report.checks:
        if c.id == check_id:
            return c
    raise AssertionError(f"no check {check_id!r} in {[c.id for c in report.checks]}")


def _mk_project(tmp_path, *, videos=("a.mp4",), pipeline_text=None):
    proj = tmp_path / "proj"
    vids = proj / "Inputs" / "Videos"
    vids.mkdir(parents=True)
    for v in videos:
        (vids / v).write_bytes(b"\x00" * 32)     # not decoded -- size only
    if pipeline_text is not None:
        pdir = proj / "Pipeline"
        pdir.mkdir(parents=True)
        (pdir / "pipeline.yaml").write_text(pipeline_text)
    return proj


def _ns(**kw):
    """A bare namespace (no _explicit_cli -> legacy YAML precedence)."""
    return argparse.Namespace(**kw)


# ── report data model ────────────────────────────────────────────────────────

def test_report_ok_and_counts():
    checks = [
        CheckResult("a", "A", "ok", "fine"),
        CheckResult("b", "B", "warn", "meh", "do x"),
        CheckResult("c", "C", "fail", "bad", "fix y"),
    ]
    r = PreflightReport(checks)
    assert r.ok is False and r.n_fail == 1 and r.n_warn == 1
    r2 = PreflightReport([c for c in checks if c.severity != "fail"])
    assert r2.ok is True and r2.n_fail == 0


# ── structure / runs ─────────────────────────────────────────────────────────

def test_structure_missing_videos_dir_fails(tmp_path):
    proj = tmp_path / "empty"
    proj.mkdir()
    report = run_preflight(proj, ns=_ns())
    assert _by_id(report, "project_structure").severity == "fail"
    assert _by_id(report, "runs_discovered").severity == "fail"
    assert report.ok is False


def test_no_videos_fails_runs(tmp_path):
    proj = _mk_project(tmp_path, videos=())
    report = run_preflight(proj, ns=_ns())
    assert _by_id(report, "project_structure").severity == "ok"
    assert _by_id(report, "runs_discovered").severity == "fail"


def test_healthy_structure_ok(tmp_path):
    proj = _mk_project(tmp_path, videos=("a.mp4", "b.mp4"))
    report = run_preflight(proj, ns=_ns())
    assert _by_id(report, "project_structure").severity == "ok"
    runs = _by_id(report, "runs_discovered")
    assert runs.severity == "ok" and "2 source" in runs.message


# ── pipeline config (extra=forbid strict load) ───────────────────────────────

def test_pipeline_config_valid_ok(tmp_path):
    proj = _mk_project(tmp_path, pipeline_text="detection:\n  conf: 0.4\n")
    assert _by_id(run_preflight(proj, ns=_ns()), "pipeline_config").severity == "ok"


def test_pipeline_config_invalid_value_fails(tmp_path):
    # A known key with a value the schema rejects -> load_yaml raises (models are
    # extra=forbid + typed) -> fail with the verbatim error surfaced.
    proj = _mk_project(tmp_path, pipeline_text="gaze:\n  ray_length: not_a_float\n")
    c = _by_id(run_preflight(proj, ns=_ns()), "pipeline_config")
    assert c.severity == "fail" and "invalid" in c.message


def test_pipeline_config_missing_warns(tmp_path):
    proj = _mk_project(tmp_path)     # no Pipeline/pipeline.yaml
    assert _by_id(run_preflight(proj, ns=_ns()), "pipeline_config").severity == "warn"


# ── weights ──────────────────────────────────────────────────────────────────

def test_weights_present_records_sha(tmp_path):
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"weightbytes")
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(model=str(weight))), "weights")
    assert c.severity == "ok" and "model.pt" in c.message


def test_weights_missing_fails(tmp_path):
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(model="/no/such/weight.pt")), "weights")
    assert c.severity == "fail" and "MISSING" in c.message


def test_weights_verify_is_sp4_stub(tmp_path):
    proj = _mk_project(tmp_path)
    assert _by_id(run_preflight(proj, ns=_ns()), "weights_verify").severity == "ok"


# ── VP file ──────────────────────────────────────────────────────────────────

def test_vp_absent_non_vp_mode_ok(tmp_path):
    proj = _mk_project(tmp_path)
    assert _by_id(run_preflight(proj, ns=_ns()), "vp_file").severity == "ok"


def test_vp_absent_vp_mode_warns(tmp_path):
    proj = _mk_project(tmp_path)
    (proj / "Inputs" / "Prompts").mkdir(parents=True)
    c = _by_id(run_preflight(proj, ns=_ns(vp_file="x.vp.json")), "vp_file")
    assert c.severity == "warn"


def test_vp_valid_ok(tmp_path):
    proj = _mk_project(tmp_path)
    prompts = proj / "Inputs" / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "p.vp.json").write_text(
        '{"version":1,"classes":[{"id":0,"name":"cup"}],'
        '"references":[{"image":"i.png","annotations":[{"cls_id":0,"bbox":[1,2,3,4]}]}]}')
    assert _by_id(run_preflight(proj, ns=_ns()), "vp_file").severity == "ok"


def test_vp_no_annotations_fails(tmp_path):
    proj = _mk_project(tmp_path)
    prompts = proj / "Inputs" / "Prompts"
    prompts.mkdir(parents=True)
    (prompts / "p.vp.json").write_text('{"version":1,"references":[]}')
    assert _by_id(run_preflight(proj, ns=_ns()), "vp_file").severity == "fail"


# ── participants / conditions coverage ───────────────────────────────────────

def test_participants_conditions_warn_when_absent(tmp_path):
    proj = _mk_project(tmp_path)
    report = run_preflight(proj, ns=_ns())
    assert _by_id(report, "participants_coverage").severity == "warn"
    assert _by_id(report, "conditions_coverage").severity == "warn"


def test_participants_conditions_ok_from_project_yaml(tmp_path):
    proj = _mk_project(tmp_path)
    (proj / "project.yaml").write_text(
        "version: 1\nconditions:\n  a.mp4: [G1]\nparticipants:\n  a.mp4:\n    0: S1\n")
    report = run_preflight(proj, ns=_ns())
    assert _by_id(report, "participants_coverage").severity == "ok"
    assert _by_id(report, "conditions_coverage").severity == "ok"


# ── device (injected) ────────────────────────────────────────────────────────

def test_device_auto_reports_resolution(tmp_path):
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(device="auto"),
                             device_check=lambda r: (True, "cpu")), "device")
    assert c.severity == "ok" and "cpu" in c.message


def test_device_explicit_unavailable_fails(tmp_path):
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(device="cuda"),
                             device_check=lambda r: (False, r)), "device")
    assert c.severity == "fail"


# ── disk (injected) ──────────────────────────────────────────────────────────

def test_disk_low_warns_when_saving(tmp_path):
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(save=True),
                             disk_usage=lambda p: SimpleNamespace(free=1)), "disk_space")
    assert c.severity == "warn"


def test_disk_ok_when_not_saving(tmp_path):
    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(save=None),
                             disk_usage=lambda p: SimpleNamespace(free=1)), "disk_space")
    assert c.severity == "ok"


# ── plugins: load errors (D5) via a throwaway registry ───────────────────────

def test_plugin_load_error_captured_and_failed(tmp_path):
    plug_root = tmp_path / "Plugins"
    bad = plug_root / "Bad"
    bad.mkdir(parents=True)
    (bad / "__init__.py").write_text("")
    (bad / "bad.py").write_text('raise RuntimeError("boom at import")')

    reg = PluginRegistry()
    reg.discover(plug_root, namespace="ThrowawayPlugins")
    assert reg.load_errors and "boom at import" in reg.load_errors[0][1]

    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns(), registries=[reg]), "plugins")
    assert c.severity == "fail" and "boom at import" in c.message


def test_plugins_clean_ok(tmp_path):
    proj = _mk_project(tmp_path)
    reg = PluginRegistry()      # no discovery -> no load errors
    c = _by_id(run_preflight(proj, ns=_ns(), registries=[reg]), "plugins")
    assert c.severity == "ok"


# ── Q5 Option A: data_collection wiring ──────────────────────────────────────

def test_data_collection_registry_inert_but_wired():
    # No in-repo DataCollection plugins -> empty registry, but it now carries the
    # D5 load_errors seam and factory.build_data_plugins reads it via from_args.
    assert data_collection_registry.names() == []
    assert hasattr(data_collection_registry, "load_errors")
    from mindsight.factory import build_data_plugins
    assert build_data_plugins(_ns()) == []


def test_build_data_plugins_from_args_and_listed_in_preflight(tmp_path, monkeypatch):
    class _FakeDC(DataCollectionPlugin):
        name = "fake_dc"

        @classmethod
        def from_args(cls, args):
            return cls()

    reg = PluginRegistry()
    reg.register(_FakeDC)
    monkeypatch.setattr("mindsight.factory._dc_registry", reg)

    from mindsight.factory import build_data_plugins
    active = build_data_plugins(_ns())
    assert len(active) == 1 and active[0].name == "fake_dc"

    proj = _mk_project(tmp_path)
    c = _by_id(run_preflight(proj, ns=_ns()), "plugins")
    assert c.severity == "ok" and "fake_dc" in c.message


# ── SP3.1 Batch E Step 9: run-folder layout preflight (Q1/Q2) ────────────────

def _hermetic(proj, **kw):
    """run_preflight with device/disk/registries injected (no torch, fast)."""
    return run_preflight(
        proj, ns=_ns(),
        device_check=lambda r: (True, "cpu"),
        disk_usage=lambda p: SimpleNamespace(free=10 ** 12),
        registries=[], **kw)


def _run_folder(proj, run_id, *, video="video.mp4", run_yaml=None):
    folder = proj / "Inputs" / "Runs" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / video).write_bytes(b"\x00" * 32)
    if run_yaml is not None:
        (folder / "run.yaml").write_text(run_yaml)
    return folder


def test_run_folder_structure_and_runs_ok(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01", run_yaml="conditions: [c]\nparticipants: {0: S1}\n")
    report = _hermetic(proj)
    assert _by_id(report, "project_structure").severity == "ok"
    assert "Inputs/Runs/" in _by_id(report, "project_structure").message
    runs = _by_id(report, "runs_discovered")
    assert runs.severity == "ok" and "run folder(s)" in runs.message
    assert _by_id(report, "run_metadata").severity == "ok"


def test_ambiguous_layout_fails(tmp_path):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    (proj / "Inputs" / "Videos" / "a.mp4").write_bytes(b"\x00" * 32)
    _run_folder(proj, "run01")
    report = _hermetic(proj)
    assert _by_id(report, "project_structure").severity == "fail"
    assert _by_id(report, "runs_discovered").severity == "fail"


def test_run_folder_two_videos_fails(tmp_path):
    proj = tmp_path / "proj"
    folder = _run_folder(proj, "run01", video="a.mp4")
    (folder / "b.mp4").write_bytes(b"\x00" * 32)
    runs = _by_id(_hermetic(proj), "runs_discovered")
    assert runs.severity == "fail" and "run01" in runs.message


def test_run_folder_unknown_key_warns(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01", run_yaml="conditions: [c]\ntpyo: x\n")
    c = _by_id(_hermetic(proj), "run_metadata")
    assert c.severity == "warn" and "tpyo" in c.message


def test_run_folder_bad_metadata_fails(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "run01", run_yaml="participants: [nope]\n")
    c = _by_id(_hermetic(proj), "run_metadata")
    assert c.severity == "fail" and "run01" in c.message


def test_run_folder_participant_coverage(tmp_path):
    proj = tmp_path / "proj"
    _run_folder(proj, "with_pid", run_yaml="participants: {0: S1}\nconditions: [c]\n")
    _run_folder(proj, "no_pid", run_yaml="conditions: [c]\n")
    report = _hermetic(proj)
    pc = _by_id(report, "participants_coverage")
    assert pc.severity == "warn" and "no_pid" in pc.message and "with_pid" not in pc.message
    assert _by_id(report, "conditions_coverage").severity == "ok"


def test_legacy_preflight_has_no_run_metadata_check(tmp_path):
    # The run_metadata check is run-folder only; legacy checklist is unchanged.
    proj = _mk_project(tmp_path)
    ids = [c.id for c in _hermetic(proj).checks]
    assert "run_metadata" not in ids


# ── pretty-print + Project facade ────────────────────────────────────────────

def test_format_report_renders_tags(tmp_path):
    proj = _mk_project(tmp_path, videos=())
    text = format_report(run_preflight(proj, ns=_ns()), title="proj")
    assert "[FAIL]" in text and "Preflight FAILED" in text and "fix:" in text


def test_project_facade_preflight(tmp_path):
    from mindsight.project.project import Project
    proj = _mk_project(tmp_path, pipeline_text="detection:\n  conf: 0.4\n")
    report = Project.open(proj).preflight(ns=_ns())
    assert isinstance(report, PreflightReport)
    assert _by_id(report, "project_structure").severity == "ok"
