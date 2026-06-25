"""Fast offscreen coverage for the Analyze Footage (Run Study) tab (SP3.1 Step 15).

No models, no video decoding -- these pin the CONSUMPTION contract (D11): the tab
renders a PreflightReport as a checklist, lists staged runs with ledger status,
previews the resume plan (``Project.decisions``), and the manual dialog builds a
valid RunSpec via the staging helpers.  The model-driven run/cancel-through-the-
worker behavior lives in tests/test_gui_smoke.py (slow).
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _make_project(tmp_path, *, videos=("a.mp4", "b.mp4"), pipeline=True):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    for v in videos:
        (proj / "Inputs" / "Videos" / v).write_bytes(b"\x00" * 32)
    if pipeline:
        (proj / "Pipeline").mkdir()
        (proj / "Pipeline" / "pipeline.yaml").write_text("detection:\n  conf: 0.35\n")
    return proj


# ── Project.decisions preview (project layer, D11) ───────────────────────────

def test_decisions_resume_off_is_all_redo(tmp_path):
    from mindsight.project.project import Project
    from argparse import Namespace
    proj = _make_project(tmp_path)
    project = Project.open(proj)
    plan = project.decisions(Namespace(), resume=False)
    assert plan == {"a.mp4": "redo", "b.mp4": "redo"}


def test_decisions_skip_when_ledger_matches(tmp_path, monkeypatch):
    from argparse import Namespace

    from mindsight.outputs import provenance
    from mindsight.project.ledger import Ledger, compute_video_hash
    from mindsight.project.project import Project

    proj = _make_project(tmp_path)
    # Neutralize the heavy identity computation.
    monkeypatch.setattr(provenance, "collect_weights", lambda ns: {})
    monkeypatch.setattr(provenance, "run_identity",
                        lambda ns, *, config, weights: "CFGHASH")

    project = Project.open(proj)
    specs = {s.run_id: s for s in project.runs()}
    a = specs["a.mp4"]
    vh = compute_video_hash(a.source, pid_map=a.pid_map,
                            conditions=a.conditions, aux_streams=a.aux_streams)
    # Write a done ledger record for a.mp4 with matching hashes.
    led = Ledger.load(proj / "Outputs")
    led.mark_started("a.mp4", ("CFGHASH", vh), a.output_paths)
    led.mark_done("a.mp4")
    plan = project.decisions(Namespace(), resume=True)
    assert plan["a.mp4"] == "skip"
    assert plan["b.mp4"] == "redo"     # no ledger record


# ── Tab: preflight checklist + runs table ────────────────────────────────────

def test_checklist_renders_all_checks(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._open_project(str(proj))
    report = tab._project.preflight()
    n = len(report.checks)
    assert n >= 8
    # One QLabel per check + fix-hint rows + a summary line.
    rows = tab._checklist._lay.count()
    assert rows >= n + 1


def test_runs_table_matches_ledger_fixture(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    # Seed a ledger with one done + one error run.
    run_dir = proj / "Outputs" / "_run"
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.json").write_text(json.dumps({
        "ledger_version": 1,
        "videos": {
            "a.mp4": {"status": "done", "config_hash": "c", "video_hash": "v",
                      "finished": "2026-07-07T00:00:00+00:00", "error": None},
            "b.mp4": {"status": "error", "config_hash": "c", "video_hash": "v",
                      "finished": "2026-07-07T00:00:00+00:00",
                      "error": "boom"},
        },
    }))
    tab = RunStudyTab()
    tab._open_project(str(proj))
    assert tab._runs_table.rowCount() == 2
    by_run = {}
    for r in range(tab._runs_table.rowCount()):
        by_run[tab._runs_table.item(r, 0).text()] = r
    assert tab._runs_table.item(by_run["a.mp4"], 4).text() == "done"
    assert tab._runs_table.item(by_run["b.mp4"], 4).text() == "error"
    assert tab._runs_table.item(by_run["b.mp4"], 7).text() == "boom"


def test_rerun_all_previews_redo(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._open_project(str(proj))
    tab._resume = False
    tab._refresh_runs_table()
    for r in range(tab._runs_table.rowCount()):
        assert tab._runs_table.item(r, 5).text() == "will process"


# ── Manual dialog builds a valid RunSpec ─────────────────────────────────────

def test_manual_dialog_builds_valid_runspec(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import ManualRunDialog
    from mindsight.project.staging import single_run_spec

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    dlg = ManualRunDialog(None)
    dlg._video.setText(str(video))
    dlg._participants.setText("0:S70, 1:S71")
    dlg._conditions.setText("collab, kitchenA")
    dlg._date.setText("2026-07-02")
    meta = dlg._collect_meta()
    assert meta["participants"] == {0: "S70", 1: "S71"}
    assert meta["conditions"] == ["collab", "kitchenA"]
    assert meta["date"] == "2026-07-02"

    spec = single_run_spec(str(video), meta, str(tmp_path / "out"))
    assert spec.run_id == "clip"
    assert spec.pid_map == {0: "S70", 1: "S71"}
    assert spec.conditions == "collab|kitchenA"
    assert spec.output_paths["summary"].endswith("clip_summary.csv")
