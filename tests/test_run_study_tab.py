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


# ── Editable project path field (G-FIX-3) ────────────────────────────────────

def test_typed_path_opens_project(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._project_dir.setText(str(proj))
    tab._open_typed_path()
    assert tab._project is not None
    assert tab._project_path == proj.resolve()
    assert tab._status_label.text().startswith("Open:")


def test_typed_invalid_path_shows_inline_error(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    tab._project_dir.setText(str(tmp_path / "definitely" / "not" / "there"))
    tab._open_typed_path()          # must not raise
    assert tab._project is None
    assert "Invalid project" in tab._status_label.text()
    # a path pointing at a FILE (not a directory) is also a readable error
    f = tmp_path / "somefile.txt"
    f.write_text("x")
    tab._project_dir.setText(str(f))
    tab._open_typed_path()
    assert tab._project is None
    assert "Invalid project" in tab._status_label.text()


def test_typed_empty_path_is_noop(qapp):
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    tab._project_dir.setText("   ")
    tab._open_typed_path()
    assert tab._project is None
    assert tab._status_label.text() == "No project open."


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


# ── One-click fetch of missing weights (Step 11) ─────────────────────────────

def test_preflight_offers_and_fetches_missing_weights(qapp, tmp_path, monkeypatch):
    """A failing preflight with fetchable missing weights offers a download
    button; clicking it downloads (monkeypatched -- no network) and re-runs
    preflight, which clears the offer."""
    from argparse import Namespace

    from mindsight import weights
    from mindsight.GUI.run_study_tab import RunStudyTab

    proj = _make_project(tmp_path)
    fetched = tmp_path / "fetched.onnx"
    entry = {
        "backend": "MGaze", "filename": "resnet50_gaze.onnx",
        "label": "MobileGaze (ResNet-50, ONNX)",
        "url": "https://example.invalid/w", "sha256": "0" * 64, "size": 1,
        "license": "MIT", "required": True,
        "source": weights.SOURCE_GITHUB, "note": None,
    }

    def fake_missing(names, *, path=None):
        # The manifest module is the single authority (consume-don't-compute):
        # offer the entry until it has been fetched.
        return [] if fetched.exists() else [entry]

    calls = []

    def fake_download(e, *, dest=None, progress=print, retries=2):
        calls.append(e["filename"])
        fetched.write_bytes(b"ok")
        return fetched

    monkeypatch.setattr(weights, "downloadable_missing", fake_missing)
    monkeypatch.setattr(weights, "download", fake_download)

    class FakeGaze:
        def _build_namespace(self):
            # A configured model that is absent on disk -> weights check FAILS.
            return Namespace(model=str(tmp_path / "no_such_weight.pt"))

    tab = RunStudyTab()
    tab._open_project(str(proj))
    tab._gaze_tab = FakeGaze()          # attach after open (no apply_namespace)

    tab._run_preflight()
    assert tab._fetchable == [entry]    # the offer surfaced

    tab._start_weight_fetch()
    for t in tab._weight_threads:
        t.join(timeout=5)
    tab._drain_weight_fetch()           # applies results, re-runs preflight

    assert calls == ["resnet50_gaze.onnx"]
    assert tab._fetchable == []         # offer cleared once satisfied


# ── Edit run metadata before running (G-DEFER-1) ─────────────────────────────

def test_edit_run_dialog_prefills_and_collects(qapp):
    from mindsight.GUI.run_study_tab import EditRunDialog
    dlg = EditRunDialog("a.mp4", {0: "S70", 1: "S71"}, "collab|kitchenA")
    assert dlg._participants.text() == "0:S70, 1:S71"
    assert dlg._conditions.text() == "collab, kitchenA"
    dlg._participants.setText("2:S99")
    dlg._conditions.setText("solo")
    dlg._finish()
    assert dlg.participants == {2: "S99"}
    assert dlg.conditions == ["solo"]


def test_edit_run_flow_writes_project_yaml(qapp, tmp_path, monkeypatch):
    from mindsight.GUI import run_study_tab as rst
    from mindsight.GUI.run_study_tab import RunStudyTab
    from mindsight.project.runner import load_project_config

    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._open_project(str(proj))

    class FakeDlg:
        participants = {0: "S70"}
        conditions = ["collab"]
        def exec(self):
            from PyQt6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted
    monkeypatch.setattr(rst, "EditRunDialog",
                        lambda *a, **k: FakeDlg())
    tab._edit_run("a.mp4")

    cfg = load_project_config(proj)
    assert cfg.participants["a.mp4"] == {0: "S70"}
    assert cfg.conditions["a.mp4"] == ["collab"]
    # runs table refreshed with the new values
    row = {tab._runs_table.item(r, 0).text(): r
           for r in range(tab._runs_table.rowCount())}["a.mp4"]
    assert "S70" in tab._runs_table.item(row, 2).text()
    assert "collab" in tab._runs_table.item(row, 3).text()


# ── Anonymize Footage toggle (G-DEFER-3) ─────────────────────────────────────

def test_anonymize_toggle_sets_ns(qapp):
    from argparse import Namespace
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    # default OFF -> ns.anonymize forced None (byte-neutral)
    ns = Namespace(anonymize="black")
    tab._apply_anonymize(ns)
    assert ns.anonymize is None
    # checked -> the selected mode reaches the ns
    tab._anonymize_cb.setChecked(True)
    tab._anonymize_mode.setCurrentText("black")
    ns2 = Namespace()
    tab._apply_anonymize(ns2)
    assert ns2.anonymize == "black"


def test_anonymize_toggle_reaches_project_worker(qapp, tmp_path, monkeypatch):
    from mindsight.GUI import workers as workers_mod
    from mindsight.GUI.run_study_tab import RunStudyTab

    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._open_project(str(proj))
    tab._anonymize_cb.setChecked(True)
    tab._anonymize_mode.setCurrentText("blur")

    captured = {}

    class FakeWorker:
        def __init__(self, path, ns, *a, **k):
            captured["ns"] = ns
        def start(self):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr(workers_mod, "ProjectWorker", FakeWorker)
    tab._start()
    assert captured["ns"].anonymize == "blur"


# ── Run-folder project opens without a flat Inputs/Videos/ dir (Batch H fix) ─

def test_open_run_folder_project_without_flat_videos_dir(qapp, tmp_path):
    """A pure run-folder project has NO Inputs/Videos/; opening it must not
    crash the study-setup population (discover_sources returns [])."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = tmp_path / "rfproj"
    run = proj / "Inputs" / "Runs" / "dyad07"
    run.mkdir(parents=True)
    (run / "session.mp4").write_bytes(b"\x00" * 32)
    (proj / "Pipeline").mkdir()
    (proj / "Pipeline" / "pipeline.yaml").write_text("detection:\n  conf: 0.35\n")
    tab = RunStudyTab()
    tab._open_project(str(proj))
    assert tab._project is not None
    assert tab._runs_table.rowCount() == 1
    assert tab._runs_table.item(0, 0).text() == "dyad07"
