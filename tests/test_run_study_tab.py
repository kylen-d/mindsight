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
    # UP1r2: quick analysis is a mode now; the resting status is plain.
    assert tab._status_label.text() == "No project open."


# ── New Project button (scaffold + open) ─────────────────────────────────────

def test_new_project_scaffolds_and_opens(qapp, tmp_path, monkeypatch):
    import mindsight.GUI.run_study_tab as rst
    from mindsight.GUI.run_study_tab import RunStudyTab

    # Bypass the native dialogs: pick tmp_path as parent, "MyStudy" as name.
    monkeypatch.setattr(rst.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(rst.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("MyStudy", True)))

    tab = RunStudyTab()
    tab._new_project_dialog()

    new_dir = tmp_path / "MyStudy"
    # Standard blank-project structure landed on disk.
    assert (new_dir / "project.yaml").is_file()
    assert (new_dir / "Inputs" / "Videos").is_dir()
    assert (new_dir / "Inputs" / "Prompts").is_dir()
    assert (new_dir / "Pipeline").is_dir()
    # And it opened in the tab (preflight ran -> report present).
    assert tab._project is not None
    assert tab._project_path == new_dir.resolve()
    assert tab._status_label.text().startswith("Open:")
    assert tab._project.preflight().checks


def test_new_project_cancel_is_noop(qapp, tmp_path, monkeypatch):
    import mindsight.GUI.run_study_tab as rst
    from mindsight.GUI.run_study_tab import RunStudyTab

    monkeypatch.setattr(rst.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    # User cancels the name prompt (ok=False).
    monkeypatch.setattr(rst.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("", False)))

    tab = RunStudyTab()
    tab._new_project_dialog()
    assert tab._project is None
    assert not any(tmp_path.iterdir())     # nothing created


def test_new_project_duplicate_shows_inline_error(qapp, tmp_path, monkeypatch):
    import mindsight.GUI.run_study_tab as rst
    from mindsight.GUI.run_study_tab import RunStudyTab

    existing = tmp_path / "Dup"
    (existing / "Inputs" / "Videos").mkdir(parents=True)
    (existing / "Inputs" / "Videos" / "x.mp4").write_bytes(b"\x00")

    monkeypatch.setattr(rst.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(rst.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Dup", True)))

    tab = RunStudyTab()
    tab._new_project_dialog()          # must not raise
    assert tab._project is None
    assert "Could not create project" in tab._status_label.text()


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

    tab = RunStudyTab()
    tab._open_project(str(proj))
    # Decoupling (UP2): the run config comes from the RunSettings store, not the
    # Gaze Tuning tab.  Commit a configured-but-absent model so preflight FAILS.
    cfg_ns = tab._settings.working_copy()
    cfg_ns.model = str(tmp_path / "no_such_weight.pt")
    tab._settings.commit(cfg_ns)

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


# ── Mode switch + quick modes (UP1r2) ────────────────────────────────────────

def test_starts_in_project_mode_with_all_modes_offered(qapp):
    """Fresh tab: project mode active, all three segmented buttons present."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    assert tab._mode == "project"
    assert set(tab._mode_btns) == {"project", "video", "camera"}
    assert tab._mode_btns["project"].isChecked()
    assert tab._source_stack.currentIndex() == 0
    assert tab._left_stack.currentIndex() == 0
    # Live dashboard lives in the output tabs (project mode).
    assert tab._output_tabs.indexOf(tab._dashboard_panel) >= 0
    assert tab._output_tabs.isTabVisible(1)      # post-run Charts tab


def test_mode_switch_morphs_the_whole_tab(qapp):
    """Video/camera modes: quick source card, live charts on the left, the
    dashboard reparented out of the tabs, Charts tab hidden, status-bar
    Run/Stop hidden.  Switching back restores project mode exactly."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    tab._set_mode("video")
    assert tab._source_stack.currentIndex() == 1
    assert tab._left_stack.currentIndex() == 1       # live charts pane
    assert tab._output_tabs.indexOf(tab._dashboard_panel) < 0
    assert tab._charts_pane_lay.indexOf(tab._dashboard_panel) >= 0
    assert not tab._output_tabs.isTabVisible(1)      # Charts hidden
    tab._set_mode("camera")
    assert tab._source_stack.currentIndex() == 2
    assert tab._left_stack.currentIndex() == 1
    tab._set_mode("project")
    assert tab._source_stack.currentIndex() == 0
    assert tab._left_stack.currentIndex() == 0
    assert tab._output_tabs.indexOf(tab._dashboard_panel) >= 0
    assert tab._output_tabs.tabText(
        tab._output_tabs.indexOf(tab._dashboard_panel)) == "Live"
    assert tab._output_tabs.isTabVisible(1)


def test_opening_project_commits_to_project_mode(qapp, tmp_path):
    from mindsight.GUI.run_study_tab import RunStudyTab
    proj = _make_project(tmp_path)
    tab = RunStudyTab()
    tab._set_mode("video")
    tab._open_project(str(proj))
    assert tab._mode == "project"
    assert not tab._pf_grp.isHidden()
    assert not tab._runs_grp.isHidden()
    assert not tab._study_setup_grp.isHidden()


def test_last_mode_is_remembered_across_instances(qapp):
    """The chosen mode persists via gui_state.json (isolated settings dir)."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    tab._set_mode("camera")
    tab2 = RunStudyTab()
    assert tab2._mode == "camera"
    tab2._set_mode("project")     # leave a clean default behind


def test_quick_output_auto_defaults_and_sticks_when_edited(qapp, tmp_path,
                                                           monkeypatch):
    from mindsight import constants
    from mindsight.GUI.run_study_tab import RunStudyTab
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)
    tab = RunStudyTab()
    # Video mode: default output is <PROJECT_ROOT>/Outputs/<stem>.
    tab._set_mode("video")
    tab._quick_video.setText(str(tmp_path / "footage" / "sess1.mp4"))
    assert tab._video_output.text() == str(tmp_path / "Outputs" / "sess1")
    # Camera mode has its own field: <PROJECT_ROOT>/Outputs/camera<idx>.
    tab._set_mode("camera")
    tab._camera_combo.setCurrentIndex(2)
    assert tab._camera_output.text() == str(tmp_path / "Outputs" / "camera2")
    # A manual edit sticks: no more auto-recompute.
    tab._set_mode("video")
    tab._video_output.setText("/my/custom/out")
    tab._video_mark_output_dirty()
    tab._quick_video.setText(str(tmp_path / "other.mp4"))
    assert tab._video_output.text() == "/my/custom/out"


def test_run_quick_video_builds_spec_and_guards(qapp, tmp_path, monkeypatch):
    """_run_quick builds the expected spec, starts the worker, flips the inline
    button to Stop, and the worker-alive guard blocks a second start."""
    from mindsight import constants
    from mindsight.GUI import workers as workers_mod
    from mindsight.GUI.run_study_tab import RunStudyTab

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)

    started = []

    class FakeWorker:
        def __init__(self, ns, *a, **k):
            self.ns = ns
            started.append(ns)

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(workers_mod, "GazeWorker", FakeWorker)

    tab = RunStudyTab()
    tab._set_mode("video")
    tab._quick_video.setText(str(video))
    # Output auto-defaulted to <PROJECT_ROOT>/Outputs/clip.
    assert tab._video_output.text() == str(tmp_path / "Outputs" / "clip")
    tab._run_quick()
    assert len(started) == 1
    ns = started[0]
    assert ns.source == str(video)
    assert ns.log == str(tmp_path / "Outputs" / "clip" / "clip_Events.csv")
    # The output folder is created on run (UP1 ruling 1).
    assert (tmp_path / "Outputs" / "clip").is_dir()
    # Inline primary button became the stop control (UP1r2).
    assert tab._video_go.text() == "■  Stop"
    # Worker-alive guard blocks a second start.
    tab._run_quick()
    assert len(started) == 1
    tab._poll_timer.stop()


def test_run_quick_camera_builds_camera_spec(qapp, tmp_path, monkeypatch):
    from mindsight import constants
    from mindsight.GUI import workers as workers_mod
    from mindsight.GUI.run_study_tab import RunStudyTab

    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)
    started = []

    class FakeWorker:
        def __init__(self, ns, *a, **k):
            started.append(ns)

        def start(self):
            pass

        def is_alive(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(workers_mod, "GazeWorker", FakeWorker)

    tab = RunStudyTab()
    tab._set_mode("camera")
    tab._camera_combo.setCurrentIndex(0)
    assert tab._camera_output.text() == str(tmp_path / "Outputs" / "camera0")
    tab._run_quick()
    assert len(started) == 1
    # source is the camera index string; open_video_source normalizes to int.
    assert started[0].source == "0"
    tab._poll_timer.stop()


def test_run_quick_no_output_folder_warns(qapp, monkeypatch):
    from mindsight.GUI import run_study_tab as rst
    from mindsight.GUI.run_study_tab import RunStudyTab

    warned = []
    monkeypatch.setattr(rst.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    tab = RunStudyTab()
    tab._set_mode("video")
    tab._video_output.setText("")
    tab._video_output_dirty = True   # keep it empty (no auto-recompute)
    tab._run_quick()
    assert warned


def test_drop_routes_video_and_folder_to_their_modes(qapp, tmp_path):
    """A dropped video file lands in Video File mode; a dropped folder opens
    as a project (the handler only reads mimeData().urls())."""
    from PyQt6.QtCore import QUrl

    from mindsight.GUI.run_study_tab import RunStudyTab

    class FakeMime:
        def __init__(self, url):
            self._url = url

        def urls(self):
            return [self._url]

    class FakeEvent:
        def __init__(self, path):
            self._mime = FakeMime(QUrl.fromLocalFile(str(path)))

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            pass

    video = tmp_path / "drop.mp4"
    video.write_bytes(b"\x00" * 32)
    tab = RunStudyTab()
    tab.dropEvent(FakeEvent(video))
    assert tab._mode == "video"
    assert tab._quick_video.text() == str(video)

    proj = _make_project(tmp_path)
    tab.dropEvent(FakeEvent(proj))
    assert tab._mode == "project"
    assert tab._project is not None


def test_finished_quick_run_csvs_land_in_viewer(qapp, tmp_path):
    """_register_one_off_outputs makes the quick run's CSVs selectable."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    out = tmp_path / "Outputs" / "clip"
    out.mkdir(parents=True)
    (out / "clip_Events.csv").write_text("frame\n1\n")
    (out / "clip_summary.csv").write_text("metric,value\n")
    tab = RunStudyTab()
    tab._last_one_off = ("clip", str(out))
    tab._register_one_off_outputs()
    assert tab._csv_run.currentText() == "clip"
    names = [tab._csv_file.itemText(i) for i in range(tab._csv_file.count())]
    assert "clip_Events.csv" in names and "clip_summary.csv" in names


def test_go_buttons_flip_colours_with_run_state(qapp, tmp_path, monkeypatch):
    """UP1r3: every card's go button is green when idle and flips to a red
    Stop while any run is live; the project Run greys out until a project is
    open."""
    from mindsight import constants
    from mindsight.GUI import workers as workers_mod
    from mindsight.GUI.run_study_tab import RunStudyTab

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)

    class FakeWorker:
        def __init__(self, ns, *a, **k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(workers_mod, "GazeWorker", FakeWorker)

    tab = RunStudyTab()
    # Idle: green go texts; project Run disabled with no project open.
    assert tab._video_go.text() == "▶  Analyze"
    assert "2a7a2a" in tab._video_go.styleSheet()
    assert not tab._project_go.isEnabled()
    assert tab._project_go.text() == "▶  Run"

    tab._set_mode("video")
    tab._quick_video.setText(str(video))
    tab._run_quick()
    # Running: all three cards show a red, clickable Stop.
    for btn in (tab._project_go, tab._video_go, tab._camera_go):
        assert btn.text() == "■  Stop"
        assert "7a2a2a" in btn.styleSheet()
        assert btn.isEnabled()
    tab._poll_timer.stop()


def test_preset_labels_follow_the_store(qapp):
    """The quick cards show 'Preset: <source>' and react to store commits."""
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    assert tab._video_preset.text().startswith("Preset: ")
    assert tab._video_preset.text() == tab._camera_preset.text()
    ns = tab._settings.working_copy()
    ns.ray_length = 4.2
    tab._settings.commit(ns)
    assert tab._video_preset.text().endswith("(modified)")
