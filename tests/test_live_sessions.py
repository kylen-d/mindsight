"""UP5: live sessions -- capture engine, planned runs, attach, camera meta."""

from pathlib import Path

import pytest
import yaml


def _make_video(path: Path, frames: int = 15, w: int = 64, h: int = 48):
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (w, h))
    for i in range(frames):
        wr.write(np.full((h, w, 3), i * 16 % 255, dtype=np.uint8))
    wr.release()
    return path


def _frame_count(path: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


# ── LiveRecorder (file-backed "camera") ──────────────────────────────────────

def test_live_recorder_records_all_frames_with_sidecar(tmp_path):
    from mindsight.io.live_capture import LiveRecorder
    src = _make_video(tmp_path / "cam.mp4", frames=15)
    rec = LiveRecorder(str(src), tmp_path / "session.mp4")
    rec.start()
    rec.join(timeout=30)          # file source ends naturally
    assert not rec.is_alive()
    assert rec.error is None
    assert rec.frames_captured == 15
    assert rec.dest.is_file() and _frame_count(rec.dest) == 15
    assert rec.measured_fps and rec.measured_fps > 0
    assert rec.sidecar is not None and rec.sidecar.is_file()
    lines = rec.sidecar.read_text().splitlines()
    assert lines[0].startswith("# capture_start_epoch=")
    assert "measured_fps=" in lines[0]
    assert lines[1] == "frame,t_wall"
    assert len(lines) == 2 + 15   # header comment + csv header + one per frame


def test_live_recorder_bad_source_sets_error(tmp_path):
    from mindsight.io.live_capture import LiveRecorder
    rec = LiveRecorder(str(tmp_path / "nope.mp4"), tmp_path / "out.mp4")
    rec.start()
    rec.join(timeout=10)
    assert rec.error and "cannot open" in rec.error
    assert not (tmp_path / "out.mp4").exists()


# ── Planned sessions ─────────────────────────────────────────────────────────

def _project_with_one_run(tmp_path):
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run
    proj = create_project(tmp_path, "LiveStudy")
    stage_run(proj, _make_video(tmp_path / "vid" / "done1.mp4"),
              {"participants": {0: "S70"}})
    return proj


def test_plan_run_creates_planned_folder(tmp_path):
    from mindsight.project.staging import (discover_run_specs, plan_run,
                                           planned_runs)
    proj = _project_with_one_run(tmp_path)
    folder = plan_run(proj, "session5",
                      {"participants": {0: "S80", 1: "S81"},
                       "conditions": "warm", "session": "5"})
    assert folder == proj / "Inputs" / "Runs" / "session5"
    assert (folder / "run.yaml").is_file()
    planned = planned_runs(proj)
    assert [p.run_id for p in planned] == ["session5"]
    assert planned[0].meta.pid_map == {0: "S80", 1: "S81"}
    # The strict producer SKIPS planned sessions instead of raising.
    specs = discover_run_specs(proj)
    assert [s.run_id for s in specs] == ["done1"]
    # Collision-safe: planning the same id again suffixes.
    assert plan_run(proj, "session5").name == "session5_2"
    # Empty meta still writes the run.yaml marker.
    bare = plan_run(proj, "bare")
    assert (bare / "run.yaml").is_file()


def test_videoless_folder_without_yaml_still_fails(tmp_path):
    from mindsight.project.staging import discover_run_specs
    proj = _project_with_one_run(tmp_path)
    (proj / "Inputs" / "Runs" / "junk").mkdir()
    with pytest.raises(ValueError, match="has no video"):
        discover_run_specs(proj)


def test_preflight_reports_awaiting_recording(tmp_path):
    from mindsight.project.project import Project
    from mindsight.project.staging import plan_run
    proj = _project_with_one_run(tmp_path)
    plan_run(proj, "session5", {"session": "5"})
    report = Project.open(str(proj)).preflight()
    runs_check = next(c for c in report.checks if c.id == "runs_discovered")
    assert runs_check.severity == "ok"
    assert "1 session(s) awaiting recording" in runs_check.message

    # All-planned project: warn, not fail.
    from mindsight.project.runner import create_project
    proj2 = create_project(tmp_path, "AllPlanned")
    plan_run(proj2, "s1")
    report2 = Project.open(str(proj2)).preflight()
    runs_check2 = next(c for c in report2.checks if c.id == "runs_discovered")
    assert runs_check2.severity == "warn"
    assert "awaiting" in runs_check2.message


# ── attach_recording ─────────────────────────────────────────────────────────

def test_attach_recording_fills_planned_run(tmp_path):
    from mindsight.project.staging import (attach_recording,
                                           discover_run_specs, plan_run,
                                           planned_runs)
    proj = _project_with_one_run(tmp_path)
    plan_run(proj, "session5", {"participants": {0: "S80"},
                                "conditions": "warm"})
    recording = _make_video(tmp_path / "tmp_capture.mp4")
    sidecar = tmp_path / "tmp_capture_capture_timestamps.csv"
    sidecar.write_text("frame,t_wall\n0,0.0\n")

    dest = attach_recording(proj, "session5", recording, sidecar=sidecar,
                            meta={"session": "5"})
    assert dest == proj / "Inputs" / "Runs" / "session5" / "session5.mp4"
    assert dest.is_file()
    assert not recording.exists()               # move mode (live temp file)
    assert (dest.parent / "session5_capture_timestamps.csv").is_file()
    assert not planned_runs(proj)               # no longer planned
    spec = next(s for s in discover_run_specs(proj)
                if s.run_id == "session5")
    assert spec.pid_map == {0: "S80"}           # plan meta preserved
    assert spec.conditions == "warm"
    assert spec.meta.get("session") == "5"      # attach meta merged
    # Second attach to the same run refuses (video already present).
    with pytest.raises(ValueError, match="already has a video"):
        attach_recording(proj, "session5", _make_video(tmp_path / "x.mp4"))


def test_attach_recording_copy_mode_preserves_external_file(tmp_path):
    """UP5r2: footage from a separate device attaches by COPY."""
    from mindsight.project.staging import attach_recording
    proj = _project_with_one_run(tmp_path)
    external = _make_video(tmp_path / "gopro" / "S80_session5.mp4")
    dest = attach_recording(proj, "session5", external, mode="copy",
                            meta={"participants": {0: "S80"}})
    assert dest.is_file() and external.is_file()    # original untouched
    data = yaml.safe_load(
        (proj / "Inputs" / "Runs" / "session5" / "run.yaml").read_text())
    assert data["participants"] == {0: "S80"}


# ── camera_run_spec session details (UP5 ruling 2) ──────────────────────────

def test_camera_run_spec_carries_session_meta(tmp_path):
    from mindsight.project.staging import camera_run_spec
    spec = camera_run_spec(0, str(tmp_path), meta={
        "participants": {0: "S80", 1: "S81"},
        "conditions": ["warm"], "session": "5", "notes": "one-off"})
    assert spec.pid_map == {0: "S80", 1: "S81"}
    assert spec.conditions == "warm"
    assert spec.meta == {"session": "5", "notes": "one-off"}
    # Meta stays optional.
    assert camera_run_spec(0, str(tmp_path)).pid_map == {}


# ── GUI: record flow / planned rows / camera session details ────────────────

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_record_dialog_prefills_from_planned(qapp, tmp_path):
    from mindsight.GUI.record_session_dialog import RecordSessionDialog
    from mindsight.project.staging import plan_run, planned_runs
    proj = _project_with_one_run(tmp_path)
    plan_run(proj, "session5", {"participants": {0: "S80", 1: "S81"},
                                "conditions": ["warm"], "session": "5"})
    dlg = RecordSessionDialog(planned_runs(proj), preselect="session5")
    assert dlg._use_planned_radio.isChecked()
    assert dlg._participants.text() == "S80, S81"
    assert dlg._conditions.text() == "warm"
    assert dlg._session.text() == "5"
    meta = dlg._build_meta()
    assert meta["participants"] == {0: "S80", 1: "S81"}
    assert meta["conditions"] == ["warm"]
    # No planned sessions -> new-session mode with a dated default id.
    dlg2 = RecordSessionDialog([])
    assert dlg2._new_radio.isChecked()
    assert dlg2._run_id_edit.text().startswith("session_")


def test_recording_flow_stages_and_launches_analysis(qapp, tmp_path,
                                                     monkeypatch):
    """Record (file-backed camera) -> End Session -> staged as the run's
    primary video with meta + sidecar -> analysis batch launched."""
    import mindsight.io.live_capture as lc
    from mindsight.GUI.run_study_tab import RunStudyTab
    from mindsight.project.staging import plan_run, planned_runs

    proj = _project_with_one_run(tmp_path)
    plan_run(proj, "session5", {"participants": {0: "S80"}})
    cam_file = _make_video(tmp_path / "fake_cam.mp4", frames=12)

    real = lc.LiveRecorder

    class FileBackedRecorder(real):
        def __init__(self, source, dest, **kw):
            super().__init__(str(cam_file), dest, **kw)

    monkeypatch.setattr(lc, "LiveRecorder", FileBackedRecorder)

    tab = RunStudyTab()
    tab._open_project(str(proj))
    assert "session5" in tab._planned_ids
    # Planned row rendered as awaiting recording.
    texts = [tab._runs_table.item(r, 4).text()
             for r in range(tab._runs_table.rowCount())]
    assert "awaiting recording" in texts

    started = []
    monkeypatch.setattr(tab, "_start", lambda: started.append(True))
    tab._start_session_recording(0, "session5", {"session": "5"})
    assert tab._recorder is not None
    assert tab._project_go.text() == "■  End Session"
    assert not tab._mode_btns["video"].isEnabled()
    tab._recorder.join(timeout=30)          # file source drains quickly
    tab._end_session_recording()

    staged = proj / "Inputs" / "Runs" / "session5" / "session5.mp4"
    assert staged.is_file() and _frame_count(staged) == 12
    assert (staged.parent / "session5_capture_timestamps.csv").is_file()
    data = yaml.safe_load((staged.parent / "run.yaml").read_text())
    assert data["participants"] == {0: "S80"}       # plan meta kept
    assert str(data["session"]) == "5"              # dialog meta merged
    assert started == [True]                        # analysis launched
    assert tab._recorder is None
    assert tab._mode_btns["video"].isEnabled()
    assert not planned_runs(proj)
    tab._poll_timer.stop()
    tab._record_timer.stop()


def test_camera_card_session_details_flow_into_quick_run(qapp, tmp_path,
                                                         monkeypatch):
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
    tab._cam_participants.setText("S80, S81")
    tab._cam_conditions.setText("warm")
    tab._cam_session.setText("5")
    tab._run_quick()
    assert len(started) == 1
    out_dir = tmp_path / "Outputs" / "camera0"
    session_files = list(out_dir.glob("*_session.yaml"))
    assert len(session_files) == 1
    data = yaml.safe_load(session_files[0].read_text())
    assert data["participants"] == {0: "S80", 1: "S81"}
    assert data["conditions"] == ["warm"]
    tab._poll_timer.stop()


def test_wizard_creates_planned_sessions(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog, QMessageBox

    from mindsight.GUI.project_wizard import BuildProjectWizard
    from mindsight.project.staging import planned_runs

    preset = tmp_path / "kg.yaml"
    preset.write_text("detection:\n  conf: 0.25\n")
    import mindsight.config_compat as cc
    monkeypatch.setattr(cc, "known_good_preset_path", lambda: preset)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QInputDialog, "getInt",
                        lambda *a, **k: (2, True))

    wiz = BuildProjectWizard()
    wiz._name.setText("PlannedStudy")
    wiz._location.setText(str(tmp_path))
    wiz._go_next()                       # -> videos
    wiz._add_planned_sessions()          # 2 planned, no files at all
    assert len(wiz._videos) == 2
    assert all(v["kind"] == "planned" for v in wiz._videos)
    wiz._go_next()                       # planned-only is allowed
    assert wiz._pages.currentIndex() == 2
    wiz._participant_edits[0].setText("S90")   # tag session01
    wiz._go_next()                       # warns re untagged -> Yes
    wiz._go_next()                       # pipeline -> review
    wiz._go_next()                       # create

    proj = tmp_path / "PlannedStudy"
    assert wiz.created_path == proj
    planned = planned_runs(proj)
    assert [p.run_id for p in planned] == ["session01", "session02"]
    assert planned[0].meta.pid_map == {0: "S90"}
