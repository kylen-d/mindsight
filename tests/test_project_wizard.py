"""UP3: Build New Project wizard + Projects tab + middle-frame helper."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_video(path: Path, frames: int = 9, w: int = 64, h: int = 48):
    """Synthesize a tiny decodable mp4."""
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         10.0, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)
        wr.write(frame)
    wr.release()
    return path


def test_middle_frame_pixmap_decodes(qapp, tmp_path):
    from mindsight.GUI.widgets import middle_frame_pixmap
    video = _make_video(tmp_path / "clip.mp4")
    px = middle_frame_pixmap(video, 320, 240)
    assert px is not None and not px.isNull()
    # Unreadable path -> None, no raise.
    assert middle_frame_pixmap(tmp_path / "nope.mp4") is None


def _wizard(qapp, settings=None):
    from mindsight.GUI.project_wizard import BuildProjectWizard
    return BuildProjectWizard(settings=settings)


def test_wizard_full_flow_creates_project(qapp, tmp_path, monkeypatch):
    """Study -> videos -> tag one of two -> KG preset -> create: run folders,
    run.yaml contents, pipeline copy, notes.md, collision-safe layout."""
    from PyQt6.QtWidgets import QMessageBox

    v1 = _make_video(tmp_path / "src" / "dyad01.mp4")
    v2 = _make_video(tmp_path / "src" / "dyad02.mp4")
    preset = tmp_path / "kg.yaml"
    preset.write_text("detection:\n  conf: 0.25\n")
    import mindsight.config_compat as cc
    monkeypatch.setattr(cc, "known_good_preset_path", lambda: preset)
    # Auto-accept the "untagged videos" question.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    wiz = _wizard(qapp)
    wiz._name.setText("MyStudy")
    wiz._location.setText(str(tmp_path))
    wiz._people.setValue(2)
    wiz._cond_edit.setText("warm")
    wiz._add_condition()
    wiz._cond_edit.setText("cold")
    wiz._add_condition()
    wiz._notes.setPlainText("pilot batch")
    wiz._go_next()                        # -> videos
    wiz._append_videos([str(v1), str(v2)])
    wiz._go_next()                        # -> tag (validates, builds form)
    assert wiz._pages.currentIndex() == 2
    # Tag video 1: two participants + one condition.
    wiz._participant_edits[0].setText("S70")
    wiz._participant_edits[1].setText("S71")
    wiz._condition_checks[0].setChecked(True)     # warm
    wiz._session_edit.setText("1")
    wiz._go_next()                        # -> pipeline (warns re untagged v2)
    assert wiz._pages.currentIndex() == 3
    assert wiz._pipe_kg.isChecked()
    wiz._go_next()                        # -> review
    assert wiz._pages.currentIndex() == 4
    wiz._go_next()                        # create

    proj = tmp_path / "MyStudy"
    assert wiz.created_path == proj
    assert (proj / "project.yaml").is_file()
    assert (proj / "notes.md").read_text().strip() == "pilot batch"
    assert (proj / "Pipeline" / "pipeline.yaml").read_text() == \
        preset.read_text()
    run1 = proj / "Inputs" / "Runs" / "dyad01"
    assert (run1 / "dyad01.mp4").is_file()
    meta = yaml.safe_load((run1 / "run.yaml").read_text())
    assert meta["participants"] == {0: "S70", 1: "S71"}
    assert meta["conditions"] == ["warm"]
    assert str(meta["session"]) == "1"
    assert "date" in meta                 # prefilled from file mtime
    run2 = proj / "Inputs" / "Runs" / "dyad02"
    assert (run2 / "dyad02.mp4").is_file()
    assert not (run2 / "run.yaml").exists()   # untagged: bare folder is legal
    # Copy mode: originals untouched.
    assert v1.is_file() and v2.is_file()


def test_wizard_validation_gates(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    wiz = _wizard(qapp)
    wiz._go_next()
    assert warned and "name" in warned[-1]         # no name
    wiz._name.setText("X")
    wiz._go_next()
    assert "folder" in warned[-1]                  # no location
    wiz._location.setText(str(tmp_path))
    wiz._go_next()
    assert wiz._pages.currentIndex() == 1
    wiz._go_next()
    assert "at least one video" in warned[-1]      # no videos


def test_wizard_duplicate_run_names_blocked(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    v1 = _make_video(tmp_path / "a" / "same.mp4")
    v2 = _make_video(tmp_path / "b" / "same.mp4")
    wiz = _wizard(qapp)
    wiz._name.setText("X")
    wiz._location.setText(str(tmp_path))
    wiz._go_next()
    wiz._append_videos([str(v1), str(v2)])
    wiz._go_next()
    assert "unique" in warned[-1]
    assert wiz._pages.currentIndex() == 1          # stayed on videos


def test_wizard_ordinal_labels():
    from mindsight.GUI.project_wizard import _ordinal_label
    assert _ordinal_label(0, 1) == "Person"
    assert _ordinal_label(0, 3) == "Leftmost person"
    assert _ordinal_label(1, 3) == "2nd from left"
    assert _ordinal_label(2, 3) == "3rd from left"
    assert _ordinal_label(3, 4) == "4th from left"


def test_projects_tab_landing_and_overview(qapp, tmp_path):
    """Landing lists recents (missing greyed), overview shows runs + notes,
    open_in_analyze emits."""
    from mindsight.GUI.projects_tab import ProjectsTab
    from mindsight.GUI.settings_manager import SettingsManager
    from mindsight.project.runner import create_project
    from mindsight.project.staging import stage_run

    proj = create_project(tmp_path, "TabStudy")
    video = _make_video(tmp_path / "vid" / "s1.mp4")
    stage_run(proj, video, {"participants": {0: "S70"},
                            "conditions": "warm"})
    (proj / "notes.md").write_text("hello study\n")
    mgr = SettingsManager()
    mgr.add_recent_project(str(proj))
    mgr.add_recent_project(str(tmp_path / "gone-project"))

    tab = ProjectsTab()
    assert tab._stack.currentIndex() == 0
    texts = [tab._project_table.item(r, 0).text()
             for r in range(tab._project_table.rowCount())]
    assert any("gone-project" in t and "(missing)" in t for t in texts)
    assert any("TabStudy" in t for t in texts)
    assert tab._landing_hint.isHidden() or not tab._landing_hint.isVisible()

    emitted = []
    tab.open_in_analyze.connect(emitted.append)
    tab.show_overview(proj)
    assert tab._stack.currentIndex() == 1
    assert tab._ov_name.text() == "TabStudy"
    assert "hello study" in tab._ov_notes.text()
    runs = [tab._runs_table.item(r, 0).text()
            for r in range(tab._runs_table.rowCount())]
    assert "s1" in runs
    tab.open_in_analyze.emit(str(proj))
    assert emitted == [str(proj)]
    tab.show_landing()
    assert tab._stack.currentIndex() == 0


def test_main_window_has_projects_tab_second(qapp):
    from mindsight.GUI.main_window import (_TAB_ANALYZE, _TAB_MODELS,
                                           _TAB_PROJECTS, _TAB_TUNING,
                                           _TAB_VP)
    assert (_TAB_ANALYZE, _TAB_PROJECTS, _TAB_VP, _TAB_TUNING,
            _TAB_MODELS) == (0, 1, 2, 3, 4)
