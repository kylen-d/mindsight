"""Projects tab: Plan Session dialog + planned rows in the overview (UP5)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_plan_session_dialog_meta_parsing(qapp):
    from mindsight.GUI.projects_tab import _PlanSessionDialog
    dlg = _PlanSessionDialog()
    dlg._name.setText("session07")
    dlg._participants.setText("S80, S81")
    dlg._conditions.setText("baseline|toys")
    dlg._date.setText("2026-08-01")
    assert dlg.run_id() == "session07"
    meta = dlg.meta()
    assert meta["participants"] == {0: "S80", 1: "S81"}
    assert meta["conditions"] == ["baseline", "toys"]
    assert meta["date"] == "2026-08-01"
    assert "session" not in meta and "notes" not in meta  # empty -> omitted


def test_overview_lists_planned_sessions_awaiting_recording(qapp, tmp_path):
    from mindsight.GUI.projects_tab import ProjectsTab
    from mindsight.project.runner import create_project
    from mindsight.project.staging import plan_run
    proj = create_project(tmp_path, "PlanTest", layout="run_folder")
    plan_run(proj, "session07",
             meta={"participants": {0: "S80"}, "conditions": ["baseline"]})
    tab = ProjectsTab()
    tab.show_overview(proj)
    rows = {tab._runs_table.item(r, 0).text():
            tab._runs_table.item(r, 4).text()
            for r in range(tab._runs_table.rowCount())}
    assert rows.get("session07") == "awaiting recording"
