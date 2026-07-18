"""W3Y item 8: the Projects-tab Study setup panel (project_setup_panel).

The retired Analyze Footage "Study setup" pane's project-level duties --
pipeline path, participants, conditions, output root, Save project.yaml
-- now live here, edited BEFORE running. These tests pin the populate /
edit / save round-trip against a real tmp project, and the ProjectsTab
hosting (panel loads on show_overview, save refreshes + signals).
"""

import os
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _make_project(tmp_path, *, with_yaml=True) -> Path:
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    (proj / "Inputs" / "Videos" / "a.mp4").write_bytes(b"\x00" * 32)
    (proj / "Inputs" / "Videos" / "b.mp4").write_bytes(b"\x00" * 32)
    if with_yaml:
        (proj / "project.yaml").write_text(yaml.dump({
            "pipeline": "Pipeline/pipeline.yaml",
            "conditions": {"a.mp4": ["collab"]},
            "participants": {"a.mp4": {0: "S1"}},
        }))
    return proj


def test_panel_populates_from_project_yaml(qapp, tmp_path):
    from mindsight.GUI.project_setup_panel import ProjectSetupPanel
    panel = ProjectSetupPanel()
    panel.open_project(_make_project(tmp_path))
    assert panel._pipeline_path.text() == "Pipeline/pipeline.yaml"
    assert panel._source_table.rowCount() == 2
    assert panel._source_table.item(0, 0).text() == "a.mp4"
    assert panel._source_table.item(0, 1).text() == "collab"
    assert panel.dirty is False
    cfg = panel.build_config()
    assert cfg.conditions == {"a.mp4": ["collab"]}
    assert cfg.participants == {"a.mp4": {0: "S1"}}


def test_panel_save_roundtrip(qapp, tmp_path):
    from mindsight.GUI.project_setup_panel import ProjectSetupPanel
    from mindsight.project.runner import load_project_config
    proj = _make_project(tmp_path)
    panel = ProjectSetupPanel()
    panel.open_project(proj)
    panel._output_dir.setText(str(tmp_path / "custom_out"))
    assert panel.dirty is True
    saved = []
    panel.saved.connect(lambda: saved.append(1))
    panel.save()
    assert saved == [1] and panel.dirty is False
    cfg = load_project_config(proj)
    assert cfg.output.directory == str(tmp_path / "custom_out")
    assert cfg.conditions == {"a.mp4": ["collab"]}       # survived the edit


def test_panel_handles_project_without_yaml(qapp, tmp_path):
    from mindsight.GUI.project_setup_panel import ProjectSetupPanel
    panel = ProjectSetupPanel()
    panel.open_project(_make_project(tmp_path, with_yaml=False))
    assert panel._pipeline_path.text() == ""
    assert panel._source_table.rowCount() == 2
    assert panel.dirty is False


def test_projects_tab_hosts_and_signals(qapp, tmp_path):
    from mindsight.GUI.projects_tab import ProjectsTab
    proj = _make_project(tmp_path)
    tab = ProjectsTab()
    tab.show_overview(proj)
    assert tab._setup_panel._project_path == proj
    assert tab._setup_panel._pipeline_path.text() == "Pipeline/pipeline.yaml"
    emitted = []
    tab.setup_saved.connect(emitted.append)
    tab._setup_panel._output_dir.setText("elsewhere")
    tab._setup_panel.save()
    assert emitted == [str(proj)]
