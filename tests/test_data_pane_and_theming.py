"""Projects data pane (charts + CSV preview) and Fusion theming."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _write_outputs(tmp_path):
    """A minimal summary + events CSV pair shaped like real outputs."""
    summary = tmp_path / "vid_summary.csv"
    summary.write_text(
        "video_name,conditions,phenomenon,participant,partner,object,"
        "metric,value\n"
        "vid,,object_look_time,P0,,cup,pct_of_video,12.5\n"
        "vid,,object_look_time,P1,,cup,pct_of_video,3.5\n")
    events = tmp_path / "vid_Events.csv"
    events.write_text(
        "frame,time_s,participant,object\n"
        "1,0.03,P0,cup\n"
        "2,0.07,P1,cup\n")
    return summary, events


def test_data_pane_renders_outputs_and_empty_state(qapp, tmp_path):
    from mindsight.GUI.data_pane import RunDataPane
    from mindsight.GUI.run_outputs import RunOutputs
    summary, events = _write_outputs(tmp_path)
    out = RunOutputs(run_id="vid", stem="vid",
                     csv_paths=(summary, events),
                     summary=summary, events=events)
    pane = RunDataPane()
    pane.set_outputs("vid", out)
    assert pane._title.text() == "Data — vid"
    assert pane._split.isVisibleTo(pane)
    assert pane._csv_pick.count() == 2
    assert pane._csv_table.rowCount() > 0
    # No outputs -> hint, no panels.
    pane.set_outputs("planned01", None)
    assert not pane._split.isVisibleTo(pane)
    assert "No outputs yet" in pane._hint.text()


def test_apply_theme_steers_native_color_scheme(qapp):
    # Eyes-on r2: no custom palettes -- Qt's own scheme control (6.8+).
    # The offscreen QPA reports Unknown regardless of the override, so the
    # readback assert only applies where the platform reports schemes; every
    # mode must at minimum apply without raising.
    from PyQt6.QtCore import Qt
    from mindsight.GUI.theming import apply_theme
    apply_theme(qapp, "dark")
    scheme = qapp.styleHints().colorScheme()
    if scheme != Qt.ColorScheme.Unknown:  # real desktop platforms
        assert scheme == Qt.ColorScheme.Dark
        apply_theme(qapp, "light")
        assert qapp.styleHints().colorScheme() == Qt.ColorScheme.Light
    apply_theme(qapp, "light")
    apply_theme(qapp, "auto")   # unsets the override; must not raise
    apply_theme(qapp, "bogus")  # unknown mode behaves like auto
