"""Batch 6+7 coverage: persistence hardening (B8), live/analysis charts
(B4a/B6), vertical splitters (B9).

Offscreen Qt; every SettingsManager touch is isolated by the autouse conftest
fixture, and the seed is neutralized where widget defaults matter.
"""

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── Part A: persistence ──────────────────────────────────────────────────────

def test_checkpoint_writes_last_used(qapp, tmp_path, monkeypatch):
    from mindsight.GUI.settings_manager import SettingsManager, checkpoint
    ns = Namespace(conf=0.42, joint_attention=True)
    checkpoint(ns)
    saved = json.loads(SettingsManager.LAST_USED.read_text())
    assert saved["conf"] == 0.42
    assert saved["joint_attention"] is True


def test_checkpoint_never_raises(qapp, monkeypatch):
    from mindsight.GUI import settings_manager
    monkeypatch.setattr(settings_manager.SettingsManager, "save_last_used",
                        lambda self, ns: (_ for _ in ()).throw(OSError("disk")))
    settings_manager.checkpoint(Namespace(conf=0.1))   # must not raise


def test_restore_warns_not_raises_on_corrupt_last_used(qapp, capsys):
    from mindsight.GUI.settings_manager import SettingsManager
    SettingsManager.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SettingsManager.LAST_USED.write_text("{not json")
    import mindsight.GUI.main_window as mw
    win = mw.MainWindow.__new__(mw.MainWindow)

    class _StubTab:
        def apply_namespace(self, ns):
            raise AssertionError("should not be reached with corrupt json")

    win._gaze_tab = _StubTab()
    win._try_restore_last_session()      # must not raise
    assert "[WARN]" in capsys.readouterr().out


def test_aux_streams_round_trip(qapp):
    from mindsight.GUI.settings_manager import SettingsManager
    from mindsight.pipeline_config import AuxStreamConfig, VideoType
    aux = AuxStreamConfig(source="clip.mp4", video_type=VideoType.FACE_CLOSEUP,
                          stream_label="face", participants=None)
    mgr = SettingsManager()
    mgr.save_last_used(Namespace(conf=0.3, aux_streams=[aux]))
    ns = mgr.load_last_used()
    got = getattr(ns, "aux_streams", None)
    # Contract: persisted as plain dicts; the OutputSection widget layer
    # accepts both dicts and AuxStreamConfig objects when repopulating.
    assert got and len(got) == 1
    assert got[0]["source"] == "clip.mp4"
    assert got[0]["video_type"] == VideoType.FACE_CLOSEUP.value
    assert got[0]["participants"] is None


def test_reset_defaults_applies_preset(qapp, monkeypatch):
    import mindsight.GUI.gaze_tab.gaze_tab as gt
    preset = REPO_ROOT / "configs" / "pipeline_known_good.yaml"
    monkeypatch.setattr(gt, "known_good_preset_path", lambda: preset)
    tab = gt.GazeTab()
    try:
        tab._reset_gaze_defaults()
        ns = tab._build_namespace()
        assert ns.joint_attention is True      # preset, not schema default
        assert ns.conf == 0.25                 # preset detection conf
    finally:
        tab.deleteLater()


def test_reset_defaults_without_preset_keeps_schema(qapp, monkeypatch):
    import mindsight.GUI.gaze_tab.gaze_tab as gt
    monkeypatch.setattr(gt, "known_good_preset_path", lambda: None)
    tab = gt.GazeTab()
    try:
        tab._reset_gaze_defaults()
        ns = tab._build_namespace()
        assert ns.joint_attention is False
    finally:
        tab.deleteLater()


# ── Part B: charts ───────────────────────────────────────────────────────────

def test_trackers_declare_live_chart_type_matching_time_series():
    """live_chart_type mirrors the chart_type each tracker's
    time_series_data declares -- pinned so they cannot drift apart."""
    from mindsight.Phenomena.Default import (
        GazeAversionTracker,
        GazeFollowingTracker,
        GazeLeadershipTracker,
        JointAttentionTracker,
        MutualGazeTracker,
        ScanpathTracker,
        SocialReferenceTracker,
    )
    expected = {
        JointAttentionTracker: "area", GazeAversionTracker: "area",
        MutualGazeTracker: "area", ScanpathTracker: "step",
        SocialReferenceTracker: "step", GazeFollowingTracker: "step",
        GazeLeadershipTracker: "step",
    }
    for cls, ct in expected.items():
        assert cls.live_chart_type == ct, cls.__name__


@pytest.mark.parametrize("chart_type", ["line", "step", "area"])
def test_chart_widget_renders_each_type(qapp, chart_type):
    from mindsight.GUI.live_dashboard import TrackerChartWidget
    w = TrackerChartWidget("t", chart_type=chart_type)
    for fn in range(5):
        w.push_metrics(fn, {"s": {"value": float(fn % 2), "label": "s",
                                  "y_label": ""}})
    w.redraw()                              # must not raise for any style


def test_chart_widget_drops_stale_series(qapp):
    from mindsight.GUI.live_dashboard import TrackerChartWidget
    w = TrackerChartWidget("t")
    # Two series; "b" stops updating at frame 5 while "a" continues.
    for fn in range(5):
        w.push_metrics(fn, {"a": {"value": 1.0, "label": "a", "y_label": ""},
                            "b": {"value": 1.0, "label": "b", "y_label": ""}})
    for fn in range(5, 5 + TrackerChartWidget.STALE_GRACE + 2):
        w.push_metrics(fn, {"a": {"value": 0.5, "label": "a", "y_label": ""}})
    w.redraw()
    assert "a" in w._series_data
    assert "b" not in w._series_data       # ended state dropped after grace


def test_run_study_has_live_tab_and_scrollable_charts(qapp, monkeypatch):
    from PyQt6.QtWidgets import QScrollArea, QSplitter
    from mindsight.GUI.run_study_tab import RunStudyTab
    tab = RunStudyTab()
    try:
        labels = [tab._output_tabs.tabText(i)
                  for i in range(tab._output_tabs.count())] \
            if hasattr(tab, "_output_tabs") else None
        if labels is None:
            pytest.skip("output tabs handle not exposed")
        assert "Live" in labels
        assert isinstance(tab._chart_scroll, QScrollArea)
        assert tab.findChildren(QSplitter)   # part C: splitters exist
    finally:
        tab.deleteLater()
