"""Offscreen-Qt coverage for the startup known-good preset seed (Bug B2).

Bug B2: every phenomena tracker defaulted OFF, the shipped preset carried no
phenomena keys, and the GUI never seeded from the preset -- so a default run
recorded no phenomena data. User ruling 2026-07-09: all phenomena default ON via
the preset, and the GUI seeds from the preset at startup.

These tests pin the seam MainWindow.__init__ -> _seed_from_preset:
  (a) with the preset resolvable, the gaze tab's namespace comes up with every
      phenomena toggle ON (and a preset scalar seeded);
  (b) with no preset resolvable, startup does not raise and the toggles keep
      their schema defaults (OFF);
  (c) a last_used.json session still wins over the preset for the keys it
      carries (defaults < preset seed < last_used restore).

The known_good_preset_path resolver is monkeypatched in the main_window module
namespace rather than by relocating constants.PROJECT_ROOT: PROJECT_ROOT feeds
the import-time-bound Weights root, so patching it can leak a temp weight path
across tests (the stale-weight-path hazard). The PROJECT_ROOT / MINDSIGHT_HOME
resolution itself is covered purely in test_known_good_config.py, where no GUI or
weights are constructed.

CRITICAL isolation: ~/.mindsight is keyed off Path.home() and is NOT isolated by
conftest, so every MainWindow/SettingsManager construction here runs against a
monkeypatched SettingsManager pointed at a temp dir -- never the developer's
real home.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PRESET = REPO_ROOT / "configs" / "pipeline_known_good.yaml"

PHENOMENA_TOGGLES = (
    "joint_attention", "mutual_gaze", "social_ref", "gaze_follow",
    "gaze_aversion", "scanpath", "gaze_leader", "attn_span",
)


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app  # never quit -- shared process singleton


def _isolate_settings(monkeypatch, home_dir: Path):
    """Point SettingsManager at a private .mindsight under *home_dir*."""
    from mindsight.GUI.settings_manager import SettingsManager
    settings_dir = home_dir / ".mindsight"
    monkeypatch.setattr(SettingsManager, "SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(SettingsManager, "LAST_USED",
                        settings_dir / "last_used.json")
    monkeypatch.setattr(SettingsManager, "PRESETS_DIR",
                        settings_dir / "presets")
    monkeypatch.setattr(SettingsManager, "RECENT_PROJECTS",
                        settings_dir / "recent_projects.json")
    return settings_dir


def _seed_resolves_to(monkeypatch, path_or_none):
    """Make the startup seed resolve to *path_or_none* (a preset path or None)."""
    import mindsight.GUI.main_window as mw
    monkeypatch.setattr(mw, "known_good_preset_path", lambda: path_or_none)


def _ship_preset(root: Path) -> Path:
    """Copy the real shipped preset into *root*/configs/ and return its path."""
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    dst = configs / "pipeline_known_good.yaml"
    shutil.copy(REAL_PRESET, dst)
    return dst


def _build_window():
    from mindsight.GUI.main_window import MainWindow
    return MainWindow()


def test_seed_enables_all_phenomena(qapp, monkeypatch, tmp_path):
    """With the preset resolvable and no saved session, the gaze-tab namespace
    comes up with every phenomena toggle enabled (and a preset scalar seeded)."""
    preset = _ship_preset(tmp_path / "resources")
    _seed_resolves_to(monkeypatch, preset)
    _isolate_settings(monkeypatch, tmp_path / "home")  # no last_used.json

    win = _build_window()
    try:
        ns = win._gaze_tab._build_namespace()
        for toggle in PHENOMENA_TOGGLES:
            assert getattr(ns, toggle) is True, f"{toggle} not seeded on"
        # A preset scalar seeds too (detection conf 0.25, not the 0.35 default).
        assert ns.conf == 0.25
    finally:
        win.close()


def test_no_preset_keeps_defaults_and_does_not_raise(qapp, monkeypatch, tmp_path):
    """With no preset resolvable, startup must not raise and the phenomena
    toggles keep their schema defaults (OFF)."""
    _seed_resolves_to(monkeypatch, None)
    _isolate_settings(monkeypatch, tmp_path / "home")

    win = _build_window()             # must not raise
    try:
        ns = win._gaze_tab._build_namespace()
        for toggle in PHENOMENA_TOGGLES:
            assert getattr(ns, toggle) is False, f"{toggle} unexpectedly on"
    finally:
        win.close()


def test_last_used_overrides_preset(qapp, monkeypatch, tmp_path):
    """A saved session wins over the preset for the keys it carries: defaults <
    preset seed < last_used restore. The preset sets conf 0.25; a last_used
    carrying conf 0.5 must win."""
    preset = _ship_preset(tmp_path / "resources")
    _seed_resolves_to(monkeypatch, preset)
    settings_dir = _isolate_settings(monkeypatch, tmp_path / "home")

    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "last_used.json").write_text(
        json.dumps({"_version": 1, "conf": 0.5}))

    win = _build_window()
    try:
        ns = win._gaze_tab._build_namespace()
        assert ns.conf == 0.5     # last_used overrode the preset's 0.25
    finally:
        win.close()
