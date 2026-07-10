"""
tests/conftest.py — pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_mindsight_settings(monkeypatch, tmp_path_factory):
    """Redirect the per-user settings dir to a per-test temp dir.

    SettingsManager keys ~/.mindsight off Path.home(), so without this any
    test that constructs MainWindow (closeEvent -> save_last_used) or touches
    add_recent_project writes into the developer's REAL profile. That poisoned
    a live install once (stale worktree weight paths in last_used.json); no
    test run may ever depend on -- or write to -- the real ~/.mindsight.

    Deliberately NOT tmp_path: several tests assert their tmp_path stays
    empty, so the settings dir gets its own private temp tree.
    """
    from mindsight.GUI.settings_manager import SettingsManager

    settings_dir = tmp_path_factory.mktemp("mindsight-settings") / ".mindsight"
    monkeypatch.setattr(SettingsManager, "SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(SettingsManager, "LAST_USED",
                        settings_dir / "last_used.json")
    monkeypatch.setattr(SettingsManager, "PRESETS_DIR",
                        settings_dir / "presets")
    monkeypatch.setattr(SettingsManager, "RECENT_PROJECTS",
                        settings_dir / "recent_projects.json")
