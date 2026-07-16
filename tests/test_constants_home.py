"""Tests for the MINDSIGHT_HOME data-root override in mindsight.constants.

The env var lets a non-editable wheel install redirect Weights/Outputs/Plugins/
projects at a user-visible home.  When unset, PROJECT_ROOT must resolve exactly
as it did before the seam existed (the repository root, two levels up from the
constants module) so the source-tree / editable-install layout is unchanged.
"""
import importlib
from pathlib import Path

import mindsight.constants as constants


def _reload_constants(monkeypatch, home_value):
    """Reload constants with MINDSIGHT_HOME set/unset; restore afterwards."""
    if home_value is None:
        monkeypatch.delenv("MINDSIGHT_HOME", raising=False)
    else:
        monkeypatch.setenv("MINDSIGHT_HOME", str(home_value))
    return importlib.reload(constants)


def test_project_root_default_unset(monkeypatch):
    """Unset -> PROJECT_ROOT is the repo root (constants.py's parent.parent)."""
    mod = _reload_constants(monkeypatch, None)
    expected = Path(mod.__file__).resolve().parent.parent
    assert mod.PROJECT_ROOT == Path(mod.__file__).parent.parent
    assert mod.PROJECT_ROOT.resolve() == expected
    # Derived roots follow PROJECT_ROOT.
    assert mod.OUTPUTS_ROOT == mod.PROJECT_ROOT / "Outputs"


def test_project_root_override(monkeypatch, tmp_path):
    """Set -> PROJECT_ROOT and every derived root resolve under the home dir."""
    mod = _reload_constants(monkeypatch, tmp_path)
    assert mod.PROJECT_ROOT == tmp_path
    assert mod.OUTPUTS_ROOT == tmp_path / "Outputs"

    # weights.py derives WEIGHTS_ROOT / MANIFEST_PATH from PROJECT_ROOT too.
    import mindsight.weights as weights
    weights = importlib.reload(weights)
    assert weights.WEIGHTS_ROOT == tmp_path / "Weights"
    assert weights.MANIFEST_PATH == tmp_path / "weights_manifest.json"


def test_empty_env_falls_back_to_default(monkeypatch):
    """An empty MINDSIGHT_HOME is treated as unset (no empty-path root)."""
    monkeypatch.setenv("MINDSIGHT_HOME", "")
    mod = importlib.reload(constants)
    assert mod.PROJECT_ROOT == Path(mod.__file__).parent.parent


def test_settings_dir_honors_relocation_seams(monkeypatch, tmp_path):
    """v1.1 W0.7: the GUI settings dir follows MINDSIGHT_STATE_DIR, then
    MINDSIGHT_HOME/.mindsight, then ~/.mindsight -- previously hard-keyed to
    Path.home(), so relocated installs shared one ~/.mindsight."""
    from mindsight.GUI.settings_manager import _settings_dir

    monkeypatch.delenv("MINDSIGHT_STATE_DIR", raising=False)
    monkeypatch.delenv("MINDSIGHT_HOME", raising=False)
    assert _settings_dir() == Path.home() / ".mindsight"

    monkeypatch.setenv("MINDSIGHT_HOME", str(tmp_path / "homeA"))
    assert _settings_dir() == tmp_path / "homeA" / ".mindsight"

    monkeypatch.setenv("MINDSIGHT_STATE_DIR", str(tmp_path / "state"))
    assert _settings_dir() == tmp_path / "state"   # wins over MINDSIGHT_HOME

    # Empty values are treated as unset.
    monkeypatch.setenv("MINDSIGHT_STATE_DIR", "")
    monkeypatch.setenv("MINDSIGHT_HOME", "")
    assert _settings_dir() == Path.home() / ".mindsight"


def teardown_module(module):
    """Restore the real (unset-env) module state for later tests in the run."""
    import os
    os.environ.pop("MINDSIGHT_HOME", None)
    importlib.reload(constants)
    import mindsight.weights as weights
    importlib.reload(weights)
