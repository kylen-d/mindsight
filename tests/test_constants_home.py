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


def teardown_module(module):
    """Restore the real (unset-env) module state for later tests in the run."""
    import os
    os.environ.pop("MINDSIGHT_HOME", None)
    importlib.reload(constants)
    import mindsight.weights as weights
    importlib.reload(weights)
