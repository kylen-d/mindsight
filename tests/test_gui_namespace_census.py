"""Offscreen census of the default GUI namespace (SP3 D8 golden).

``gaze_tab._build_namespace()`` is the single seam every GUI run flows
through (presets, workers, pipeline import/export, project runs).  A dest
that silently disappears does not crash -- downstream ``getattr(ns, dest,
default)`` substitutes a default and CHANGES RUNS.  This module welds the
full default namespace (dests + values, serialized exactly the way
``SettingsManager`` persists sessions) to a committed golden:
``tests/data/gui_namespace_golden.json``.

The golden is regenerated ONLY deliberately (reviewed diff stated in
advance), never casually -- same discipline as the CLI parser goldens.

The second test replays a stale pre-SP3 ``last_used.json`` fixture
(``tests/data/last_used_pre_sp3.json``: the golden minus a few dests plus
unknown dests) through ``apply_namespace`` -- old saved sessions from
``~/.mindsight/`` are applied into the new GUI on launch and must be
tolerated gracefully (ignore unknown dests, default missing ones).

Fast (no models loaded, no video); offscreen Qt only.
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "data" / "gui_namespace_golden.json"
STALE_FIXTURE_PATH = REPO_ROOT / "tests" / "data" / "last_used_pre_sp3.json"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app  # never quit -- shared process singleton


@pytest.fixture()
def main_window(qapp, monkeypatch, tmp_path):
    """A MainWindow whose last-session restore is neutralized.

    The census must capture pure widget DEFAULTS -- the developer's real
    ``~/.mindsight/last_used.json`` must never leak into it.
    """
    from mindsight.GUI.settings_manager import SettingsManager
    settings_dir = tmp_path / ".mindsight"
    monkeypatch.setattr(SettingsManager, "SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(SettingsManager, "LAST_USED",
                        settings_dir / "last_used.json")
    monkeypatch.setattr(SettingsManager, "PRESETS_DIR",
                        settings_dir / "presets")

    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()
    yield win
    win.close()


def _census(win) -> dict:
    from mindsight.GUI.settings_manager import SettingsManager
    ns = win._gaze_tab._build_namespace()
    return SettingsManager._ns_to_dict(ns)


def test_default_namespace_matches_golden(main_window):
    """The full default namespace census equals the committed golden."""
    census = _census(main_window)
    golden = json.loads(GOLDEN_PATH.read_text())

    if census == golden:
        return

    # Full per-key diff on failure -- a bare assert on a 140-key dict is
    # undebuggable.
    missing = sorted(set(golden) - set(census))
    extra = sorted(set(census) - set(golden))
    changed = {
        k: {"golden": golden[k], "census": census[k]}
        for k in sorted(set(golden) & set(census))
        if golden[k] != census[k]
    }
    pytest.fail(
        "GUI namespace census diverged from tests/data/gui_namespace_golden.json\n"
        f"dests missing from census ({len(missing)}): {missing}\n"
        f"dests new in census ({len(extra)}): {extra}\n"
        f"values changed ({len(changed)}):\n"
        + json.dumps(changed, indent=2, sort_keys=True)
    )


def test_stale_last_used_session_applies_cleanly(main_window):
    """A pre-SP3 saved session (missing + unknown dests) must not raise,
    and must leave the dest census intact (unknown ignored, missing
    defaulted)."""
    from mindsight.GUI.settings_manager import SettingsManager

    stale = json.loads(STALE_FIXTURE_PATH.read_text())
    ns = SettingsManager._dict_to_ns(stale)

    main_window._gaze_tab.apply_namespace(ns)  # must not raise

    census = _census(main_window)
    golden = json.loads(GOLDEN_PATH.read_text())

    # The dest census is defined by the widgets, not by whatever an old
    # session file happened to contain.
    assert set(census) == set(golden), (
        f"stale-session replay changed the dest census:\n"
        f"missing: {sorted(set(golden) - set(census))}\n"
        f"extra:   {sorted(set(census) - set(golden))}"
    )
    # Unknown dests from the stale file must not leak into the namespace.
    assert "sp2_era_removed_knob" not in census
    assert "definitely_not_a_dest" not in census
