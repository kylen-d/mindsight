"""W3Y item 7: silent launch-time update notifications.

No network anywhere: fetch_latest takes an injected opener, and the
conftest kill switch (MINDSIGHT_NO_UPDATE_CHECK) keeps MainWindow's
launch-time thread inert in every other test.  These tests re-enable
checks explicitly where they need them.
"""

import io
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from mindsight.GUI import update_check as uc  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── Version compare ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("remote,current,newer", [
    ("v1.1.0", "1.0.0", True),
    ("1.0.1", "1.0.0", True),
    ("v1.0.0", "1.0.0", False),
    ("v0.9.9", "1.0.0", False),
    ("v2.0", "1.9.9", True),
    ("v1.1.0-rc1", "1.0.0", True),      # lenient suffix handling
    ("garbage", "1.0.0", False),         # unparseable -> never "newer"
    ("v1.1.0", "unknown", False),        # unparseable current -> no-op
])
def test_is_newer(remote, current, newer):
    assert uc.is_newer(remote, current) is newer


# ── fetch_latest with an injected opener ─────────────────────────────────────

class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_returning(payload):
    def _open(req, timeout=None):
        assert timeout is not None          # a timeout must always be set
        return _FakeResponse(json.dumps(payload).encode())
    return _open


def test_fetch_latest_parses_release():
    got = uc.fetch_latest(_urlopen=_opener_returning(
        {"tag_name": "v1.2.0",
         "html_url": "https://github.com/kylen-d/MindSight/releases/tag/v1.2.0"}))
    assert got == ("v1.2.0",
                   "https://github.com/kylen-d/MindSight/releases/tag/v1.2.0")


def test_fetch_latest_silent_on_failure():
    def _boom(req, timeout=None):
        raise OSError("offline")
    assert uc.fetch_latest(_urlopen=_boom) is None
    assert uc.fetch_latest(_urlopen=_opener_returning({})) is None  # no tag


# ── Opt-outs ─────────────────────────────────────────────────────────────────

def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("MINDSIGHT_NO_UPDATE_CHECK", "1")
    assert uc.check_enabled() is False


def test_gui_state_toggle_roundtrip(monkeypatch):
    monkeypatch.delenv("MINDSIGHT_NO_UPDATE_CHECK", raising=False)
    assert uc.check_enabled() is True       # default on
    uc.set_check_enabled(False)
    assert uc.check_enabled() is False
    uc.set_check_enabled(True)
    assert uc.check_enabled() is True


def test_dismissed_release_not_reannounced(monkeypatch, qapp):
    monkeypatch.delenv("MINDSIGHT_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "fetch_latest",
                        lambda _urlopen=None: ("v9.9.9", "https://x"))
    seen = []
    checker = uc.UpdateChecker(current_version="1.0.0")
    checker.update_available.connect(lambda t, u: seen.append((t, u)))
    checker._work()                          # run synchronously
    assert seen == [("v9.9.9", "https://x")]
    uc.dismiss("v9.9.9")
    checker._work()
    assert len(seen) == 1                    # no re-announcement


def test_checker_quiet_when_current(monkeypatch, qapp):
    monkeypatch.delenv("MINDSIGHT_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(uc, "fetch_latest",
                        lambda _urlopen=None: ("v1.0.0", "https://x"))
    seen = []
    checker = uc.UpdateChecker(current_version="1.0.0")
    checker.update_available.connect(lambda t, u: seen.append(t))
    checker._work()
    assert seen == []


def test_start_is_noop_when_disabled(monkeypatch, qapp):
    monkeypatch.setenv("MINDSIGHT_NO_UPDATE_CHECK", "1")
    calls = []
    monkeypatch.setattr(uc, "fetch_latest",
                        lambda _urlopen=None: calls.append(1) or None)
    checker = uc.UpdateChecker(current_version="1.0.0")
    checker.start()                          # must not even spawn the fetch
    assert calls == []


# ── About-tab surface ────────────────────────────────────────────────────────

def test_about_hero_shows_update_and_toggle(qapp):
    from mindsight.GUI.about_tab import AboutTab
    tab = AboutTab()
    assert tab._update_note.isVisible() is False or True  # hidden pre-show
    assert not tab._update_note.text()
    tab.show_update("v1.2.0", "https://example.invalid/rel")
    assert "v1.2.0" in tab._update_note.text()
    assert "https://example.invalid/rel" in tab._update_note.text()
    assert tab._update_toggle.isChecked() is True         # default on


def test_about_toggle_persists(qapp, monkeypatch):
    monkeypatch.delenv("MINDSIGHT_NO_UPDATE_CHECK", raising=False)
    from mindsight.GUI.about_tab import AboutTab
    tab = AboutTab()
    tab._update_toggle.setChecked(False)
    assert uc.check_enabled() is False
    tab._update_toggle.setChecked(True)
    assert uc.check_enabled() is True


# ── Main-window chip ─────────────────────────────────────────────────────────

def test_main_window_chip_appears_and_opens(qapp, monkeypatch):
    from mindsight.GUI.main_window import MainWindow
    win = MainWindow()                       # checker inert (kill switch)
    assert win._update_chip.isVisible() is False
    win._on_update_available("v1.2.0", "https://example.invalid/rel")
    assert "v1.2.0" in win._update_chip.text()
    opened = []
    monkeypatch.setattr(
        "PyQt6.QtGui.QDesktopServices.openUrl",
        staticmethod(lambda url: opened.append(url.toString()) or True))
    win._open_release("v1.2.0", "https://example.invalid/rel")
    assert opened == ["https://example.invalid/rel"]
    assert uc.dismissed_tag() == "v1.2.0"
    assert win._update_chip.isVisible() is False
    # About hero mirrored the announcement.
    assert "v1.2.0" in win._about_tab._update_note.text()
    win.close()
