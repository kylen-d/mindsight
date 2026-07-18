"""GUI/update_check.py -- silent launch-time update notification (W3Y item 7).

Once per launch, a daemon thread asks the GitHub Releases API for the
latest release and compares it against ``mindsight.__version__``.  On a
newer version it emits ``update_available(tag, url)`` -- the main window
shows a subtle status-bar chip and the About hero a matching line; a
click opens the release page in the browser (v1 deliberately downloads
nothing: macOS quarantine makes the browser strictly better, and
auto-executing downloads is off the table).

Failure policy: ANY problem (offline, rate-limited, bad JSON, timeout)
is a silent no-op.  Opt-outs: the "Check for updates on launch" toggle
(gui_state ``check_updates``, default on) and the
``MINDSIGHT_NO_UPDATE_CHECK=1`` environment kill switch for frozen lab
machines.  A release the user already opened is not re-announced
(gui_state ``dismissed_release``).
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

from PyQt6.QtCore import QObject, pyqtSignal

RELEASES_API = "https://api.github.com/repos/kylen-d/MindSight/releases/latest"
RELEASES_PAGE = "https://github.com/kylen-d/MindSight/releases"
_TIMEOUT_S = 5.0


def parse_version(text: str) -> tuple:
    """Lenient version parse: 'v1.2.3-rc1' -> (1, 2, 3). Empty on garbage."""
    nums = []
    for part in str(text).strip().lstrip("vV").split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums)


def is_newer(remote_tag: str, current: str) -> bool:
    r, c = parse_version(remote_tag), parse_version(current)
    return bool(r) and bool(c) and r > c


def check_enabled() -> bool:
    if os.environ.get("MINDSIGHT_NO_UPDATE_CHECK"):
        return False
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        return bool(SettingsManager().load_gui_state()
                    .get("check_updates", True))
    except Exception:
        return True


def set_check_enabled(on: bool) -> None:
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        SettingsManager().save_gui_state({"check_updates": bool(on)})
    except Exception:
        pass


def dismissed_tag() -> str:
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        return str(SettingsManager().load_gui_state()
                   .get("dismissed_release", ""))
    except Exception:
        return ""


def dismiss(tag: str) -> None:
    """Record that the user opened this release -- never announce it again."""
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        SettingsManager().save_gui_state({"dismissed_release": str(tag)})
    except Exception:
        pass


def fetch_latest(_urlopen=None) -> tuple[str, str] | None:
    """(tag, release-page url) from the GitHub API, or None on ANY failure.

    Network call -- run off the GUI thread.  Anonymous access (60 req/h)
    is ample for once per launch.
    """
    try:
        from mindsight import __version__
        req = urllib.request.Request(RELEASES_API, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"MindSight/{__version__}",
        })
        opener = _urlopen if _urlopen is not None else urllib.request.urlopen
        with opener(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tag = str(data.get("tag_name") or "")
        url = str(data.get("html_url") or RELEASES_PAGE)
        return (tag, url) if tag else None
    except Exception:
        return None


class UpdateChecker(QObject):
    """Launch-time checker; emits ``update_available(tag, url)`` at most once.

    The worker runs on a daemon thread; the cross-thread signal emit is
    delivered queued on the GUI thread, so slots can touch widgets.
    """

    update_available = pyqtSignal(str, str)

    def __init__(self, current_version: str | None = None, parent=None):
        super().__init__(parent)
        if current_version is None:
            from mindsight import __version__
            current_version = __version__
        self._current = current_version

    def start(self) -> None:
        if not check_enabled():
            return
        threading.Thread(target=self._work, daemon=True,
                         name="mindsight-update-check").start()

    def _work(self) -> None:
        res = fetch_latest()
        if res is None:
            return
        tag, url = res
        if not is_newer(tag, self._current):
            return
        if tag == dismissed_tag():
            return
        self.update_available.emit(tag, url)
