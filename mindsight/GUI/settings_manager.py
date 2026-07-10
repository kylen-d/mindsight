"""
GUI/settings_manager.py — Save and load user presets and last-used settings.

Stores settings as JSON files in ~/.mindsight/. Presets are named files in
~/.mindsight/presets/. The last-used session is auto-saved on close and
auto-restored on next launch.
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path


def _is_aux_stream(x) -> bool:
    """True when *x* looks like an ``AuxStreamConfig`` (duck-typed to avoid a
    hard import of pipeline_config in this lightweight settings module)."""
    return all(hasattr(x, attr) for attr in
               ("source", "video_type", "stream_label", "participants",
                "auto_detect_faces"))


def _aux_stream_to_dict(a) -> dict:
    """Serialize one ``AuxStreamConfig`` to a JSON-safe dict (video_type as its
    plain string value; the enum reconstructs on restore)."""
    vtype = getattr(a.video_type, "value", a.video_type)
    return {
        "source": a.source,
        "video_type": str(vtype),
        "stream_label": a.stream_label,
        "participants": (list(a.participants)
                         if a.participants is not None else None),
        "auto_detect_faces": bool(a.auto_detect_faces),
    }


class SettingsManager:
    """Manages persistent user settings (presets and last-used session)."""

    SETTINGS_DIR = Path.home() / ".mindsight"
    LAST_USED = SETTINGS_DIR / "last_used.json"
    PRESETS_DIR = SETTINGS_DIR / "presets"
    RECENT_PROJECTS = SETTINGS_DIR / "recent_projects.json"

    RECENT_LIMIT = 10

    def __init__(self):
        self.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    # ── GUI state (window/tab prefs -- NOT inference settings) ────────────
    # Path derived from SETTINGS_DIR at call time so test fixtures that
    # repoint the settings dir isolate this file too.

    def load_gui_state(self) -> dict:
        """Small GUI preferences (e.g. the Analyze Footage input mode)."""
        try:
            return json.loads(
                (self.SETTINGS_DIR / "gui_state.json").read_text())
        except Exception:
            return {}

    def save_gui_state(self, updates: dict) -> None:
        state = self.load_gui_state()
        state.update(updates)
        try:
            (self.SETTINGS_DIR / "gui_state.json").write_text(
                json.dumps(state, indent=2))
        except Exception:  # pragma: no cover - best-effort persistence
            pass

    # ── Serialization helpers ─────────────────────────────────────────────

    @staticmethod
    def _ns_to_dict(ns: Namespace) -> dict:
        """Convert a Namespace to a JSON-serializable dict."""
        d = {}
        for k, v in vars(ns).items():
            # Skip non-serializable values
            if v is None or isinstance(v, (str, int, float, bool)):
                d[k] = v
            elif isinstance(v, (list, tuple)):
                # Lists of primitives round-trip as-is.
                if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                    d[k] = list(v)
                # AuxStreamConfig lists (the Auxiliary Streams table) serialize
                # to plain dicts so they survive a save/restore -- silently
                # dropping them left the table empty on every relaunch.
                elif k == "aux_streams" and all(_is_aux_stream(x) for x in v):
                    d[k] = [_aux_stream_to_dict(x) for x in v]
            elif isinstance(v, set):
                d[k] = sorted(v)
        d["_version"] = 1
        return d

    @staticmethod
    def _dict_to_ns(d: dict) -> Namespace:
        """Convert a dict back to a Namespace."""
        d = dict(d)
        d.pop("_version", None)
        return Namespace(**d)

    # ── Presets ───────────────────────────────────────────────────────────

    def save_preset(self, name: str, ns: Namespace) -> Path:
        """Save a named preset. Returns the file path."""
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        path = self.PRESETS_DIR / f"{safe_name}.json"
        path.write_text(json.dumps(self._ns_to_dict(ns), indent=2))
        return path

    def load_preset(self, name: str) -> Namespace:
        """Load a named preset."""
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        path = self.PRESETS_DIR / f"{safe_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")
        return self._dict_to_ns(json.loads(path.read_text()))

    def list_presets(self) -> list[str]:
        """Return sorted list of preset names."""
        return sorted(p.stem for p in self.PRESETS_DIR.glob("*.json"))

    def delete_preset(self, name: str):
        """Delete a named preset."""
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        path = self.PRESETS_DIR / f"{safe_name}.json"
        if path.exists():
            path.unlink()

    # ── Last-used session ─────────────────────────────────────────────────

    def save_last_used(self, ns: Namespace):
        """Save the current session as last-used (auto-restored on next launch)."""
        self.LAST_USED.write_text(json.dumps(self._ns_to_dict(ns), indent=2))

    def load_last_used(self) -> Namespace | None:
        """Load the last-used session, or None if not available."""
        if not self.LAST_USED.exists():
            return None
        try:
            return self._dict_to_ns(json.loads(self.LAST_USED.read_text()))
        except Exception as exc:  # noqa: BLE001 -- a bad file must not kill startup
            print(f"[WARN] could not read last session "
                  f"({self.LAST_USED.name}): {exc}")
            return None

    # ── Recent projects (D12) ─────────────────────────────────────────────

    def list_recent_projects(self) -> list[str]:
        """Return the most-recently-opened project paths (newest first)."""
        if not self.RECENT_PROJECTS.exists():
            return []
        try:
            data = json.loads(self.RECENT_PROJECTS.read_text())
        except (json.JSONDecodeError, Exception):
            return []
        items = data.get("projects", []) if isinstance(data, dict) else []
        return [str(p) for p in items if isinstance(p, str)]

    def add_recent_project(self, path: str) -> list[str]:
        """Record *path* as the most-recently-opened project (dedup, newest first).

        Returns the updated recent list (capped at ``RECENT_LIMIT``).
        """
        path = str(path)
        recent = [p for p in self.list_recent_projects() if p != path]
        recent.insert(0, path)
        recent = recent[: self.RECENT_LIMIT]
        self.RECENT_PROJECTS.write_text(
            json.dumps({"_version": 1, "projects": recent}, indent=2))
        return recent


def checkpoint(ns: Namespace) -> None:
    """Save *ns* as the last-used session, WARNING (never raising) on failure.

    A run start is the natural checkpoint of a configuration worth keeping:
    ``MainWindow.closeEvent`` is otherwise the ONLY writer, so a crash or
    force-quit mid-run loses the whole session. Shared by the Gaze tab Start
    button and the Analyze Footage run starts so there is one guarded writer,
    not three copies of the try/except.
    """
    try:
        SettingsManager().save_last_used(ns)
    except Exception as exc:  # noqa: BLE001 -- a checkpoint must never break a run
        print(f"[WARN] could not checkpoint session: {exc}")
