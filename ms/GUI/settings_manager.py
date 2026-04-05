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


class SettingsManager:
    """Manages persistent user settings (presets and last-used session)."""

    SETTINGS_DIR = Path.home() / ".mindsight"
    LAST_USED = SETTINGS_DIR / "last_used.json"
    PRESETS_DIR = SETTINGS_DIR / "presets"

    def __init__(self):
        self.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.PRESETS_DIR.mkdir(parents=True, exist_ok=True)

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
                # Only keep lists of primitives
                if all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
                    d[k] = list(v)
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
        except (json.JSONDecodeError, Exception):
            return None
