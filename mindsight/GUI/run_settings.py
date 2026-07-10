"""
GUI/run_settings.py -- the RunSettings store (UP2 Batch A).

The single source of inference configuration for every run launched from
Analyze Footage (project batch, per-row re-run, manual dialog, quick video,
camera).  Decoupling ruling (2026-07-10): this store -- NOT the Gaze Tuning
tab -- drives those runs.  Gaze Tuning keeps its own namespace; the only
bridge is explicit "Import from Gaze Tuning".

State is a fully-resolved argparse ``Namespace`` (``parse_cli([])`` defaults
with an optional YAML/preset overlay), so it carries every core + plugin dest
a run needs.  Construction builds the parser and reads YAML only -- it loads
no models or weights.

Seeding order: ``~/.mindsight/run_settings.json`` (a persisted prior state) ->
else the shipped KNOWN_GOOD preset -> else pure parser defaults.

Weight dests are stored PORTABLE: on save, any ``model`` / ``vp_model`` /
``mgaze_model`` / ``rf_gazelle_model`` / ``gazelle_model`` value that points
inside the shared Weights root is reduced to its bare filename, so a persisted
state never pins this machine's absolute paths onto a fresh install.
"""
from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from mindsight.cli_flags import parse_cli
from mindsight.config_compat import known_good_preset_path, load_pipeline

from .settings_manager import SettingsManager

# Weight dests reduced to portable bare names on persist (standing rule: no
# absolute weight paths persisted; resolution is global and device-family logic
# handles bare names).
_WEIGHT_DESTS = ("model", "vp_model", "mgaze_model",
                 "rf_gazelle_model", "gazelle_model")

# Output artifact toggles (Q7): store-level booleans.  Default = produce, so a
# freshly seeded store reproduces today's behavior until a user unticks one.
_OUTPUT_TOGGLES = ("save", "heatmap", "charts")


def _weights_root() -> Path:
    """The shared Weights root, read at call time so a relocated install and
    tests that repoint ``constants.PROJECT_ROOT`` are both honored."""
    from mindsight import constants
    return constants.PROJECT_ROOT / "Weights"


def _portable_weight(value):
    """Reduce an absolute weight path under the shared Weights root to a bare
    filename; leave bare names, family names, and foreign absolute paths as-is.

    Guards the "absolute default trap": ``mgaze_model``'s parser default is an
    ABSOLUTE path baked at parse time -- persisted verbatim it would follow this
    machine onto a fresh install.
    """
    if not isinstance(value, str) or not value:
        return value
    p = Path(value)
    if not p.is_absolute():
        return value
    try:
        p.relative_to(_weights_root())
    except ValueError:
        return value            # foreign absolute path -- preserve it
    return p.name               # under the shared Weights root -> bare name


def want_artifact(ns, key: str) -> bool:
    """True when artifact *key* (save/heatmap/charts) should be produced.

    The store's toggles ride on the launched namespace as booleans; only an
    explicit ``False`` suppresses.  Absent / ``None`` / truthy -> produce, which
    is today's default.  Shared by the one-off launch and the project batch so
    both apply the same mapping (A3).
    """
    return getattr(ns, key, True) is not False


class RunSettingsStore(QObject):
    """Holds the inference namespace that Analyze Footage launches from."""

    changed = pyqtSignal()          # any mutation (commit / apply_yaml / reset)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_label = "defaults"
        self._ns = self._seed()
        self._snapshot = self._signature(self._ns)

    # -- construction / seeding ----------------------------------------------

    @staticmethod
    def _path() -> Path:
        """Persistence path, keyed off ``SettingsManager.SETTINGS_DIR`` at call
        time so the test fake-HOME fixture isolates it automatically."""
        return SettingsManager.SETTINGS_DIR / "run_settings.json"

    @staticmethod
    def _coerce_toggles(ns) -> None:
        """Default unset output toggles to ON (produce), the preset behavior."""
        for key in _OUTPUT_TOGGLES:
            if getattr(ns, key, None) is None:
                setattr(ns, key, True)

    def _seed(self) -> Namespace:
        base = parse_cli([])        # full defaults incl. plugin dests; no models
        path = self._path()

        # 1. Persisted prior state.
        if path.exists():
            try:
                d = json.loads(path.read_text())
                self._source_label = d.pop("_source_label", "custom")
                d.pop("_version", None)
                vars(base).update(d)
                self._coerce_toggles(base)
                return base
            except Exception as exc:  # noqa: BLE001 -- bad file must not crash startup
                print(f"[WARN] could not read run settings "
                      f"({path.name}): {exc}; falling back to preset")
                base = parse_cli([])  # discard a half-applied update

        # 2. Shipped KNOWN_GOOD preset.
        preset = known_good_preset_path()
        if preset is not None:
            try:
                load_pipeline(str(preset), base)
                self._source_label = "KG_Standard"
                self._coerce_toggles(base)
                return base
            except Exception as exc:  # noqa: BLE001 -- reset must survive a bad preset
                print(f"[WARN] could not seed run settings from preset: {exc}")
                base = parse_cli([])

        # 3. Pure parser defaults.
        self._source_label = "defaults"
        self._coerce_toggles(base)
        return base

    # -- normalization / persistence -----------------------------------------

    @staticmethod
    def _normalized_dict(ns) -> dict:
        """JSON-safe dict with weight dests reduced to portable bare names."""
        d = SettingsManager._ns_to_dict(ns)
        for dest in _WEIGHT_DESTS:
            if dest in d:
                d[dest] = _portable_weight(d[dest])
        d.pop("_explicit_cli", None)   # never a setting
        return d

    def _signature(self, ns) -> str:
        """Stable signature for modified-tracking.

        ``PipelineConfig.canonical_hash`` covers only schema fields, so it MISSES
        model-wiring and plugin dests (``mgaze_model``, ``rf_gazelle_model``,
        blend betas, ...).  Per A1's fallback clause we instead compare the full
        normalized persisted dict, which captures every dest uniformly.
        """
        d = self._normalized_dict(ns)
        return json.dumps(d, sort_keys=True)

    def _persist(self) -> None:
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            d = self._normalized_dict(self._ns)
            d["_source_label"] = self._source_label
            path.write_text(json.dumps(d, indent=2))
        except Exception as exc:  # noqa: BLE001 -- persistence must never break the GUI
            print(f"[WARN] could not persist run settings: {exc}")

    # -- public API ----------------------------------------------------------

    def ns(self) -> Namespace:
        """DEEP COPY of the state for launching runs / preflight."""
        return copy.deepcopy(self._ns)

    def working_copy(self) -> Namespace:
        """DEEP COPY for the dialog to edit (committed back via ``commit``)."""
        return copy.deepcopy(self._ns)

    def commit(self, ns: Namespace) -> None:
        """Replace state (dialog OK/Apply), persist, emit.  Does NOT reset the
        source snapshot, so ``is_modified`` reflects the edit."""
        self._ns = copy.deepcopy(ns)
        self._coerce_toggles(self._ns)
        self._persist()
        self.changed.emit()

    def apply_yaml(self, path, source_label: str = "custom") -> None:
        """Overlay a pipeline YAML onto fresh parser defaults (preset / project
        pipeline / import).  Resets the source snapshot (loaded state is the new
        baseline), persists, emits."""
        base = parse_cli([])
        load_pipeline(str(path), base)
        self._coerce_toggles(base)
        self._ns = base
        self._source_label = source_label
        self._snapshot = self._signature(self._ns)
        self._persist()
        self.changed.emit()

    def reset_to_preset(self) -> None:
        """Reseed from the shipped KNOWN_GOOD preset (or defaults if absent)."""
        preset = known_good_preset_path()
        base = parse_cli([])
        if preset is not None:
            try:
                load_pipeline(str(preset), base)
                self._source_label = "KG_Standard"
            except Exception as exc:  # noqa: BLE001 -- reset must survive a bad preset
                print(f"[WARN] could not reset to preset: {exc}")
                self._source_label = "defaults"
        else:
            self._source_label = "defaults"
        self._coerce_toggles(base)
        self._ns = base
        self._snapshot = self._signature(self._ns)
        self._persist()
        self.changed.emit()

    def source_label(self) -> str:
        """The active source: "KG_Standard", "project pipeline", "custom", ..."""
        return self._source_label

    def is_modified(self) -> bool:
        """True when the current state differs from the last loaded source."""
        return self._signature(self._ns) != self._snapshot

    def output_toggles(self) -> dict:
        """The three artifact toggles as booleans (convenience for callers)."""
        return {key: want_artifact(self._ns, key) for key in _OUTPUT_TOGGLES}
