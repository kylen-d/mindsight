"""
GUI/pipeline_dialog.py — Pipeline YAML import and export for the MindSight GUI.

Provides two functions:
  - import_pipeline(parent) -> Namespace | None
  - export_pipeline(parent, ns)
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QFileDialog, QMessageBox


def import_pipeline(parent) -> Namespace | None:
    """Show a file dialog to select a YAML pipeline file, load it, and return a Namespace.

    Returns None if the user cancels or an error occurs.
    """
    path, _ = QFileDialog.getOpenFileName(
        parent, "Import Pipeline Configuration", "",
        "YAML (*.yaml *.yml);;All (*)")
    if not path:
        return None

    try:
        from mindsight.config_compat import load_pipeline
        ns = load_pipeline(path, Namespace())
        QMessageBox.information(
            parent, "Pipeline Imported",
            f"Loaded pipeline configuration from:\n{Path(path).name}")
        return ns
    except Exception as e:
        QMessageBox.critical(
            parent, "Import Error",
            f"Failed to import pipeline:\n{e}")
        return None


def export_pipeline(parent, ns: Namespace) -> bool:
    """Show a file dialog and export the current settings as a YAML pipeline file.

    Returns True if the file was saved successfully.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, "Export Pipeline Configuration", "",
        "YAML (*.yaml *.yml);;All (*)")
    if not path:
        return False
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"

    try:
        cfg = _namespace_to_yaml_dict(ns)
        Path(path).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        QMessageBox.information(
            parent, "Pipeline Exported",
            f"Saved pipeline configuration to:\n{Path(path).name}")
        return True
    except Exception as e:
        QMessageBox.critical(
            parent, "Export Error",
            f"Failed to export pipeline:\n{e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Schema-driven export (SP3.1 Step 17)
# ══════════════════════════════════════════════════════════════════════════════
#
# The export is derived from the CANONICAL config-compat tables -- ``_YAML_MAP``
# (yaml key <-> dest), ``_PHENOMENA_TOGGLES`` / ``_PHENOMENA_PARAMS`` (phenomena
# list names + param keys) -- and the schema field defaults (via the ui-spec),
# so there is no hand-written toggle/param map to drift.  Only-non-default keys
# are emitted; the loader's baseline is the pipeline (schema/parser) default, so
# a GUI namespace re-imports (load_pipeline -> apply_namespace) census-equal.

# Params that live in the phenomena list, grouped under their owning toggle.
# The NAMES come from config-compat (_PHENOMENA_PARAMS); only the ownership
# grouping lives here (the phenomena tracker each param configures).
_PHENOMENA_PARAM_OWNER: dict[str, str] = {
    "ja_window": "joint_attention",
    "ja_quorum": "joint_attention",
    "ja_window_thresh": "joint_attention",
    "social_ref_window": "social_ref",
    "gaze_follow_lag": "gaze_follow",
    "aversion_window": "gaze_aversion",
    "aversion_conf": "gaze_aversion",
    "scanpath_dwell": "scanpath",
}

# Model-wiring dests (gaze backends) with no schema home -- passed through the
# ``plugins`` section verbatim (hyphens<->underscores) by the loader.
_MODEL_WIRING_DESTS = (
    "mgaze_model", "mgaze_arch", "mgaze_dataset",
    "gazelle_model", "gazelle_name", "gazelle_inout_threshold",
    "gazelle_device", "gazelle_skip_frames", "gazelle_fp16", "gazelle_compile",
)


def _norm(v):
    """Normalize a value for a default comparison (sets -> sorted lists)."""
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def _export_baseline() -> dict:
    """Per-dest pipeline default (what an omitted YAML key means on re-import).

    Parser defaults, overridden by the schema field default for every
    schema-backed ui field -- so a GUI namespace whose schema knobs sit at their
    pipeline default exports nothing for them, while GUI-only run-loop defaults
    (e.g. no_dashboard on) still export as deliberate overrides.
    """
    from mindsight.cli_flags import build_parser
    from mindsight.GUI.ui_spec import build_ui_spec, iter_fields
    base = dict(vars(build_parser().parse_args([])))
    for f in iter_fields(build_ui_spec()):
        if f.schema_path is not None:
            base[f.dest] = f.default
    return base


def _plugin_export_dests() -> set:
    """Schema knobs with no ``_YAML_MAP`` key + the model-wiring dests: they
    round-trip through the loader's ``plugins`` pass-through section."""
    from mindsight.config_compat import (
        _PHENOMENA_PARAMS,
        _PHENOMENA_TOGGLES,
        _YAML_MAP,
    )
    from mindsight.GUI.ui_spec import all_dests, build_ui_spec
    yaml_dests = set(_YAML_MAP.values())
    phen_dests = set(_PHENOMENA_TOGGLES.values()) | set(_PHENOMENA_PARAMS.values())
    ui_only = all_dests(build_ui_spec()) - yaml_dests - phen_dests
    return ui_only | set(_MODEL_WIRING_DESTS)


def _set_nested(cfg: dict, yaml_key: str, value) -> None:
    """Place *value* at a (possibly dotted) YAML key inside *cfg*."""
    if "." not in yaml_key:
        cfg[yaml_key] = value
        return
    section, field = yaml_key.split(".", 1)
    cfg.setdefault(section, {})[field] = value


def _namespace_to_yaml_dict(ns: Namespace) -> dict:
    """Convert a Namespace to a structured pipeline-YAML dict (schema-driven).

    Emits only keys whose value differs from the pipeline default, in the layout
    ``load_pipeline`` reads back: scalar detection/gaze/output/performance keys
    via the canonical ``_YAML_MAP``, the phenomena list via the canonical
    phenomena tables, and the model-wiring + no-YAML-key schema knobs under
    ``plugins``.  ``load_pipeline(export(ns))`` re-imports census-equal.
    """
    from mindsight.config_compat import (
        _PHENOMENA_PARAMS,
        _PHENOMENA_TOGGLES,
        _YAML_MAP,
    )

    d = vars(ns) if hasattr(ns, "__dict__") else {}
    base = _export_baseline()
    cfg: dict = {}

    # 1. Scalar sections via the canonical yaml-key <-> dest table.  Phenomena
    #    (list format) and the interval knob (plugins) are handled below.
    for yaml_key, dest in _YAML_MAP.items():
        if yaml_key.startswith("phenomena."):
            continue
        if dest not in d:
            continue
        val = _norm(d.get(dest))
        if val == _norm(base.get(dest)):
            continue
        if val in (None, "", [], {}):
            continue
        _set_nested(cfg, yaml_key, val)

    # 2. Phenomena list: canonical yaml names + only-non-default params.
    key_by_dest = {dest: key for key, dest in _PHENOMENA_PARAMS.items()}
    phenomena: list = []
    for yaml_name, toggle_dest in _PHENOMENA_TOGGLES.items():
        if not d.get(toggle_dest):
            continue
        params: dict = {}
        for pdest, owner in _PHENOMENA_PARAM_OWNER.items():
            if owner != toggle_dest or pdest not in d:
                continue
            val = d.get(pdest)
            if val is not None and _norm(val) != _norm(base.get(pdest)):
                params[key_by_dest[pdest]] = val
        phenomena.append({yaml_name: params} if params else yaml_name)
    if phenomena:
        cfg["phenomena"] = phenomena

    # 3. Plugins section: model wiring + schema knobs with no YAML key.  The
    #    interval knob keeps its ``min_call_gap`` spelling (loader precedence).
    plugins: dict = {}
    for dest in sorted(_plugin_export_dests()):
        if dest not in d:
            continue
        val = _norm(d.get(dest))
        if val is None or val == _norm(base.get(dest)):
            continue
        plugins[dest] = val
    if plugins:
        cfg["plugins"] = plugins

    return cfg
