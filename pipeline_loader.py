"""
pipeline_loader.py — Load pipeline configuration from YAML files.

Bridges the gap between a declarative YAML pipeline file and the
argparse-based CLI interface.  The loader reads a YAML file, maps its
keys to argparse attribute names, and populates a namespace.  CLI flags
always take precedence over YAML values.

Usage
-----
    from pipeline_loader import load_pipeline

    # In _args(), after parse_args():
    ns = p.parse_args()
    if ns.pipeline:
        load_pipeline(ns.pipeline, ns)
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

# ── YAML key → argparse attribute mapping ────────────────────────────────────
# Nested YAML sections are flattened: 'detection.conf' → 'conf'.
# Keys not in this table are ignored (with a warning).

_YAML_MAP: dict[str, str] = {
    # Top-level
    'source':               'source',

    # Detection
    'detection.model':      'model',
    'detection.conf':       'conf',
    'detection.classes':    'classes',
    'detection.blacklist':  'blacklist',
    'detection.detect_scale': 'detect_scale',
    'detection.vp_file':    'vp_file',
    'detection.vp_model':   'vp_model',
    'detection.skip_frames': 'skip_frames',
    'detection.obj_persistence': 'obj_persistence',

    # Gaze
    'gaze.ray_length':     'ray_length',
    'gaze.adaptive_ray':   'adaptive_ray',
    'gaze.snap_dist':      'snap_dist',
    'gaze.snap_bbox_scale': 'snap_bbox_scale',
    'gaze.snap_w_dist':    'snap_w_dist',
    'gaze.snap_w_size':    'snap_w_size',
    'gaze.snap_w_intersect': 'snap_w_intersect',
    'gaze.conf_ray':       'conf_ray',
    'gaze.gaze_tips':      'gaze_tips',
    'gaze.tip_radius':     'tip_radius',
    'gaze.gaze_cone':      'gaze_cone',
    'gaze.gaze_lock':      'gaze_lock',
    'gaze.dwell_frames':   'dwell_frames',
    'gaze.lock_dist':      'lock_dist',
    'gaze.gaze_debug':     'gaze_debug',
    'gaze.snap_switch_frames': 'snap_switch_frames',
    'gaze.reid_grace_seconds': 'reid_grace_seconds',
    'gaze.hit_conf_gate':      'hit_conf_gate',
    'gaze.detect_extend':      'detect_extend',
    'gaze.detect_extend_scope': 'detect_extend_scope',

    # Output
    'output.save_video':       'save',
    'output.log_csv':          'log',
    'output.summary_csv':      'summary',
    'output.heatmaps':         'heatmap',
    'output.anonymize':        'anonymize',
    'output.anonymize_padding': 'anonymize_padding',

    # Participants
    'participants.csv':    'participant_csv',
    'participants.ids':    'participant_ids',

    # Performance
    'performance.fast':             'fast',
    'performance.skip_phenomena':   'skip_phenomena',
    'performance.lite_overlay':     'lite_overlay',
    'performance.no_dashboard':     'no_dashboard',
    'performance.profile':          'profile',

    # Joint attention (top-level phenomena keys)
    'phenomena.ja_window':       'ja_window',
    'phenomena.ja_window_thresh': 'ja_window_thresh',
    'phenomena.ja_quorum':       'ja_quorum',
}

# Phenomena tracker toggles — boolean flags
_PHENOMENA_TOGGLES: dict[str, str] = {
    'mutual_gaze':       'mutual_gaze',
    'social_referencing': 'social_ref',
    'gaze_following':    'gaze_follow',
    'gaze_aversion':     'gaze_aversion',
    'scanpath':          'scanpath',
    'gaze_leadership':   'gaze_leader',
    'attention_span':    'attn_span',
    'joint_attention':   'joint_attention',
}

# Phenomena per-tracker params
_PHENOMENA_PARAMS: dict[str, str] = {
    'ja_window':         'ja_window',
    'ja_quorum':         'ja_quorum',
    'ja_window_thresh':  'ja_window_thresh',
    'window':            'social_ref_window',
    'lag':               'gaze_follow_lag',
    'aversion_window':   'aversion_window',
    'aversion_conf':     'aversion_conf',
    'dwell':             'scanpath_dwell',
}


def _flatten(d: dict, prefix: str = '') -> dict:
    """Flatten a nested dict into dot-separated keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def load_pipeline(path: str | Path, ns: Namespace | None = None) -> Namespace:
    """
    Load a pipeline YAML config and populate an argparse Namespace.

    Parameters
    ----------
    path : str or Path
        Path to the YAML pipeline file.
    ns : Namespace, optional
        Existing namespace to update.  CLI-set values (non-default) are
        preserved; only defaults are overwritten by YAML values.
        If None, a new Namespace is created.

    Returns
    -------
    Namespace with YAML values merged in.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if ns is None:
        ns = Namespace()

    # 1. Process flat and nested sections (detection, gaze, output)
    flat = _flatten(cfg)
    for yaml_key, attr in _YAML_MAP.items():
        if yaml_key in flat:
            # Only set if the namespace attribute is at its default
            if not hasattr(ns, attr) or _is_default(ns, attr):
                setattr(ns, attr, flat[yaml_key])

    # 2. Process phenomena list (special format)
    phenomena_list = cfg.get('phenomena', [])
    if isinstance(phenomena_list, list):
        for item in phenomena_list:
            if isinstance(item, str):
                # Simple toggle: "- mutual_gaze"
                _set_phenomenon(ns, item, {})
            elif isinstance(item, dict):
                # Toggle with params: "- joint_attention: {ja_window: 30}"
                for name, params in item.items():
                    _set_phenomenon(ns, name, params or {})

    # 3. Process plugins section (pass-through to argparse attributes)
    #    Keys are mapped directly: hyphens → underscores.
    #    This supports ANY plugin flag without needing a hardcoded mapping.
    plugins_cfg = cfg.get('plugins', {})
    if isinstance(plugins_cfg, dict):
        for key, val in plugins_cfg.items():
            attr = key.replace('-', '_')
            if not hasattr(ns, attr) or _is_default(ns, attr):
                setattr(ns, attr, val)

    # 4. Process aux_streams section (optional per-participant video feeds)
    aux_list = cfg.get('aux_streams', [])
    if isinstance(aux_list, list) and aux_list:
        if not hasattr(ns, 'aux_streams') or _is_default(ns, 'aux_streams'):
            from pipeline_config import AuxStreamConfig
            parsed = []
            for item in aux_list:
                if isinstance(item, dict):
                    pid = item.get('pid', '')
                    stype = item.get('stream_type', '')
                    source = item.get('source', '')
                    if pid and stype and source:
                        parsed.append(AuxStreamConfig(pid=pid,
                                                      stream_type=stype,
                                                      source=source))
            if parsed:
                setattr(ns, 'aux_streams', parsed)

    # ── Backward compat: old adaptive_ray (bool) + adaptive_snap (bool) ──
    ar = getattr(ns, 'adaptive_ray', 'off')
    if isinstance(ar, bool):
        # Old YAML had adaptive_ray: true/false
        adaptive_snap = flat.get('gaze.adaptive_snap', False)
        if ar:
            setattr(ns, 'adaptive_ray', 'snap' if adaptive_snap else 'extend')
        else:
            setattr(ns, 'adaptive_ray', 'off')

    return ns


def _set_phenomenon(ns: Namespace, name: str, params: dict) -> None:
    """Enable a phenomenon tracker and set its parameters."""
    attr = _PHENOMENA_TOGGLES.get(name)
    if attr is None:
        return
    if not hasattr(ns, attr) or _is_default(ns, attr):
        setattr(ns, attr, True)

    for param_key, param_attr in _PHENOMENA_PARAMS.items():
        if param_key in params:
            if not hasattr(ns, param_attr) or _is_default(ns, param_attr):
                setattr(ns, param_attr, params[param_key])


def _is_default(ns: Namespace, attr: str) -> bool:
    """Heuristic: a value is 'default' if it matches common defaults.

    This is a best-effort check since argparse doesn't track which values
    were explicitly set by the user.  CLI flags that differ from defaults
    are considered user-set and will not be overwritten.
    """
    val = getattr(ns, attr, None)
    # None, False, 0, 0.0, empty list/string are considered default-like
    if val is None or val is False or val == 0 or val == 0.0:
        return True
    if isinstance(val, (list, str)) and len(val) == 0:
        return True
    return False
