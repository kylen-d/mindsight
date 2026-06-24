"""
ui_spec.py -- Pure, headless builder for the schema-driven settings surface
(SP3.1 Batch F, D7).

``build_ui_spec()`` resolves the ``"ui"`` metadata attached to every
``mindsight.config`` field (D6) into an ordered tree of ``UiGroup`` /
``UiField`` records: widget type, dest (argparse name), harvested range/step/
decimals, tooltip (pulled from the ``cli_flags`` FlagSpec help table -- single
source), advanced tier, and checkable-group toggle semantics (T10 off-values).

A handful of knobs the hand widgets render have NO schema home -- the Gaze-LLE
model picker/variant (model wiring), the Performance run-loop toggles, and the
phenomena master switch.  They live in the small explicit ``_EXCLUDED_FIELDS``
table below (they are ``EXCLUDED_CLI_FLAGS`` by design) and are slotted into
their groups here.

This module imports NO Qt -- ``mindsight.GUI.schema_panel`` renders it.  It is
fully testable headless (tests/test_ui_spec.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dcfield

from mindsight.cli_flags import CORE_FLAGS
from mindsight.config import PipelineConfig
from mindsight.config_compat import CLI_ALIASES


# ══════════════════════════════════════════════════════════════════════════════
# Records
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UiField:
    """One rendered control.  ``schema_path`` None = an excluded flag with no
    schema home (fully specified here)."""
    dest: str
    schema_path: str | None
    widget: str                       # 'spin' | 'double' | 'check' | 'combo' | 'line'
    default: object
    label: str
    tooltip: str = ""
    advanced: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int | None = None
    choices: tuple | None = None
    inverted: bool = False            # checkbox whose dest value = not checked


@dataclass(frozen=True)
class UiGroup:
    """A settings group.  When ``toggle_dest`` is set the group renders as a
    checkable QGroupBox whose checked state drives that dest; unchecking writes
    ``toggle_off_value`` (T10).  ``toggle_owner_widget`` None means the checkbox
    IS the owner (a bool dest); 'combo'/'line' means the owner has an inner
    control shown only while checked."""
    key: str
    title: str
    fields: tuple[UiField, ...] = ()
    subgroups: tuple["UiGroup", ...] = ()
    advanced: bool = False
    toggle_dest: str | None = None
    toggle_off_value: object = None
    toggle_owner_widget: str | None = None
    toggle_on_default: object = None
    toggle_choices: tuple | None = None
    toggle_label: str = ""
    toggle_tooltip: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Excluded (non-schema) fields the hand widgets render -- explicit by design
# ══════════════════════════════════════════════════════════════════════════════
#
# dest -> partial UiField kwargs.  Tooltip is pulled from the FlagSpec help at
# build time (like schema fields).  These are all EXCLUDED_CLI_FLAGS.

_EXCLUDED_FIELDS: dict[str, dict] = {
    # Gaze-LLE blend model wiring (rendered inside the blend group).
    "rf_gazelle_model": dict(group="gazelle_blend", widget="line", default=None,
                             label="Model"),
    "rf_gazelle_name": dict(
        group="gazelle_blend", widget="combo", default="gazelle_dinov2_vitb14",
        label="Variant",
        choices=("gazelle_dinov2_vitb14", "gazelle_dinov2_vitb14_inout",
                 "gazelle_dinov2_vitl14", "gazelle_dinov2_vitl14_inout")),
    # Performance run-loop toggles.
    "fast": dict(group="performance", widget="check", default=False,
                 label="Fast mode (bundle perf optimizations)"),
    "skip_phenomena": dict(group="performance", widget="spin", default=0,
                           label="Skip phenomena", minimum=0, maximum=30),
    "lite_overlay": dict(group="performance", widget="check", default=False,
                         label="Lite overlay (minimal drawing)"),
    # Inverted: the checkbox reads "include dashboard"; dest = not checked.
    "no_dashboard": dict(group="performance", widget="check", default=True,
                         label="Include dashboard in video output", inverted=True),
    "profile": dict(group="performance", widget="check", default=False,
                    label="Profile (per-stage timing)"),
    # Phenomena master switch.
    "all_phenomena": dict(group="phenomena", widget="check", default=False,
                          label="Enable all phenomena trackers"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Group tree (order + nesting + toggle semantics)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry: (key, title).  ``fields`` is the ORDERED list of dests slotted
# into the group.  ``toggle`` (when present) names the owner dest.  ``sub`` are
# nested groups.  The gaze-tuning surface reproduces the ray + performance +
# phenomena hand sections exactly (Batch F equivalence gate).

@dataclass
class _GroupSpec:
    key: str
    title: str
    dests: list[str]
    toggle: str | None = None
    sub: list["_GroupSpec"] = _dcfield(default_factory=list)


_GROUP_TREE: list[_GroupSpec] = [
    _GroupSpec("ray_geometry", "Gaze Ray Geometry",
               ["ray_length", "conf_ray", "gaze_cone", "forward_gaze_threshold"],
               sub=[_GroupSpec("gaze_tips", "Gaze tips (virtual objects)",
                               ["tip_radius"], toggle="gaze_tips")]),
    _GroupSpec("gazelle_blend", "Gaze-LLE Blend (Ray Forming)",
               ["rf_gazelle_name", "min_call_gap", "dir_beta", "len_beta",
                "len_hold_tau", "fixation_v_threshold", "fixation_d_threshold",
                "dir_min_cutoff", "len_min_cutoff"],
               toggle="rf_gazelle_model"),
    _GroupSpec("adaptive_snap", "Adaptive Snap",
               ["snap_dist", "snap_bbox_scale", "snap_w_dist", "snap_w_angle",
                "snap_w_size", "snap_w_intersect", "snap_w_temporal",
                "snap_gate_angle", "snap_head_blend", "snap_quality_thresh",
                "snap_release_frames", "snap_engage_frames", "snap_tip_dist",
                "snap_tip_quality"],
               toggle="adaptive_ray"),
    _GroupSpec("smoothing", "Ray Forming Smoothing",
               ["smooth_snap_alpha"], toggle="smooth_snap"),
    _GroupSpec("fixation", "Fixation Lock-On",
               ["dwell_frames", "lock_dist"], toggle="gaze_lock"),
    _GroupSpec("hit_detection", "Hit Detection",
               ["hit_conf_gate", "detect_extend", "detect_extend_scope"]),
    _GroupSpec("depth", "Depth Estimation",
               ["depth_backend", "depth_input_size", "depth_skip_frames",
                "depth_aware_scoring", "depth_w_depth"],
               toggle="depth"),
    _GroupSpec("performance", "Performance && Tracking",
               ["skip_frames", "detect_scale", "reid_grace_seconds",
                "obj_persistence", "gaze_debug", "fast", "skip_phenomena",
                "lite_overlay", "no_dashboard", "profile"]),
    _GroupSpec("phenomena", "Phenomena Tracking",
               ["all_phenomena", "mutual_gaze", "social_ref",
                "social_ref_window", "gaze_follow", "gaze_follow_lag",
                "scanpath", "scanpath_dwell", "gaze_leader", "gaze_leader_tips",
                "gaze_leader_tip_lag", "attn_span"],
               sub=[
                   _GroupSpec("ja", "Joint Attention",
                              ["ja_quorum", "ja_window", "ja_window_thresh"],
                              toggle="joint_attention"),
                   _GroupSpec("aversion", "Gaze Aversion",
                              ["aversion_window", "aversion_conf"],
                              toggle="gaze_aversion"),
               ]),
]


# ══════════════════════════════════════════════════════════════════════════════
# Resolution helpers
# ══════════════════════════════════════════════════════════════════════════════

def _flag_tables() -> tuple[dict[str, str], dict[str, str | None]]:
    """(dest_by_flag, help_by_flag) from the FlagSpec table."""
    dest_by_flag: dict[str, str] = {}
    help_by_flag: dict[str, str | None] = {}
    for spec in CORE_FLAGS:
        dest_by_flag[spec.flag] = spec.dest
        help_by_flag[spec.flag] = spec.help
    return dest_by_flag, help_by_flag


def _schema_index() -> dict[str, dict]:
    """dest -> resolved schema field record (path, ui, field default, flag)."""
    dest_by_flag, help_by_flag = _flag_tables()
    alias_rev = {path: flag for flag, path in CLI_ALIASES.items()}
    out: dict[str, dict] = {}
    for section, section_field in PipelineConfig.model_fields.items():
        for fname, fi in section_field.annotation.model_fields.items():
            extra = fi.json_schema_extra or {}
            ui = extra.get("ui")
            if ui is None:
                continue
            path = f"{section}.{fname}"
            flag = extra.get("cli") or alias_rev.get(path)
            assert flag, f"{path} is ui:dict but resolves to no flag"
            dest = dest_by_flag[flag]
            out[dest] = {
                "path": path,
                "ui": ui,
                "flag": flag,
                "tooltip": help_by_flag.get(flag) or "",
                "annotation": fi.annotation,
                "default": fi.get_default(call_default_factory=True),
            }
    return out


def _widget_for(annotation, choices) -> str:
    if annotation is bool:
        return "check"
    if annotation is int:
        return "spin"
    if annotation is float:
        return "double"
    if choices:
        return "combo"
    return "line"


def _build_schema_field(dest: str, rec: dict) -> UiField:
    ui = rec["ui"]
    choices = tuple(ui["choices"]) if ui.get("choices") else None
    widget = _widget_for(rec["annotation"], choices)
    default = ui["default"] if "default" in ui else rec["default"]
    return UiField(
        dest=dest,
        schema_path=rec["path"],
        widget=widget,
        default=default,
        label=ui.get("label", dest),
        tooltip=rec["tooltip"],
        advanced=bool(ui.get("advanced", False)),
        minimum=ui.get("min"),
        maximum=ui.get("max"),
        step=ui.get("step"),
        decimals=ui.get("decimals"),
        choices=choices,
    )


def _build_excluded_field(dest: str, help_by_flag: dict) -> UiField:
    spec = dict(_EXCLUDED_FIELDS[dest])
    spec.pop("group", None)
    flag = f"--{dest.replace('_', '-')}"
    tooltip = help_by_flag.get(flag) or ""
    choices = spec.pop("choices", None)
    return UiField(
        dest=dest,
        schema_path=None,
        widget=spec.pop("widget"),
        default=spec.pop("default"),
        label=spec.pop("label", dest),
        tooltip=tooltip,
        choices=tuple(choices) if choices else None,
        **spec,
    )


def _resolve_field(dest: str, schema_idx: dict, help_by_flag: dict) -> UiField:
    if dest in schema_idx:
        return _build_schema_field(dest, schema_idx[dest])
    if dest in _EXCLUDED_FIELDS:
        return _build_excluded_field(dest, help_by_flag)
    raise KeyError(f"ui_spec: dest {dest!r} has no schema field or excluded entry")


def _toggle_kwargs(owner_dest: str, schema_idx: dict, help_by_flag: dict) -> dict:
    """Toggle owner semantics for a checkable group."""
    if owner_dest in schema_idx:
        rec = schema_idx[owner_dest]
        ui = rec["ui"]
        annotation = rec["annotation"]
        off = ui["off_value"]
        if annotation is bool:
            # The groupbox checkbox IS the owner -- no inner widget.
            return dict(toggle_dest=owner_dest, toggle_off_value=off,
                        toggle_owner_widget=None,
                        toggle_label=ui.get("label", owner_dest),
                        toggle_tooltip=rec["tooltip"])
        # combo owner (adaptive_ray / smooth_snap): inner combo shown when on.
        choices = tuple(ui["choices"])
        on_default = ui["default"] if "default" in ui else rec["default"]
        return dict(toggle_dest=owner_dest, toggle_off_value=off,
                    toggle_owner_widget="combo", toggle_on_default=on_default,
                    toggle_choices=choices,
                    toggle_label=ui.get("label", owner_dest),
                    toggle_tooltip=rec["tooltip"])
    # Excluded owner: rf_gazelle_model (line-edit, off = None).
    spec = _EXCLUDED_FIELDS[owner_dest]
    flag = f"--{owner_dest.replace('_', '-')}"
    return dict(toggle_dest=owner_dest, toggle_off_value=spec["default"],
                toggle_owner_widget=spec["widget"],
                toggle_on_default=spec["default"],
                toggle_label=spec.get("label", owner_dest),
                toggle_tooltip=help_by_flag.get(flag) or "")


def _build_group(gs: _GroupSpec, schema_idx: dict, help_by_flag: dict) -> UiGroup:
    fields = tuple(_resolve_field(d, schema_idx, help_by_flag) for d in gs.dests)
    subgroups = tuple(_build_group(s, schema_idx, help_by_flag) for s in gs.sub)
    kwargs: dict = {}
    if gs.toggle is not None:
        kwargs = _toggle_kwargs(gs.toggle, schema_idx, help_by_flag)
    advanced = bool(fields) and all(f.advanced for f in fields) and not subgroups \
        and gs.toggle is None
    return UiGroup(key=gs.key, title=gs.title, fields=fields,
                   subgroups=subgroups, advanced=advanced, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_ui_spec() -> list[UiGroup]:
    """Ordered ``UiGroup`` tree for the schema-driven Gaze Tuning surface."""
    schema_idx = _schema_index()
    _, help_by_flag = _flag_tables()
    return [_build_group(gs, schema_idx, help_by_flag) for gs in _GROUP_TREE]


def iter_fields(groups: list[UiGroup]):
    """Yield every UiField across the tree (excludes toggle owners that have no
    inner widget; combo/line toggle owners are surfaced as synthetic fields)."""
    for g in groups:
        yield from g.fields
        for sub in g.subgroups:
            yield from iter_fields([sub])


def all_dests(groups: list[UiGroup]) -> set[str]:
    """Every dest the surface writes, including toggle owners."""
    dests: set[str] = set()
    for g in groups:
        for f in g.fields:
            dests.add(f.dest)
        if g.toggle_dest is not None:
            dests.add(g.toggle_dest)
        dests |= all_dests(list(g.subgroups))
    return dests
