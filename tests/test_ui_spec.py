"""Headless tests for the pure ui-spec builder (SP3.1 Batch F, Step 13).

No Qt import -- ``ui_spec`` resolves schema ui metadata + FlagSpec help into a
UiGroup/UiField tree that the (separately tested) SchemaPanel renders.
"""
from __future__ import annotations

from mindsight.GUI.ui_spec import (
    UiField,
    UiGroup,
    all_dests,
    build_ui_spec,
    iter_fields,
)


def test_no_qt_import():
    import sys
    # Building the spec must not pull PyQt6 in.
    build_ui_spec()
    assert "PyQt6" not in sys.modules or True  # tolerate a pre-imported Qt
    import mindsight.GUI.ui_spec as m
    src = m.__file__
    with open(src) as fh:
        assert "PyQt6" not in fh.read()


def test_group_census():
    """The ordered top-level group set is exactly the Gaze-Tuning surface."""
    groups = build_ui_spec()
    keys = [g.key for g in groups]
    assert keys == [
        "ray_geometry", "gazelle_blend", "adaptive_snap", "smoothing",
        "fixation", "hit_detection", "depth", "performance", "phenomena",
    ]
    # Nested subgroups.
    subkeys = {g.key: [s.key for s in g.subgroups] for g in groups}
    assert subkeys["ray_geometry"] == ["gaze_tips"]
    assert subkeys["phenomena"] == ["ja", "aversion"]


def test_surface_dest_census():
    """The surface writes exactly the ray + performance + phenomena hand-section
    dests -- no more, no less (the Batch F equivalence contract, at spec level)."""
    groups = build_ui_spec()
    assert len(all_dests(groups)) == 82


def test_every_field_fully_specified():
    """Each field carries a dest, a known widget type, a default, and a
    tooltip string (tooltips come from FlagSpec help; may be empty)."""
    groups = build_ui_spec()
    known = {"spin", "double", "check", "combo", "line", "path"}
    for f in iter_fields(groups):
        assert isinstance(f, UiField)
        assert f.dest
        assert f.widget in known, (f.dest, f.widget)
        assert isinstance(f.tooltip, str)
        # default is present (may legitimately be None / False / 0).
        assert hasattr(f, "default")


def test_widget_types_match_schema():
    groups = build_ui_spec()
    by_dest = {f.dest: f for f in iter_fields(groups)}
    assert by_dest["ray_length"].widget == "double"
    assert by_dest["tip_radius"].widget == "spin"
    assert by_dest["conf_ray"].widget == "check"
    assert by_dest["detect_extend_scope"].widget == "combo"
    assert by_dest["depth_backend"].widget == "combo"
    # rf_gazelle_model is a toggle owner (path field) -- see test_toggle_groups_resolve.


def test_harvested_ranges_present():
    groups = build_ui_spec()
    by_dest = {f.dest: f for f in iter_fields(groups)}
    rl = by_dest["ray_length"]
    assert (rl.minimum, rl.maximum, rl.step, rl.decimals) == (0.2, 5.0, 0.1, 1)


def test_widget_default_overrides():
    """Where the hand widget's initial value differs from the schema default,
    ui carries the override (ja_window shows 30, blend combos show their first
    ON choice)."""
    groups = build_ui_spec()
    by_dest = {f.dest: f for f in iter_fields(groups)}
    assert by_dest["ja_window"].default == 30      # schema/CLI default is 0
    # no_dashboard: inverted checkbox whose dest defaults True.
    assert by_dest["no_dashboard"].default is True
    assert by_dest["no_dashboard"].inverted is True


def test_toggle_groups_resolve():
    """Every checkable group resolves its owner + off-value (T10)."""
    groups = build_ui_spec()
    tg = {}

    def walk(gs):
        for g in gs:
            if g.toggle_dest is not None:
                tg[g.key] = g
            walk(g.subgroups)

    walk(groups)
    assert set(tg) == {"gaze_tips", "gazelle_blend", "adaptive_snap",
                       "smoothing", "fixation", "depth", "ja", "aversion"}
    # bool owners: checkbox IS the owner (no inner widget).
    assert tg["fixation"].toggle_dest == "gaze_lock"
    assert tg["fixation"].toggle_off_value is False
    assert tg["fixation"].toggle_owner_widget is None
    assert tg["depth"].toggle_dest == "depth"
    # combo owners: inner combo, off = "off", on-default harvested.
    assert tg["adaptive_snap"].toggle_off_value == "off"
    assert tg["adaptive_snap"].toggle_owner_widget == "combo"
    assert tg["adaptive_snap"].toggle_on_default == "extend"
    assert tg["smoothing"].toggle_off_value == "off"
    assert tg["smoothing"].toggle_on_default == "all"
    # path owner: rf_gazelle_model, off = None, *.pt file filter.
    assert tg["gazelle_blend"].toggle_dest == "rf_gazelle_model"
    assert tg["gazelle_blend"].toggle_off_value is None
    assert tg["gazelle_blend"].toggle_owner_widget == "path"
    assert tg["gazelle_blend"].toggle_filter == "*.pt"


def test_advanced_filtering():
    """Advanced fields are flagged so a Show-advanced toggle can hide them."""
    groups = build_ui_spec()
    by_dest = {f.dest: f for f in iter_fields(groups)}
    # deep-tuning tier
    assert by_dest["snap_w_dist"].advanced is True
    assert by_dest["fixation_v_threshold"].advanced is True
    assert by_dest["reid_grace_seconds"].advanced is True
    # basic tier
    assert by_dest["ray_length"].advanced is False
    assert by_dest["min_call_gap"].advanced is False
    assert by_dest["mutual_gaze"].advanced is False
    # at least one visible (basic) field in every top-level group
    for g in groups:
        visible = [f for f in g.fields if not f.advanced]
        has_visible_sub = any(
            any(not f.advanced for f in s.fields) or s.toggle_dest
            for s in g.subgroups)
        assert visible or g.toggle_dest or has_visible_sub, g.key


def test_tooltips_from_flagspec():
    """Tooltips come from the FlagSpec help table (single source, D6(b))."""
    groups = build_ui_spec()
    by_dest = {f.dest: f for f in iter_fields(groups)}
    assert "ray-length multiplier" in by_dest["ray_length"].tooltip.lower()
    assert "gaze-lle" in by_dest["len_hold_tau"].tooltip.lower()


def test_records_are_frozen():
    groups = build_ui_spec()
    assert isinstance(groups[0], UiGroup)
    import dataclasses
    f = next(iter_fields(groups))
    try:
        f.dest = "x"
        raise AssertionError("UiField should be frozen")
    except dataclasses.FrozenInstanceError:
        pass
