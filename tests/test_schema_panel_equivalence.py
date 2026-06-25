"""SchemaPanel <-> legacy hand-section namespace equivalence (SP3.1 Batch F,
Step 14 -- THE gate of the batch).

SchemaPanel is instantiated NEXT TO the still-live ray / performance / phenomena
hand sections and asserted to emit a byte-equal ``namespace_values()`` dict for:

  (a) defaults,
  (b) after apply_namespace of test_pipeline.yaml,
  (c) a KNOWN_GOOD Gaze-LLE blend namespace,
  (d) every toggle group checked AND unchecked (T10 off-values, with
      non-default members still emitting),
  (e) apply(build()) round-trip idempotence.

Offscreen Qt; the legacy sections are NOT modified (they are deleted in Batch G).
Namespaces are compared key-by-key with a full diff on failure (a bare assert on
a 74-key dict is undebuggable).
"""
from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _legacy_values(ray, perf, phen) -> dict:
    vals: dict = {}
    vals.update(ray.namespace_values())
    vals.update(perf.namespace_values())
    vals.update(phen.get_values())
    return vals


def _assert_equal(schema_vals: dict, legacy_vals: dict, label: str):
    if schema_vals == legacy_vals:
        return
    missing = sorted(set(legacy_vals) - set(schema_vals))
    extra = sorted(set(schema_vals) - set(legacy_vals))
    changed = {
        k: {"legacy": legacy_vals[k], "schema": schema_vals[k]}
        for k in sorted(set(legacy_vals) & set(schema_vals))
        if legacy_vals[k] != schema_vals[k]
    }
    import json
    pytest.fail(
        f"[{label}] SchemaPanel namespace diverged from legacy sections\n"
        f"missing from schema ({len(missing)}): {missing}\n"
        f"extra in schema ({len(extra)}): {extra}\n"
        f"changed ({len(changed)}):\n"
        + json.dumps(changed, indent=2, sort_keys=True, default=repr))


def _build_panels(qapp):
    from mindsight.GUI.gaze_tab.performance_section import PerformanceSection
    from mindsight.GUI.gaze_tab.ray_section import RaySection
    from mindsight.GUI.phenomena_panel import PhenomenaPanel
    from mindsight.GUI.schema_panel import SchemaPanel
    return (SchemaPanel(), RaySection(), PerformanceSection(), PhenomenaPanel())


def _apply_all(ns, schema, ray, perf, phen):
    schema.apply_namespace(ns)
    ray.apply_namespace(ns)
    perf.apply_namespace(ns)
    phen.apply_values(vars(ns))


# -- (a) defaults -------------------------------------------------------------

def test_defaults_equivalent(qapp):
    schema, ray, perf, phen = _build_panels(qapp)
    _assert_equal(schema.namespace_values(),
                  _legacy_values(ray, perf, phen), "defaults")
    # sanity: the surface writes exactly 74 dests
    assert len(schema.namespace_values()) == 74


# -- (b) test_pipeline.yaml ---------------------------------------------------

def test_pipeline_yaml_equivalent(qapp):
    from mindsight.config_compat import load_pipeline
    ns = load_pipeline(str(REPO_ROOT / "test_pipeline.yaml"), Namespace())
    schema, ray, perf, phen = _build_panels(qapp)
    _apply_all(ns, schema, ray, perf, phen)
    _assert_equal(schema.namespace_values(),
                  _legacy_values(ray, perf, phen), "test_pipeline.yaml")


# -- (c) KNOWN_GOOD blend namespace ------------------------------------------

def test_known_good_blend_equivalent(qapp):
    # Gaze-LLE Blend primary operating mode (configs/KNOWN_GOOD.md, 2026-07-05).
    ns = Namespace(
        rf_gazelle_model="Weights/Gazelle/gazelle_dinov2_vitb14.pt",
        rf_gazelle_name="gazelle_dinov2_vitb14",
        rf_gazelle_interval=10,        # -> min_call_gap via resolve_min_call_gap
        dir_beta=0.5, len_beta=0.3, len_hold_tau=5.0,
        fixation_v_threshold=0.04, fixation_d_threshold=0.15,
        ray_length=1.5, gaze_cone=5.0, reid_grace_seconds=4.0,
        adaptive_ray="snap", snap_dist=180.0,
    )
    schema, ray, perf, phen = _build_panels(qapp)
    _apply_all(ns, schema, ray, perf, phen)
    schema_vals = schema.namespace_values()
    _assert_equal(schema_vals, _legacy_values(ray, perf, phen), "known_good_blend")
    # the blend must actually be engaged in the produced namespace
    assert schema_vals["rf_gazelle_model"] == \
        "Weights/Gazelle/gazelle_dinov2_vitb14.pt"
    assert schema_vals["min_call_gap"] == 10
    assert schema_vals["len_hold_tau"] == 5.0


# -- (d) toggle groups checked AND unchecked (T10) ---------------------------
#
# T10 is a BUILD contract (what namespace_values emits per group check-state),
# so case (d) drives the checkable groups DIRECTLY on both surfaces rather than
# through apply_namespace.  (Note: the legacy ray_section.apply_namespace /
# reset_defaults never touch the Depth sub-panel at all -- depth is write-only
# there; SchemaPanel round-trips it correctly.  Driving check-states directly
# tests the T10 off-value contract faithfully without tripping that legacy
# apply gap; apply-parity for real namespaces is covered by cases a/b/c.)

# owner dest -> legacy checkable-group attribute
_LEGACY_GROUP_ATTR = {
    "rf_gazelle_model": ("_gazelle_group", "ray"),
    "adaptive_ray": ("_adaptive_snap_group", "ray"),
    "smooth_snap": ("_smooth_group", "ray"),
    "gaze_lock": ("_fixation_group", "ray"),
    "depth": ("_depth_group", "ray"),
    "gaze_tips": ("_gaze_tips_group", "ray"),
    "joint_attention": ("_ja_grp", "phen"),
    "gaze_aversion": ("_gaze_aversion", "phen"),
}

# (schema dest, schema-widget-is-owner?, legacy widget attr, owner-panel, value)
# member widgets set to NON-DEFAULT on BOTH surfaces, to prove members keep
# emitting when their group is unchecked.
_MEMBER_SETS = [
    ("tip_radius", "ray", "_tip_radius", 120),
    ("snap_dist", "ray", "_snap_dist", 200),
    ("dwell_frames", "ray", "_dwell_frames", 22),
    ("depth_input_size", "ray", "_depth_input_size", 448),
    ("ja_window", "phen", "_ja_window", 12),
    ("aversion_window", "phen", "_aversion_window", 90),
]


def _set_all_groups(schema, ray, phen, checked: bool):
    panels = {"ray": ray, "phen": phen}
    for dest, tg in schema._toggles.items():
        tg["group"].setChecked(checked)
        attr, panel = _LEGACY_GROUP_ATTR[dest]
        getattr(panels[panel], attr).setChecked(checked)


def _set_members(schema, ray, phen):
    panels = {"ray": ray, "phen": phen}
    for dest, panel, attr, value in _MEMBER_SETS:
        schema._fields[dest][1].setValue(value)
        getattr(panels[panel], attr).setValue(value)


def test_all_toggle_groups_on_equivalent(qapp):
    """Every checkable group CHECKED -> owners emit their on-values."""
    schema, ray, perf, phen = _build_panels(qapp)
    _set_members(schema, ray, phen)
    _set_all_groups(schema, ray, phen, True)
    sv = schema.namespace_values()
    _assert_equal(sv, _legacy_values(ray, perf, phen), "all_toggles_on")
    # on-values (combo owners at their first/harvested ON choice)
    assert sv["adaptive_ray"] == "extend"     # legacy combo default index 0
    assert sv["smooth_snap"] == "all"          # legacy combo default index 2
    assert sv["gaze_lock"] is True and sv["depth"] is True
    assert sv["gaze_tips"] is True and sv["joint_attention"] is True


def test_all_toggle_groups_off_keeps_members(qapp):
    """T10: every checkable group UNCHECKED -> owners emit off-values while
    non-default members keep emitting their widget values."""
    schema, ray, perf, phen = _build_panels(qapp)
    _set_members(schema, ray, phen)
    _set_all_groups(schema, ray, phen, False)
    sv = schema.namespace_values()
    _assert_equal(sv, _legacy_values(ray, perf, phen), "all_toggles_off")
    # off-values landed (T10)
    assert sv["rf_gazelle_model"] is None
    assert sv["adaptive_ray"] == "off"
    assert sv["smooth_snap"] == "off"
    assert sv["gaze_lock"] is False
    assert sv["depth"] is False
    assert sv["gaze_tips"] is False
    assert sv["joint_attention"] is False
    assert sv["gaze_aversion"] is False
    # members still emit their non-default values under an unchecked group
    assert sv["tip_radius"] == 120
    assert sv["snap_dist"] == 200.0
    assert sv["dwell_frames"] == 22
    assert sv["depth_input_size"] == 448
    assert sv["ja_window"] == 12
    assert sv["aversion_window"] == 90


# -- (e) round-trip idempotence ----------------------------------------------

def test_apply_build_roundtrip_idempotent(qapp):
    from mindsight.GUI.schema_panel import SchemaPanel
    # defaults
    panel = SchemaPanel()
    once = panel.namespace_values()
    panel.apply_namespace(Namespace(**once))
    assert panel.namespace_values() == once
    # a fully-populated blend namespace
    blend = Namespace(
        rf_gazelle_model="x.pt", rf_gazelle_name="gazelle_dinov2_vitl14",
        adaptive_ray="snap", snap_dist=200.0, smooth_snap="objects",
        smooth_snap_alpha=0.4, gaze_lock=True, dwell_frames=22, depth=True,
        depth_input_size=448, joint_attention=True, ja_window=12,
        gaze_aversion=True, aversion_window=90, len_hold_tau=7.0, ray_length=1.5,
    )
    panel2 = SchemaPanel()
    panel2.apply_namespace(blend)
    built = panel2.namespace_values()
    panel2.apply_namespace(Namespace(**built))
    assert panel2.namespace_values() == built
