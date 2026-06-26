"""Schema-driven pipeline YAML export round-trip (SP3.1 Step 17).

``_namespace_to_yaml_dict`` derives the exported YAML from the config-compat
canonical tables (``_YAML_MAP`` / ``_PHENOMENA_TOGGLES`` / ``_PHENOMENA_PARAMS``)
+ schema defaults, replacing the hand-written toggle/param maps.  These tests pin
the contract the GUI relies on: export a namespace, re-import it the way the GUI
does (``load_pipeline`` into a fresh namespace, then apply through the widgets),
and get a census-equal namespace back -- for the default census AND a KNOWN_GOOD
Gaze-LLE blend config.  Export stays only-non-default.
"""

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from mindsight.config_compat import load_pipeline  # noqa: E402
from mindsight.GUI.pipeline_dialog import _namespace_to_yaml_dict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS = REPO_ROOT / "tests" / "data" / "gui_namespace_golden.json"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _census_ns() -> Namespace:
    d = json.loads(CENSUS.read_text())
    d.pop("_version", None)
    return Namespace(**d)


def _gui_census(ns) -> dict:
    """Apply *ns* through a SchemaPanel and read back its 74-dest namespace.

    The export governs the schema-driven tuning surface; scoping the round-trip
    to the SchemaPanel isolates it from unrelated hand-widget apply/build quirks
    (DetectionSection merge-strategy, gaze_backend defaults)."""
    from mindsight.GUI.schema_panel import SchemaPanel
    panel = SchemaPanel()
    panel.apply_namespace(ns)
    return panel.namespace_values()


def _roundtrip_census(ns, tmp_path) -> dict:
    """Export *ns*, re-import the way import_pipeline does, read the census."""
    cfg = _namespace_to_yaml_dict(ns)
    p = tmp_path / "exported.yaml"
    p.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    reimported = load_pipeline(str(p), Namespace())
    return _gui_census(reimported)


def test_default_census_roundtrips(qapp, tmp_path):
    ns = _census_ns()
    before = _gui_census(ns)
    after = _roundtrip_census(ns, tmp_path)
    assert after == before


def test_default_census_export_is_minimal(qapp, tmp_path):
    """A default namespace exports no schema-relevant tuning sections."""
    cfg = _namespace_to_yaml_dict(_census_ns())
    assert "gaze" not in cfg          # every gaze value is default
    assert "phenomena" not in cfg     # no phenomena enabled by default


def test_known_good_blend_roundtrips(qapp, tmp_path):
    # Gaze-LLE Blend primary mode (configs/KNOWN_GOOD.md); non-defaults only.
    ns = _census_ns()
    ns.rf_gazelle_model = "Weights/Gazelle/gazelle_dinov2_vitb14.pt"
    ns.rf_gazelle_name = "gazelle_dinov2_vitb14"
    ns.min_call_gap = 10
    ns.ray_length = 1.5
    ns.gaze_cone = 5.0
    ns.reid_grace_seconds = 4.0
    ns.adaptive_ray = "snap"
    ns.snap_dist = 180.0

    cfg = _namespace_to_yaml_dict(ns)
    # blend wiring survives the export
    assert cfg["plugins"]["rf_gazelle_model"].endswith("vitb14.pt")
    assert cfg["plugins"]["min_call_gap"] == 10
    assert cfg["gaze"]["ray_length"] == 1.5
    assert cfg["gaze"]["adaptive_ray"] == "snap"

    before = _gui_census(ns)
    after = _roundtrip_census(ns, tmp_path)
    assert after == before


def test_phenomena_export_uses_canonical_names(qapp, tmp_path):
    ns = _census_ns()
    ns.joint_attention = True
    ns.ja_window = 12
    ns.social_ref = True
    ns.social_ref_window = 90
    ns.gaze_aversion = True
    ns.aversion_window = 45

    cfg = _namespace_to_yaml_dict(ns)
    names = set()
    for item in cfg["phenomena"]:
        names.update(item.keys() if isinstance(item, dict) else {item})
    assert "joint_attention" in names          # canonical yaml name
    assert "social_referencing" in names       # dest social_ref -> yaml name
    assert "gaze_aversion" in names

    before = _gui_census(ns)
    after = _roundtrip_census(ns, tmp_path)
    assert after == before
