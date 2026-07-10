"""Fast guard on the shipped known-good pipeline config.

``configs/pipeline_known_good.yaml`` is a documentation-as-config preset whose
values are all traceable to ``configs/KNOWN_GOOD.md``. This test pins that the
shipped file still loads under the strict schema and still carries the ruled
values.

Two loaders are exercised, because they see different parts of the file:

* ``load_yaml`` is the STRICT schema loader (used by preflight). It reflects the
  first-class keys only. It silently DROPS unknown keys -- so a load that
  "succeeds" is NOT by itself a typo guard; this test asserts VALUES, not just a
  clean load.
* ``load_pipeline`` is the RUNTIME loader. It routes the ``plugins:`` pass-through
  (weight filenames, blend wiring, merge-overlaps) onto the namespace the
  pipeline actually consumes. The merge-overlaps settings and weight names have
  no first-class schema key, so their coverage lives here.
"""

from argparse import Namespace
from pathlib import Path

from mindsight.cli_flags import build_parser
from mindsight.config_compat import load_pipeline, load_yaml
from mindsight.Phenomena.phenomena_config import PhenomenaConfig
from mindsight.Phenomena.phenomena_pipeline import init_phenomena_trackers
from mindsight.pipeline_config import DetectionConfig
from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pipeline_known_good.yaml"

# The eight built-in trackers the preset must enable (schema PhenomenaSection
# toggle fields), each paired with the tracker class init_phenomena_trackers
# instantiates for it.
PHENOMENA_TOGGLES = (
    "joint_attention", "mutual_gaze", "social_ref", "gaze_follow",
    "gaze_aversion", "scanpath", "gaze_leader", "attn_span",
)
EXPECTED_TRACKER_NAMES = {
    "JointAttentionTracker", "MutualGazeTracker", "SocialReferenceTracker",
    "GazeFollowingTracker", "AttentionSpanTracker", "GazeAversionTracker",
    "ScanpathTracker", "GazeLeadershipTracker",
}


def test_config_file_exists():
    assert CONFIG.is_file(), f"shipped config missing: {CONFIG}"


def test_strict_load_first_class_values():
    """Strict schema load reflects every first-class ruled value."""
    cfg = load_yaml(CONFIG)
    assert cfg.detection.conf == 0.25
    assert cfg.gaze.ray_length == 1.4
    assert cfg.gaze.gaze_cone_angle == 5.0
    assert cfg.tracker.reid_grace_seconds == 4.5
    # rf_gazelle_interval 10 routes to rayforming.min_call_gap.
    assert cfg.rayforming.min_call_gap == 10
    # TESTGOOD export merged 2026-07-09 (user-validated, layered on top).
    # forward_gaze_threshold rides the plugins passthrough (no strict key),
    # so it is asserted in test_runtime_testgood_values instead.
    assert cfg.gaze.gaze_tips is True
    assert cfg.gaze.tip_radius == 70
    assert cfg.gaze.detect_extend_scope == "both"


def test_runtime_testgood_values():
    """The TESTGOOD keys reach the runtime namespace, incl. the
    performance-section no_dashboard passthrough (schema_path=None)."""
    ns = load_pipeline(CONFIG, Namespace())
    assert getattr(ns, "gaze_tips") is True
    assert getattr(ns, "tip_radius") == 70
    assert getattr(ns, "detect_extend_scope") == "both"
    assert getattr(ns, "forward_gaze_threshold") == 13.0
    assert getattr(ns, "no_dashboard") is True


def test_runtime_weight_names_are_bare_filenames():
    """Weight paths are bare filenames (portable via resolve_weight)."""
    ns = load_pipeline(CONFIG, Namespace())
    for attr, expected in (
        ("model", "yoloe-v8l-seg.pt"),
        ("mgaze_model", "resnet50_gaze.onnx"),
        ("rf_gazelle_model", "gazelle_dinov2_vitb14.pt"),
    ):
        val = getattr(ns, attr)
        assert val == expected
        assert "/" not in val and "\\" not in val, f"{attr} is not a bare filename: {val}"


def test_runtime_blend_and_interval():
    ns = load_pipeline(CONFIG, Namespace())
    assert getattr(ns, "rf_gazelle_interval") == 10
    assert resolve_min_call_gap(ns) == 10
    assert getattr(ns, "mgaze_arch") == "resnet50"


def test_all_phenomena_enabled_strict_load():
    """Every built-in phenomena toggle is ON under the strict schema loader.

    User ruling 2026-07-09: all phenomena default ON via the preset. The strict
    loader silently drops unknown keys, so this asserts VALUES, not just a load.
    """
    cfg = load_yaml(CONFIG)
    for toggle in PHENOMENA_TOGGLES:
        assert getattr(cfg.phenomena, toggle) is True, f"{toggle} not enabled"


def test_all_phenomena_build_eight_trackers():
    """The preset enables all 8 trackers via the runtime loader, and
    init_phenomena_trackers instantiates one instance per tracker."""
    ns = load_pipeline(CONFIG, build_parser().parse_args([]))
    cfg = PhenomenaConfig.from_namespace(ns)
    assert all(getattr(cfg, t) for t in PHENOMENA_TOGGLES)
    trackers = init_phenomena_trackers(cfg)
    assert len(trackers) == 8
    assert {type(t).__name__ for t in trackers} == EXPECTED_TRACKER_NAMES


def test_known_good_preset_path_present():
    """The resolver returns the shipped preset when it exists under the resource
    root (PROJECT_ROOT), which is what the GUI seeds from."""
    from mindsight.config_compat import known_good_preset_path
    resolved = known_good_preset_path()
    assert resolved is not None
    assert resolved == CONFIG
    assert resolved.name == "pipeline_known_good.yaml"


def test_known_good_preset_path_absent(monkeypatch, tmp_path):
    """When the resource root carries no preset, the resolver returns None.

    Patches only mindsight.constants.PROJECT_ROOT (which is what MINDSIGHT_HOME
    resolves into at import); the resolver reads it at call time. No GUI or
    weights are constructed here, so the import-time-bound Weights root is
    untouched -- no stale-path leak.
    """
    import mindsight.constants as constants
    monkeypatch.setattr(constants, "PROJECT_ROOT", tmp_path)
    from mindsight.config_compat import known_good_preset_path
    assert known_good_preset_path() is None


def test_runtime_merge_overlaps_effective():
    """Merge-overlaps (plugins pass-through) reaches the detection config."""
    ns = load_pipeline(CONFIG, Namespace(detect_scale=1.0))
    assert getattr(ns, "merge_overlaps") is True
    assert getattr(ns, "merge_overlap_strategy") == "dynamic"
    assert getattr(ns, "merge_overlap_threshold") == 0.55
    det = DetectionConfig.from_namespace(ns, None, set())
    assert det.merge_overlaps is True
    assert det.merge_overlap_strategy == "dynamic"
    assert det.merge_overlap_threshold == 0.55
    assert det.conf == 0.25
