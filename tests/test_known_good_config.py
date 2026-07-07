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

from mindsight.config_compat import load_pipeline, load_yaml
from mindsight.pipeline_config import DetectionConfig
from mindsight.PostProcessing.RayForming.ray_config import resolve_min_call_gap

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pipeline_known_good.yaml"


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
