"""LP2: --no-detector mode (faces + gaze rays + tip-based phenomena only)."""

from argparse import Namespace

import pytest


def test_flag_parses_and_defaults_off():
    from mindsight.cli_flags import parse_cli
    assert parse_cli([]).no_detector is False
    assert parse_cli(["--no-detector"]).no_detector is True


def test_null_detector_duck_type():
    """Same contract parse_dets relies on: callable -> results, .names."""
    from mindsight.ObjectDetection.model_factory import NullDetector
    det = NullDetector()
    assert det(object(), conf=0.35, classes=None, verbose=False) == []
    assert det.names == {}


def test_collect_weights_omits_yolo_family():
    from mindsight.outputs.provenance import collect_weights
    ns = Namespace(model="yolov8n.pt", vp_model="yoloe-26l-seg.pt",
                   vp_file=None, mgaze_model=None, rf_gazelle_model=None,
                   gazelle_model=None, no_detector=True)
    assert "model" not in collect_weights(ns)
    ns.no_detector = False
    assert "model" in collect_weights(ns)


def test_no_detector_with_vp_file_is_rejected_early():
    """The mutual exclusion raises BEFORE any model loads (plain English)."""
    from mindsight.cli_flags import parse_cli
    from mindsight.factory import build_from_namespace
    ns = parse_cli(["--no-detector", "--vp-file", "whatever.vp.json"])
    with pytest.raises(ValueError, match="cannot be combined"):
        build_from_namespace(ns)
