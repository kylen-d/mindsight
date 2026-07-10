"""Regression: YOLOE model without classes/VP must not crash the detector
factory (fresh-install path).

The known-good preset seeds detection.model = a YOLOE weight into the Gaze
Tuning tab. A fresh install (no last_used.json) therefore starts a worker
with a YOLOE model, classes=None, and no VP prompt -- create_yolo_detector
used to call yolo.set_classes(None), which raises TypeError inside
ultralytics and aborted every run. Prompt-free YOLOE (built-in vocabulary)
is the correct fallback.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLOE_WEIGHT = REPO_ROOT / "Weights" / "YOLO" / "yoloe-v8s-seg.pt"


@pytest.mark.skipif(not YOLOE_WEIGHT.exists(), reason="YOLOE weight missing")
def test_yoloe_without_classes_loads_prompt_free():
    from mindsight.ObjectDetection.model_factory import create_yolo_detector

    yolo, class_ids, blacklist = create_yolo_detector(
        model_path=str(YOLOE_WEIGHT), classes=None, device="cpu")
    assert yolo is not None
    assert class_ids is None
    # Prompt-free YOLOE keeps its built-in vocabulary available.
    assert getattr(yolo, "names", None)
