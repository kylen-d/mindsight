"""W3Z: ONNX gaze-target engine (gazelle-dinov3 exports) + .onnx dispatch.

The engine is exercised through a fake onnxruntime session (no model file,
no onnxruntime import); the provider dispatch test only checks routing.
"""
from __future__ import annotations

import numpy as np
import pytest

from Plugins.GazeTracking.Gazelle.gazelle_onnx_engine import GazelleOnnxEngine

FRAME = np.full((480, 640, 3), 90, dtype=np.uint8)
BOXES = [(100, 100, 200, 200), (400, 150, 500, 260)]


class _IO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _FakeSession:
    """Mimics the gazelle-dinov3 1xNx4 export I/O: image + bboxes in,
    [N,S,S] heatmaps (+ optional [N] inout) out."""

    def __init__(self, hm_size=32, with_inout=True, size=320):
        self._hm = hm_size
        self._inout = with_inout
        self._inputs = [_IO("image_bgr", [1, 3, size, size]),
                        _IO("bboxes_x1y1x2y2", [1, "N", 4])]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        n = 2 if self._inout else 1
        return [_IO(f"out{i}", None) for i in range(n)]

    def run(self, _names, feed):
        self.last_feed = feed
        n = feed["bboxes_x1y1x2y2"].shape[1]
        hms = np.zeros((n, self._hm, self._hm), dtype=np.float32)
        for k in range(n):
            hms[k, self._hm // 2, self._hm // 2] = 1.0
        outs = [hms]
        if self._inout:
            outs.append(np.full((n,), 0.9, dtype=np.float32))
        return outs


def test_engine_resizes_heatmaps_to_64_and_sets_inout():
    sess = _FakeSession(hm_size=32, with_inout=True)
    eng = GazelleOnnxEngine("fake.onnx", session=sess)
    hms = eng.raw_heatmaps(FRAME, BOXES)
    assert hms.shape == (2, 64, 64) and hms.dtype == np.float32
    assert eng._last_inout is not None
    np.testing.assert_allclose(eng._last_inout, [0.9, 0.9])
    # peak survives the 32 -> 64 resize near the map centre
    gy, gx = np.unravel_index(np.argmax(hms[0]), hms[0].shape)
    assert abs(gy - 32) <= 2 and abs(gx - 32) <= 2


def test_engine_feeds_resized_bgr_and_normalized_boxes():
    sess = _FakeSession(size=320)
    eng = GazelleOnnxEngine("fake.onnx", session=sess)
    eng.raw_heatmaps(FRAME, BOXES)
    img = sess.last_feed["image_bgr"]
    assert img.shape == (1, 3, 320, 320) and img.dtype == np.float32
    boxes = sess.last_feed["bboxes_x1y1x2y2"]
    np.testing.assert_allclose(
        boxes[0, 0], [100 / 640, 100 / 480, 200 / 640, 200 / 480])


def test_engine_no_faces_and_no_inout_variant():
    eng = GazelleOnnxEngine("fake.onnx", session=_FakeSession(with_inout=False))
    assert eng.raw_heatmaps(FRAME, []).shape == (0, 64, 64)
    assert eng._last_inout is None
    eng.raw_heatmaps(FRAME, BOXES)
    assert eng._last_inout is None            # single-output variant


def test_engine_native_64_maps_pass_through():
    eng = GazelleOnnxEngine("fake.onnx", session=_FakeSession(hm_size=64))
    hms = eng.raw_heatmaps(FRAME, BOXES)
    assert hms.shape == (2, 64, 64)
    assert hms[0, 32, 32] == 1.0              # untouched, no interpolation


def test_engine_rejects_unrecognized_io():
    class _Bad(_FakeSession):
        def get_inputs(self):
            return [_IO("bboxes_a", None), _IO("bboxes_b", None)]

    with pytest.raises(ValueError):
        GazelleOnnxEngine("fake.onnx", session=_Bad())


def test_provider_dispatches_onnx_by_extension(tmp_path, monkeypatch):
    from argparse import Namespace

    import Plugins.GazeTracking.Gazelle.gazelle_onnx_engine as onnx_mod
    from mindsight.PostProcessing.RayForming.gazelle_provider import (
        GazelleProvider,
    )

    onnx_file = tmp_path / "gazelle_hgnetv2_atto.onnx"
    onnx_file.write_bytes(b"x")
    built = {}

    class _StubEngine:
        def __init__(self, path, providers=None, session=None):
            built["path"] = str(path)
            self._last_inout = None

    monkeypatch.setattr(onnx_mod, "GazelleOnnxEngine", _StubEngine)
    ns = Namespace(rf_gazelle_model=str(onnx_file))
    provider = GazelleProvider.from_namespace(ns)
    assert provider is not None
    assert built["path"] == str(onnx_file)
    # The cheap length channel shares the ONNX engine (no CUDA fp16 sibling).
    assert provider._length_engine is provider._engine
