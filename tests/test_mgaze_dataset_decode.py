"""W3X: --mgaze-dataset now reaches the ONNX backend's decode geometry.

The vendored GazeEstimationONNX hardcodes gaze360 decode constants
(90 bins x 4 deg - 180), so an MPIIGaze-trained export (28 bins x 3 deg
- 42, the vendored library's own training config) mis-decoded through it.
_GazeONNXWithConf now selects the bin geometry from DATA_CONFIG exactly
like the PyTorch backend; the default construction is bit-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from mindsight.GazeTracking.Backends.MGaze.MGaze_Tracking import _GazeONNXWithConf


class _Io:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _StubSession:
    """Just enough onnxruntime surface for GazeEstimationONNX.__init__."""

    def __init__(self, bins):
        self._bins = bins

    def get_inputs(self):
        return [_Io("input", [1, 3, 448, 448])]

    def get_outputs(self):
        return [_Io("pitch"), _Io("yaw")]

    def run(self, names, feed):
        logits = np.full((1, self._bins), -20.0, dtype=np.float32)
        logits[0, self._bins // 2] = 20.0
        return [logits, logits.copy()]


def _one_hot(bins, k):
    logits = np.full((1, bins), -20.0, dtype=np.float32)
    logits[0, k] = 20.0
    return logits


def test_default_dataset_keeps_gaze360_geometry():
    eng = _GazeONNXWithConf(None, session=_StubSession(90))
    assert (eng._bins, eng._binwidth, eng._angle_offset) == (90, 4, 180)
    pitch, yaw = eng.decode(_one_hot(90, 45), _one_hot(90, 45))
    assert pitch == pytest.approx(0.0, abs=1e-4)   # 45*4 - 180 = 0 deg


def test_mpiigaze_dataset_switches_geometry():
    eng = _GazeONNXWithConf(None, session=_StubSession(28),
                            dataset="mpiigaze")
    assert (eng._bins, eng._binwidth, eng._angle_offset) == (28, 3, 42)
    assert eng.idx_tensor.shape == (28,)
    pitch, _yaw = eng.decode(_one_hot(28, 20), _one_hot(28, 14))
    assert pitch == pytest.approx(np.radians(18.0), abs=1e-4)  # 20*3 - 42


def test_mpiigaze_estimate_end_to_end_with_confidence():
    eng = _GazeONNXWithConf(None, session=_StubSession(28),
                            dataset="mpiigaze")
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    pitch, yaw, conf = eng.estimate(face)
    assert pitch == pytest.approx(np.radians(14 * 3 - 42), abs=1e-4)
    assert yaw == pytest.approx(pitch)
    assert conf == pytest.approx(1.0, abs=1e-3)    # one-hot softmax peak


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="mgaze-dataset"):
        _GazeONNXWithConf(None, session=_StubSession(28), dataset="typo")
