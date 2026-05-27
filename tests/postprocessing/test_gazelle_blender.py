"""Integration tests for the fixation-aware GazeLLEBlender.

Verifies the blender's behavior under scheduler-driven trust signals:
  - trust=1 with accepted heatmap: output anchors to Gaze-LLE (direction
    from belief centroid, length from heatmap centroid).
  - trust=0 (pursuit): output is pure PY, length is PY baseline.
  - Unaccepted heatmaps are ignored.
  - Belief accumulates across multiple accepted heatmaps in a fixation.
"""
from __future__ import annotations

import numpy as np
import pytest

from ms.PostProcessing.RayForming.gazelle_blender import GazeLLEBlender
from ms.PostProcessing.RayForming.ray_config import RayFormingConfig


ORIGIN = np.array([320.0, 240.0])
FACE_WIDTH = 100.0
FRAME_W, FRAME_H = 640, 480
PITCH, YAW = 0.5, 0.0
DT = 1.0 / 30.0


def _make_blender(**kw) -> GazeLLEBlender:
    cfg = RayFormingConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return GazeLLEBlender(cfg)


def _blob_heatmap(gx: float, gy: float, sigma: float = 4.0,
                  size: int = 64) -> np.ndarray:
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    dx = xs - gx
    dy = ys - gy
    return np.exp(-0.5 * (dy[:, None] ** 2 + dx[None, :] ** 2) / sigma ** 2
                  ).astype(np.float32)


def _length(endpoint):
    return float(np.linalg.norm(np.asarray(endpoint) - ORIGIN))


def _run(b, *, hm=None, accept=False, trust=0.0, pitch=PITCH, yaw=YAW):
    return b.update(
        track_id=0, pitch=pitch, yaw=yaw, gaze_conf=1.0,
        origin=ORIGIN, face_width=FACE_WIDTH,
        frame_h=FRAME_H, frame_w=FRAME_W,
        gazelle_hm=hm, accept_heatmap=accept, trust=trust, dt=DT)


def test_no_inference_pure_py_output():
    """With trust=0 and no inference, output tracks PY exactly (length = PY baseline)."""
    b = _make_blender(dir_min_cutoff=100.0, len_min_cutoff=100.0)  # nearly no smoothing
    endpoint = _run(b, hm=None, accept=False, trust=0.0)
    assert _length(endpoint) == pytest.approx(FACE_WIDTH * 1.0, rel=0.05)


def test_accepted_inference_sets_length_from_heatmap_centroid():
    """When accept=True and trust=1, length converges to the heatmap-centroid distance."""
    b = _make_blender(dir_min_cutoff=100.0, len_min_cutoff=100.0)
    _run(b, hm=None, accept=False, trust=0.0)     # init state
    hm = _blob_heatmap(gx=48.0, gy=16.0, sigma=4.0)
    # A few frames of accepted + trust=1 to let One Euro converge.
    endpoint = None
    for _ in range(10):
        endpoint = _run(b, hm=hm, accept=True, trust=1.0)
    expected = float(np.linalg.norm(np.array([480.0, 120.0]) - ORIGIN))
    assert _length(endpoint) == pytest.approx(expected, rel=0.1), \
        f"length should converge toward heatmap centroid distance; got {_length(endpoint):.2f}, want {expected:.2f}"


def test_length_holds_after_trust_drops():
    """Regression: length must NOT bounce back to PY when trust dips.

    After an accepted inference latches the LLE length, dropping trust to 0
    for 20 frames (~0.67 s << len_hold_tau=5 s) must leave the ray still
    extended near the LLE reach.  Direction reverts on trust; length holds.
    """
    b = _make_blender(dir_min_cutoff=100.0, len_min_cutoff=100.0)
    _run(b, hm=None, accept=False, trust=0.0)
    hm = _blob_heatmap(gx=48.0, gy=16.0, sigma=4.0)
    for _ in range(5):
        _run(b, hm=hm, accept=True, trust=1.0)
    lle_length = float(np.linalg.norm(np.array([480.0, 120.0]) - ORIGIN))  # 200
    endpoint = None
    for _ in range(20):
        endpoint = _run(b, hm=None, accept=False, trust=0.0)
    assert _length(endpoint) > 0.8 * lle_length, \
        f"length should hold near LLE reach through a trust dip; " \
        f"got {_length(endpoint):.2f}, LLE reach {lle_length:.2f}"


def test_length_decays_to_py_on_long_timescale():
    """With a tiny len_hold_tau, an aged latch decays back to PY baseline."""
    b = _make_blender(dir_min_cutoff=100.0, len_min_cutoff=100.0,
                      len_hold_tau=0.05)
    _run(b, hm=None, accept=False, trust=0.0)
    hm = _blob_heatmap(gx=48.0, gy=16.0, sigma=4.0)
    for _ in range(5):
        _run(b, hm=hm, accept=True, trust=1.0)
    endpoint = None
    for _ in range(20):  # 0.67 s = 13x tau
        endpoint = _run(b, hm=None, accept=False, trust=0.0)
    py_length = FACE_WIDTH * 1.0
    assert _length(endpoint) == pytest.approx(py_length, rel=0.05), \
        f"aged latch should decay to PY; got {_length(endpoint):.2f}, want {py_length:.2f}"


def test_unaccepted_heatmap_does_not_change_length():
    """A heatmap arrives cached (accept=False, trust=0) -- blender ignores it."""
    b = _make_blender(dir_min_cutoff=100.0, len_min_cutoff=100.0)
    _run(b, hm=None, accept=False, trust=0.0)
    hm = _blob_heatmap(gx=48.0, gy=16.0, sigma=4.0)
    endpoint = _run(b, hm=hm, accept=False, trust=0.0)   # cached but not applied
    assert _length(endpoint) == pytest.approx(FACE_WIDTH * 1.0, rel=0.05)


def test_belief_accumulates_across_accepted_heatmaps():
    """Two accepted heatmaps at the same location produce a sharper belief peak."""
    b = _make_blender()
    _run(b, hm=None, accept=False, trust=1.0)
    hm = _blob_heatmap(gx=48.0, gy=16.0, sigma=6.0)   # wide blob
    _run(b, hm=hm, accept=True, trust=1.0)
    peak_1 = float(b._beliefs[0].max())
    _run(b, hm=hm, accept=True, trust=1.0)
    peak_2 = float(b._beliefs[0].max())
    assert peak_2 > peak_1 * 1.1, \
        f"second accepted heatmap at same location should sharpen belief; peak {peak_1:.4f} -> {peak_2:.4f}"


def test_prune_removes_track_state():
    b = _make_blender()
    _run(b, hm=None, accept=False, trust=0.0)
    assert 0 in b._beliefs
    b.prune(active_tids=set())
    assert 0 not in b._beliefs


def test_conf_ray_scales_py_length():
    """With conf_ray on and trust=0, blender length matches the
    confidence-scaled fallback, not the raw ray_length."""
    from ms.constants import CR_MIN, CR_MAX
    b = _make_blender(conf_ray=True, len_min_cutoff=100.0)
    # gaze_conf=0.5 -> rl = ray_length * (CR_MIN + 0.5*(CR_MAX-CR_MIN))
    endpoint = b.update(
        track_id=0, pitch=PITCH, yaw=YAW, gaze_conf=0.5,
        origin=ORIGIN, face_width=FACE_WIDTH,
        frame_h=FRAME_H, frame_w=FRAME_W,
        gazelle_hm=None, accept_heatmap=False, trust=0.0, dt=DT)
    expected_rl = 1.0 * (CR_MIN + 0.5 * (CR_MAX - CR_MIN))
    assert _length(endpoint) == pytest.approx(FACE_WIDTH * expected_rl, rel=0.05)
