"""Focused unit tests for FixationDetector.

Verifies the four canonical PY signal scenarios produce the expected
fixation_likelihood range:
  - pure fixation -> ~1
  - pure saccade  -> ~0
  - small jitter  -> ~1 (regression against problem 3)
  - slow drift    -> ~0 via dispersion gate
"""
from __future__ import annotations

import numpy as np
import pytest

from ms.PostProcessing.RayForming.py_history import PYHistoryBuffer
from ms.PostProcessing.RayForming.fixation_detector import FixationDetector


def _feed(det, buf, direction_generator, n):
    """Push n samples into buf and update det on each."""
    last = None
    for i in range(n):
        d = direction_generator(i)
        buf.push(d)
        last = det.update(buf)
    return last


def _constant(theta):
    v = np.array([np.cos(theta), np.sin(theta)])
    return lambda i: v


def test_pure_fixation_yields_high_likelihood():
    det = FixationDetector(v_threshold=0.02, d_threshold=0.10)
    buf = PYHistoryBuffer(size=10)
    likelihood = _feed(det, buf, _constant(0.5), n=15)
    assert likelihood > 0.9, f"constant py_dir should give likelihood ~1; got {likelihood:.3f}"


def test_pure_saccade_yields_low_likelihood():
    det = FixationDetector(v_threshold=0.02, d_threshold=0.10)
    buf = PYHistoryBuffer(size=10)
    # Ramp yaw quickly -- large frame-to-frame direction change.
    def ramp(i):
        theta = 0.05 * i    # 0.05 rad/frame >> v_threshold
        return np.array([np.cos(theta), np.sin(theta)])
    likelihood = _feed(det, buf, ramp, n=15)
    assert likelihood < 0.1, f"fast ramp should give likelihood ~0; got {likelihood:.3f}"


def test_small_jitter_around_fixation_stays_fixated():
    """Regression against problem 3: raw frame-to-frame velocity is
    dominated by jitter during real fixations; the smoothed velocity
    inside the detector should strip that out.
    """
    det = FixationDetector(v_threshold=0.02, d_threshold=0.10)
    buf = PYHistoryBuffer(size=10)
    rng = np.random.default_rng(seed=7)
    def jittered(i):
        theta = 0.5 + 0.005 * rng.standard_normal()  # tiny angular jitter
        return np.array([np.cos(theta), np.sin(theta)])
    likelihood = _feed(det, buf, jittered, n=30)
    assert likelihood > 0.8, \
        f"small-amplitude jitter should not defeat fixation detection; got {likelihood:.3f}"


def test_slow_drift_drops_via_dispersion_gate():
    """Velocity below threshold but cumulative drift outside dispersion
    tolerance -> likelihood drops via the dispersion factor.

    Uses a larger buffer (size=30) because the standard 10-frame window
    holds at most ~5*v_threshold of dispersion -- a below-v_threshold
    drift needs a longer window to accumulate enough spread to trip
    d_threshold.
    """
    det = FixationDetector(v_threshold=0.02, d_threshold=0.10)
    buf = PYHistoryBuffer(size=30)
    # Slow drift: 0.018 rad/frame -- below v_threshold, but 15 frames = 0.252 rad
    # total drift, max deviation from mean ~ 0.126 rad > d_threshold.
    def drift(i):
        theta = 0.018 * i
        return np.array([np.cos(theta), np.sin(theta)])
    likelihood = _feed(det, buf, drift, n=15)
    assert likelihood < 0.3, \
        f"slow drift accumulates dispersion and should drop likelihood; got {likelihood:.3f}"


def test_nonpositive_thresholds_rejected():
    with pytest.raises(ValueError):
        FixationDetector(v_threshold=0.0, d_threshold=0.10)
    with pytest.raises(ValueError):
        FixationDetector(v_threshold=0.02, d_threshold=-1.0)
    with pytest.raises(ValueError):
        FixationDetector(v_threshold=0.02, d_threshold=0.10, v_scale=0.0)


def test_unstable_buffer_gives_zero_likelihood():
    det = FixationDetector(v_threshold=0.02, d_threshold=0.10)
    buf = PYHistoryBuffer(size=10)
    # Push just 3 samples -- buffer is still unstable (< half).
    buf.push(np.array([1.0, 0.0]))
    buf.push(np.array([1.0, 0.0]))
    buf.push(np.array([1.0, 0.0]))
    likelihood = det.update(buf)
    assert likelihood == 0.0, \
        f"unstable buffer should force likelihood=0 (safe default); got {likelihood:.3f}"
