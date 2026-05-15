"""Focused unit tests for ms.utils.one_euro.OneEuroFilter.

Verifies the standard One Euro behaviour: heavy smoothing at rest (jitter
suppression), light smoothing on real motion (no lag).  Reference values
match the paper's canonical algorithm applied to synthetic inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from ms.utils.one_euro import OneEuroFilter


def test_first_sample_returns_input_unchanged():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dt=1.0 / 30.0)
    assert f.update(3.14) == pytest.approx(3.14, rel=1e-9)


def test_constant_input_settles_to_constant_output():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0, dt=1.0 / 30.0)
    for _ in range(50):
        y = f.update(5.0)
    assert y == pytest.approx(5.0, abs=1e-3)


def test_jitter_at_rest_is_suppressed_more_than_ema():
    """With beta=0, One Euro acts as a fixed low-pass at min_cutoff.  A
    small-amplitude high-frequency jitter around a mean value should be
    significantly damped after enough samples.
    """
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dt=1.0 / 30.0)
    rng = np.random.default_rng(seed=42)
    outputs = []
    for _ in range(200):
        x = 10.0 + 0.5 * rng.standard_normal()  # jitter amplitude 0.5
        outputs.append(f.update(x))
    # Ignore transient; measure last 100 samples' std.
    tail_std = float(np.std(outputs[-100:]))
    assert tail_std < 0.25, \
        f"one euro at min_cutoff=1.0 Hz, beta=0 should suppress jitter well below its input amplitude; got std {tail_std:.3f}"


def test_fast_ramp_signal_is_tracked_when_beta_large():
    """A monotonic ramp with a large derivative should track closely when
    beta is large, because the adaptive cutoff opens up on speed.
    """
    f = OneEuroFilter(min_cutoff=1.0, beta=5.0, d_cutoff=1.0, dt=1.0 / 30.0)
    for i in range(50):
        x = float(i)  # ramp: derivative 1.0/sample, 30/sec at dt=1/30
        y = f.update(x)
    # After 50 samples of x=i, output should be close to input.
    assert y == pytest.approx(x, rel=0.05), \
        f"with high beta, one euro should track fast ramps; got {y:.3f} for input {x:.3f}"


def test_step_input_settles_asymptotically():
    """A single step from 0 to 1 should approach 1 monotonically (no
    ringing) with default parameters.
    """
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5, d_cutoff=1.0, dt=1.0 / 30.0)
    f.update(0.0)  # prime state at 0
    ys = [f.update(1.0) for _ in range(30)]
    # Monotonically increasing.
    for prev, curr in zip(ys[:-1], ys[1:]):
        assert curr >= prev - 1e-9, f"non-monotonic: {prev} -> {curr}"
    # Reaches close to 1 by the end.
    assert ys[-1] > 0.95, f"step response should have converged near 1; got {ys[-1]:.3f}"
