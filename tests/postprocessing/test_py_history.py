"""Focused unit tests for ms.PostProcessing.RayForming.py_history.PYHistoryBuffer."""
from __future__ import annotations

import numpy as np
import pytest

from ms.PostProcessing.RayForming.py_history import PYHistoryBuffer


def test_new_buffer_is_unstable():
    b = PYHistoryBuffer(size=10)
    assert b.unstable is True
    assert b.count == 0


def test_pushes_up_to_size_then_wraps():
    b = PYHistoryBuffer(size=5)
    for i in range(8):
        b.push(np.array([float(i), 0.0]))
    # Only the last 5 survive.
    samples = b.samples()
    assert len(samples) == 5
    assert samples[0][0] == pytest.approx(3.0)  # first-surviving index
    assert samples[-1][0] == pytest.approx(7.0)


def test_stable_after_half_full():
    b = PYHistoryBuffer(size=10)
    for i in range(4):
        b.push(np.array([1.0, 0.0]))
    assert b.unstable is True
    b.push(np.array([1.0, 0.0]))     # 5th sample
    assert b.unstable is False


def test_samples_returns_copy_not_reference():
    """Modifying the returned array must not corrupt buffer state."""
    b = PYHistoryBuffer(size=3)
    b.push(np.array([1.0, 0.0]))
    s = b.samples()
    s[0][0] = 999.0
    assert b.samples()[0][0] == pytest.approx(1.0)


def test_reset_clears_state():
    b = PYHistoryBuffer(size=5)
    for i in range(10):
        b.push(np.array([float(i), 0.0]))
    b.reset()
    assert b.unstable is True
    assert b.count == 0
