"""
utils/one_euro.py -- One Euro Filter for adaptive smoothing of noisy input.

Reference: Casiez, Roussel, Vogel "1 Euro Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012).

The filter's cutoff frequency adapts to instantaneous speed: aggressive
smoothing when the signal is slow (jitter suppression), light smoothing
when the signal is fast (no lag on real motion).  Two intuitive knobs:
``min_cutoff`` (the floor cutoff, Hz) and ``beta`` (how aggressively
the cutoff opens on speed, unitless).

MindSight uses this as the output smoother for the Gaze-LLE blender's
direction and length channels, replacing the previous fixed-alpha EMA
that forced a hard trade-off between latency and jitter.
"""
from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    """Convert a cutoff frequency (Hz) and sample interval (s) to an
    EMA alpha in (0, 1]."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    """Standard first-order low-pass filter with settable per-frame alpha."""

    def __init__(self):
        self._y_prev: float | None = None

    def update(self, x: float, alpha: float) -> float:
        if self._y_prev is None:
            y = x
        else:
            y = alpha * x + (1.0 - alpha) * self._y_prev
        self._y_prev = y
        return y

    @property
    def last(self) -> float | None:
        return self._y_prev


class OneEuroFilter:
    """One Euro Filter for a scalar signal.

    Parameters
    ----------
    min_cutoff : float
        Floor cutoff frequency (Hz).  Lower = smoother at rest.
    beta : float
        Speed coefficient.  Higher = more responsive to fast motion.
    d_cutoff : float
        Cutoff for the derivative low-pass (Hz).  Internal; usually 1.0.
    dt : float
        Sample interval (seconds).  In MindSight, ``1.0 / fps`` from the
        video source; passed at construction and reused for every call.
    """

    def __init__(self, min_cutoff: float, beta: float,
                 d_cutoff: float = 1.0, dt: float = 1.0 / 30.0):
        if dt <= 0.0:
            raise ValueError(f"OneEuroFilter dt must be positive; got {dt}")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dt = float(dt)
        self._x_filter = _LowPass()
        self._dx_filter = _LowPass()
        self._x_prev: float | None = None

    def update(self, x: float) -> float:
        """Push one sample; return the filtered value."""
        if self._x_prev is None:
            self._x_prev = x
            # Prime the value filter with x at the min_cutoff alpha so
            # subsequent updates have a well-defined predecessor.
            alpha_x = _alpha(self.min_cutoff, self.dt)
            return self._x_filter.update(x, alpha_x)

        # Estimate the derivative and low-pass it at d_cutoff.
        dx = (x - self._x_prev) / self.dt
        alpha_d = _alpha(self.d_cutoff, self.dt)
        edx = self._dx_filter.update(dx, alpha_d)

        # Adaptive cutoff and value low-pass.
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha_x = _alpha(cutoff, self.dt)
        y = self._x_filter.update(x, alpha_x)

        self._x_prev = x
        return y

    def reset(self) -> None:
        """Clear internal state (for track eviction)."""
        self._x_filter = _LowPass()
        self._dx_filter = _LowPass()
        self._x_prev = None
