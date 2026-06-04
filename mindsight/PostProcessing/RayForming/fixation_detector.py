"""
RayForming/fixation_detector.py -- Per-face fixation likelihood from PY signal.

Given a ``PYHistoryBuffer`` of pitch/yaw direction samples, produces a
continuous ``fixation_likelihood`` in [0, 1] that reflects whether the
participant is currently in a stable fixation (near 1) or in a saccade
or pursuit (near 0).

Two signals combine into the likelihood via a soft AND:
  - smoothed velocity ``v_smooth`` (rad/frame) -- low-pass of the PY
    signal strips per-frame jitter, so the resulting velocity reflects
    real motion, not sensor noise.
  - windowed dispersion ``d`` (rad) -- max angular deviation from the
    windowed mean direction across the buffer.  Catches slow drifts
    that velocity gating misses.

The likelihood function is
    v_fit = sigmoid(-(v_smooth - v_threshold) / v_scale)
    d_fit = sigmoid(-(d - d_threshold) / d_scale)
    likelihood = v_fit * d_fit

The product (soft AND) is smooth in both signals so no explicit hysteresis
is needed -- the continuous scalar itself provides the transition damping.
"""
from __future__ import annotations

import math

import numpy as np

from mindsight.PostProcessing.RayForming.py_history import PYHistoryBuffer


_JITTER_LOWPASS_ALPHA = 0.3
"""Fixed internal EMA rate for the velocity low-pass -- not user-tunable.

This is jitter defence, not a tuning axis.  A small alpha would over-
smooth and delay saccade detection; a large alpha would let per-frame
jitter through into the velocity estimate.  0.3 balances both well.
"""


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between 2D vectors, radians in [0, pi]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.acos(dot)


class FixationDetector:
    """Compute fixation_likelihood from a PYHistoryBuffer.

    Parameters
    ----------
    v_threshold : float
        Smoothed angular velocity at which v_fit = 0.5.  Radians / frame.
        Below this = fixating (v_fit -> 1); above = moving (v_fit -> 0).
    d_threshold : float
        Windowed dispersion at which d_fit = 0.5.  Radians.
    v_scale : float | None
        Softness of the velocity sigmoid.  Defaults to ``v_threshold * 0.25``.
    d_scale : float | None
        Softness of the dispersion sigmoid.  Defaults to ``d_threshold * 0.25``.
    """

    def __init__(self, *, v_threshold: float, d_threshold: float,
                 v_scale: float | None = None, d_scale: float | None = None):
        self.v_threshold = float(v_threshold)
        self.d_threshold = float(d_threshold)
        self.v_scale = float(v_scale if v_scale is not None else v_threshold * 0.25)
        self.d_scale = float(d_scale if d_scale is not None else d_threshold * 0.25)
        if self.v_threshold <= 0.0 or self.d_threshold <= 0.0:
            raise ValueError(
                f"FixationDetector thresholds must be positive; "
                f"got v_threshold={self.v_threshold}, d_threshold={self.d_threshold}")
        if self.v_scale <= 0.0 or self.d_scale <= 0.0:
            raise ValueError(
                f"FixationDetector scales must be positive; "
                f"got v_scale={self.v_scale}, d_scale={self.d_scale}")

    def update(self, buf: PYHistoryBuffer) -> float:
        """Read the buffer, return fixation_likelihood in [0, 1]."""
        if buf.unstable:
            return 0.0

        samples = buf.samples()
        # Low-pass the direction stream -- reduces per-frame jitter before
        # computing velocity, so real fixations with small-region jitter
        # do not falsely register as motion.
        smoothed: list[np.ndarray] = []
        acc: np.ndarray | None = None
        for s in samples:
            if acc is None:
                acc = s.copy()
            else:
                acc = _JITTER_LOWPASS_ALPHA * s + (1.0 - _JITTER_LOWPASS_ALPHA) * acc
                n = float(np.linalg.norm(acc))
                if n > 1e-9:
                    acc = acc / n
            smoothed.append(acc.copy())

        # Smoothed velocity = angle between last two smoothed samples.
        v_smooth = _angle_between(smoothed[-1], smoothed[-2]) if len(smoothed) >= 2 else 0.0

        # Windowed dispersion = max angle from the windowed mean direction.
        mean_vec = np.mean(np.stack(samples), axis=0)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 1e-9:
            mean_dir = mean_vec / mean_norm
        else:
            mean_dir = samples[-1]
        dispersion = max(_angle_between(s, mean_dir) for s in samples)

        # Soft AND via product of sigmoids.
        v_fit = _sigmoid(-(v_smooth - self.v_threshold) / self.v_scale)
        d_fit = _sigmoid(-(dispersion - self.d_threshold) / self.d_scale)
        return float(v_fit * d_fit)

    def reset(self) -> None:
        """No-op -- the detector recomputes from the buffer each update.

        Kept for API symmetry with PYHistoryBuffer.reset().
        """
