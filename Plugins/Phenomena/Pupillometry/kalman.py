"""
Plugins/Phenomena/Pupillometry/kalman.py -- 1D Kalman filter for pupil ratio.

Provides smoother, less laggy filtering than EMA for pupil/iris ratio
time-series, with automatic adaptation to measurement noise.
"""

from __future__ import annotations


class PupilKalman:
    """1D Kalman filter for pupil/iris ratio smoothing.

    Uses a constant-state model (pupil ratio changes slowly between frames)
    with tuneable process and measurement noise.
    """

    __slots__ = ('x', 'P', 'Q', 'R')

    def __init__(self, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2) -> None:
        self.x: float | None = None   # state estimate
        self.P: float = 1.0           # estimate uncertainty
        self.Q: float = process_noise
        self.R: float = measurement_noise

    def update(self, measurement: float) -> float:
        """Incorporate a new measurement and return the filtered estimate."""
        if self.x is None:
            self.x = measurement
            self.P = self.R
            return self.x

        # Predict (constant model: x_pred = x)
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

    def reset(self) -> None:
        """Reset filter state."""
        self.x = None
        self.P = 1.0
