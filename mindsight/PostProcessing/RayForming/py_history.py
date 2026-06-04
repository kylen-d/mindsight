"""
RayForming/py_history.py -- Per-face pitch/yaw direction history buffer.

Ring buffer of the last K frames of ``py_dir`` (2D unit vector) samples.
Consumed by ``FixationDetector`` to compute smoothed velocity and
windowed dispersion for fixation-likelihood estimation.

Reports ``unstable=True`` until half the buffer has been populated so a
freshly-appeared track does not report a spurious fixation on the basis
of a single-frame history.
"""
from __future__ import annotations

import numpy as np


class PYHistoryBuffer:
    """Fixed-size ring buffer of 2D direction vectors.

    Parameters
    ----------
    size : int
        Number of samples retained.  Old samples are overwritten when
        the buffer is full.  Typical default: 10 frames (~ 1/3 sec at 30 fps).
    """

    def __init__(self, size: int = 10):
        if size < 2:
            raise ValueError(f"PYHistoryBuffer size must be >= 2; got {size}")
        self._size = int(size)
        self._buf: list[np.ndarray] = []

    def push(self, py_dir: np.ndarray) -> None:
        """Append a new direction sample.  Overwrites the oldest when full."""
        self._buf.append(np.asarray(py_dir, dtype=float).copy())
        if len(self._buf) > self._size:
            self._buf.pop(0)

    def samples(self) -> list[np.ndarray]:
        """Return copies of the buffered samples in insertion order."""
        return [s.copy() for s in self._buf]

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def unstable(self) -> bool:
        """True until at least half of the buffer is populated."""
        return len(self._buf) < (self._size // 2)

    def reset(self) -> None:
        self._buf.clear()
