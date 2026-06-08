"""
depth_backend.py -- DepthBackend protocol and factory.

Any depth estimation backend (monocular, stereo, depth camera) must satisfy
the ``DepthBackend`` protocol.  The ``create_depth_backend`` factory
instantiates the appropriate backend from a ``DepthConfig``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mindsight.pipeline_config import DepthConfig


@runtime_checkable
class DepthBackend(Protocol):
    """Protocol that all depth backends must satisfy."""

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an HxW float32 relative depth map (higher = farther)."""
        ...

    def warmup(self) -> None:
        """Pre-load model weights and run a dummy inference."""
        ...

    @property
    def supports_metric(self) -> bool:
        """True if backend produces metric (absolute) depth values."""
        ...


def create_depth_backend(cfg: DepthConfig, device: str = "auto") -> DepthBackend | None:
    """Instantiate a depth backend from config.

    Returns ``None`` if depth is disabled.
    """
    if not cfg.enabled:
        return None

    if cfg.backend == "midas_small":
        from mindsight.DepthEstimation.midas_backend import MiDaSBackend
        return MiDaSBackend(input_size=cfg.input_size, device=device)

    raise ValueError(f"Unknown depth backend: {cfg.backend!r}")
