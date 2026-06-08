"""
midas_backend.py -- MiDaS v2.1 Small depth estimation backend.

Uses ``torch.hub`` to load the lightweight MiDaS small model.  The raw
output is *inverse* depth (higher = nearer), which is inverted so the
returned map follows the protocol convention (higher = farther).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class MiDaSBackend:
    """Monocular depth estimation via MiDaS v2.1 Small."""

    def __init__(self, input_size: int = 384, device: str = "auto"):
        self._input_size = input_size
        self._device_str = device
        self._model = None
        self._transform = None
        self._device = None

    # -- Protocol properties ---------------------------------------------------

    @property
    def supports_metric(self) -> bool:
        return False

    # -- Lifecycle -------------------------------------------------------------

    def warmup(self) -> None:
        """Eagerly load the model and run a dummy inference."""
        self._ensure_loaded()
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self.estimate(dummy)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._device = _resolve_device(self._device_str)
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True,
        )
        self._model.to(self._device).eval()
        self._transform = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True,
        ).small_transform

    # -- Inference -------------------------------------------------------------

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an HxW float32 depth map (higher = farther).

        The input is a BGR frame at any resolution.  It is converted to RGB,
        passed through the MiDaS transform (which handles resizing), and the
        raw inverse-depth output is inverted so that larger values represent
        greater distance from the camera.
        """
        self._ensure_loaded()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(frame_rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_tensor)
            # Interpolate to input frame resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        inverse_depth = prediction.cpu().numpy().astype(np.float32)
        # Invert: MiDaS outputs higher=nearer, we need higher=farther.
        # Guard against division by zero in flat regions.
        max_val = inverse_depth.max()
        if max_val > 1e-6:
            depth = max_val - inverse_depth
        else:
            depth = np.zeros_like(inverse_depth)
        return depth


def _resolve_device(device_str: str) -> torch.device:
    """Map MindSight's device string to a torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
