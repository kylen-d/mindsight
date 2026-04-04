"""
GazeTracking/Backends/UniGaze/UniGaze_Tracking.py -- UniGaze gaze estimation.

Wraps UniGaze (ViT + MAE pre-training) for per-face gaze estimation.
Registers as the ``"unigaze"`` gaze plugin.

.. note::
    UniGaze was designed for **normalized face images** (perspective warp
    removing head rotation).  This backend currently uses direct face crops,
    which means the reported accuracy may be lower than the paper numbers.
    A full normalization pipeline using head-pose estimation and perspective
    warping can be added as a future enhancement.

Activation
----------
Activated via ``--unigaze-model <variant_name>``
(e.g. ``--unigaze-model unigaze_h14_joint``).

Requires ``pip install unigaze timm==0.3.2``.

References
----------
Paper : https://arxiv.org/abs/2502.02307  (WACV 2025)
Repo  : https://github.com/ut-vision/UniGaze
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import torch
import unigaze  # raises ImportError if not installed (caught by __init__.py)
from torchvision import transforms

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Plugins import GazePlugin  # noqa: E402

from .UniGaze_Config import (  # noqa: E402
    DEFAULT_VARIANT,
    INPUT_SIZE,
    MODEL_VARIANTS,
)

# ==============================================================================
# Estimation backend
# ==============================================================================

class _UniGazeTorch:
    """UniGaze PyTorch estimator -> (pitch_rad, yaw_rad, confidence)."""

    def __init__(self, variant=DEFAULT_VARIANT, device="auto"):
        from utils.device import resolve_device
        self.device = str(resolve_device(device))
        self.model = unigaze.load(variant, device=device)

        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def estimate(self, face_bgr):
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = self._tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(t)

        pred = output["pred_gaze"][0].cpu().numpy()  # (2,) -> [pitch, yaw]
        pitch_rad = float(pred[0])
        yaw_rad   = float(pred[1])

        # UniGaze does not output confidence; use a fixed high value
        # since the model is always confident in its prediction.
        conf = 0.85

        return pitch_rad, yaw_rad, conf


# ==============================================================================
# Plugin class
# ==============================================================================

class UniGazePlugin(GazePlugin):
    """
    UniGaze per-face gaze estimation plugin (optional, non-commercial).

    Requires the ``unigaze`` package to be installed separately.
    """

    name = "unigaze"
    mode = "per_face"
    is_fallback = False

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        return self._engine.estimate(face_bgr)

    def run_pipeline(self, **kwargs):
        from GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    # -- CLI protocol ----------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("UniGaze backend (optional, non-commercial)")
        g.add_argument("--unigaze-model", default=None,
                        choices=list(MODEL_VARIANTS.keys()),
                        help="UniGaze model variant (requires: pip install unigaze timm==0.3.2)")

    @classmethod
    def from_args(cls, args):
        variant = getattr(args, "unigaze_model", None)
        if not variant:
            return None
        print(f"Backend: UniGaze  {variant}")
        warnings.warn(
            "UniGaze currently uses direct face crops (no head-pose "
            "normalization).  Accuracy may be lower than paper numbers.",
            stacklevel=2,
        )
        device = getattr(args, "device", "auto")
        engine = _UniGazeTorch(variant=variant, device=device)
        return cls(engine)


PLUGIN_CLASS = UniGazePlugin
