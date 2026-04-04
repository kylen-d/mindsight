"""
GazeTracking/Backends/MGaze/MGaze_Tracking.py — MGaze estimation backends (ONNX + PyTorch).

Wraps the gaze-estimation library to provide per-face gaze estimation
with confidence scoring.  Registers as the ``"mgaze"`` gaze plugin and
serves as the default/fallback backend when no other plugin is activated.

Activation
----------
Activated automatically via the default ``--mgaze-model`` flag (ONNX), or
explicitly with ``--mgaze-model /path/to/model.pt --mgaze-arch <arch>``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure the repo root is on sys.path so sibling package imports resolve.
_REPO_ROOT    = Path(__file__).parent.parent.parent.parent
_GAZE_EST_DIR = Path(__file__).parent / "gaze-estimation"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_GAZE_EST_DIR) not in sys.path:
    sys.path.insert(0, str(_GAZE_EST_DIR))

import utils_gaze.helpers  # noqa: E402
from onnx_inference import GazeEstimationONNX  # noqa: E402

from Plugins import GazePlugin  # noqa: E402

from .MGaze_Config import ARCH_CHOICES, DATA_CONFIG, DEFAULT_ONNX_MODEL  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Shared confidence helper
# ══════════════════════════════════════════════════════════════════════════════

def _softmax_confidence(pitch_probs_max, yaw_probs_max, n_bins):
    """Map average softmax peak from [1/n_bins, 1] onto [0, 1].

    Used by both PyTorch and ONNX backends so the formula stays in sync.
    """
    uniform = 1.0 / n_bins
    return float(np.clip(((pitch_probs_max + yaw_probs_max) / 2 - uniform)
                         / (1 - uniform), 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# Estimation backends
# ══════════════════════════════════════════════════════════════════════════════

class GazeEstimationTorch:
    """PyTorch gaze estimator -> (pitch_rad, yaw_rad, confidence)."""

    def __init__(self, weight_path, arch, dataset="gaze360", device="auto"):
        from utils.device import resolve_device
        cfg = DATA_CONFIG[dataset]
        self._bins, self._binwidth, self._angle = cfg["bins"], cfg["binwidth"], cfg["angle"]
        self.device     = resolve_device(device)
        self.idx_tensor = torch.arange(self._bins, dtype=torch.float32, device=self.device)
        model = utils_gaze.helpers.get_model(arch, self._bins, inference_mode=True)
        model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=False))
        self.model = model.to(self.device).eval()
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def estimate(self, face_bgr):
        t = self._tf(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pl, yl = self.model(t)
        pp, yp = F.softmax(pl, 1), F.softmax(yl, 1)
        to_rad = lambda p: float(np.radians(
            (torch.sum(p * self.idx_tensor) * self._binwidth - self._angle).item()))
        conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        return to_rad(pp), to_rad(yp), conf


class _GazeONNXWithConf(GazeEstimationONNX):
    """ONNX gaze backend extended with a confidence score."""

    def estimate(self, face_bgr):
        out        = self.session.run(self.output_names, {"input": self.preprocess(face_bgr)})
        pitch, yaw = self.decode(out[0], out[1])
        pp, yp     = self.softmax(out[0]), self.softmax(out[1])
        conf       = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        return pitch, yaw, conf


# ══════════════════════════════════════════════════════════════════════════════
# Plugin class
# ══════════════════════════════════════════════════════════════════════════════

class MGazePlugin(GazePlugin):
    """
    MGaze per-face gaze estimation plugin.

    Wraps either the ONNX or PyTorch gaze-estimation backend, selected
    automatically based on the model file extension.  Serves as the default
    fallback when no other gaze plugin is activated.
    """

    name = "mgaze"
    mode = "per_face"
    is_fallback = True

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        """Per-face estimation via the wrapped backend."""
        return self._engine.estimate(face_bgr)

    def run_pipeline(self, **kwargs):
        """Delegate to the generic pitch/yaw per-face pipeline."""
        from GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    # ── CLI protocol ─────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        """Register MGaze-specific CLI flags."""
        g = parser.add_argument_group("MGaze backend")
        g.add_argument("--mgaze-model", default=DEFAULT_ONNX_MODEL,
                        help="Path to MGaze model weights (.onnx or .pt)")
        g.add_argument("--mgaze-arch", default=None, choices=ARCH_CHOICES,
                        help="Architecture name (required for .pt models)")
        g.add_argument("--mgaze-dataset", default="gaze360",
                        help="Dataset config key (default: gaze360)")

    @classmethod
    def from_args(cls, args):
        """Create an MGaze engine from parsed CLI args."""
        model = getattr(args, "mgaze_model", None)
        if not model:
            return None
        model = Path(model)
        if not model.exists():
            raise FileNotFoundError(f"MGaze model not found: {model}")

        arch    = getattr(args, "mgaze_arch", None)
        dataset = getattr(args, "mgaze_dataset", "gaze360")

        if model.suffix.lower() == ".pt":
            if not arch:
                raise ValueError("--mgaze-arch is required for .pt models")
            device = getattr(args, "device", "auto")
            print(f"Backend: MGaze PyTorch  {arch}/{dataset}")
            engine = GazeEstimationTorch(str(model), arch, dataset, device=device)
        else:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prefs = [
                "CoreMLExecutionProvider", "CUDAExecutionProvider",
                "DirectMLExecutionProvider", "CPUExecutionProvider",
            ]
            prov = [p for p in prefs if p in avail] or ["CPUExecutionProvider"]
            print(f"Backend: MGaze ONNX  {prov}")
            engine = _GazeONNXWithConf(
                model_path=None,
                session=ort.InferenceSession(str(model), providers=prov),
            )
        return cls(engine)


# ── Exported symbol consumed by PluginRegistry.discover() ────────────────────
PLUGIN_CLASS = MGazePlugin
