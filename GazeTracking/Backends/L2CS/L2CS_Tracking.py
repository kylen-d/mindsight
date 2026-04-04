"""
GazeTracking/Backends/L2CS/L2CS_Tracking.py -- L2CS-Net gaze estimation backends.

Wraps L2CS-Net (ResNet + dual classification heads for pitch/yaw) to provide
per-face gaze estimation with confidence scoring.  Registers as the ``"l2cs"``
gaze plugin and is preferred over the MGaze fallback backend.

Activation
----------
Activated via ``--l2cs-model /path/to/L2CSNet_gaze360.pkl`` (PyTorch) or
``--l2cs-model /path/to/l2cs_gaze360.onnx`` (ONNX).

References
----------
Paper : https://arxiv.org/abs/2203.03339
Repo  : https://github.com/Ahmednull/L2CS-Net
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Ensure the repo root is on sys.path so sibling package imports resolve.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Plugins import GazePlugin  # noqa: E402

from .L2CS_Config import (  # noqa: E402
    ARCH_CHOICES,
    ARCH_CONFIGS,
    DATA_CONFIG,
    INPUT_SIZE,
)

# ==============================================================================
# Shared confidence helper
# ==============================================================================

def _softmax_confidence(pitch_probs_max, yaw_probs_max, n_bins):
    """Map average softmax peak from [1/n_bins, 1] onto [0, 1]."""
    uniform = 1.0 / n_bins
    return float(np.clip(((pitch_probs_max + yaw_probs_max) / 2 - uniform)
                         / (1 - uniform), 0, 1))


# ==============================================================================
# L2CS model definition (self-contained -- no external l2cs package required)
# ==============================================================================

class _L2CSModel(nn.Module):
    """L2CS-Net: ResNet backbone with two FC classification heads (yaw, pitch).

    This is a self-contained reimplementation so the backend does not depend
    on the upstream ``l2cs`` pip package.  Weights are fully compatible.
    """

    def __init__(self, block, layers, num_bins):
        super().__init__()
        import torchvision.models.resnet as resnet_mod
        block_cls = resnet_mod.BasicBlock if block == "BasicBlock" else resnet_mod.Bottleneck

        # Build a standard ResNet trunk (minus the final FC)
        base = resnet_mod.ResNet(block_cls, layers)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        expansion = block_cls.expansion
        in_features = 512 * expansion

        # Two independent classification heads
        self.fc_yaw_gaze   = nn.Linear(in_features, num_bins)
        self.fc_pitch_gaze = nn.Linear(in_features, num_bins)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        yaw   = self.fc_yaw_gaze(x)
        pitch = self.fc_pitch_gaze(x)
        return yaw, pitch


# ==============================================================================
# PyTorch estimation backend
# ==============================================================================

class _L2CSTorch:
    """PyTorch L2CS estimator -> (pitch_rad, yaw_rad, confidence)."""

    def __init__(self, weight_path, arch="ResNet50", dataset="gaze360", device="auto"):
        from utils.device import resolve_device
        cfg = DATA_CONFIG[dataset]
        self._bins     = cfg["bins"]
        self._binwidth = cfg["binwidth"]
        self._angle    = cfg["angle"]

        self.device = resolve_device(device)
        self.idx_tensor = torch.arange(self._bins, dtype=torch.float32,
                                       device=self.device)

        block, layers = ARCH_CONFIGS[arch]
        model = _L2CSModel(block, layers, self._bins)
        state = torch.load(weight_path, map_location=self.device, weights_only=False)
        # strict=False: L2CS weights contain an extra fc_finetune layer
        # used during training that is not needed for inference.
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device).eval()

        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def estimate(self, face_bgr):
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = self._tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            yaw_logits, pitch_logits = self.model(t)

        pp = F.softmax(pitch_logits, dim=1)
        yp = F.softmax(yaw_logits, dim=1)

        to_rad = lambda p: float(np.radians(
            (torch.sum(p * self.idx_tensor) * self._binwidth - self._angle).item()))

        conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        return to_rad(pp), to_rad(yp), conf


# ==============================================================================
# ONNX estimation backend
# ==============================================================================

class _L2CSONNX:
    """ONNX L2CS estimator -> (pitch_rad, yaw_rad, confidence)."""

    def __init__(self, session, dataset="gaze360"):
        cfg = DATA_CONFIG[dataset]
        self._bins     = cfg["bins"]
        self._binwidth = cfg["binwidth"]
        self._angle    = cfg["angle"]
        self._session  = session
        self._input_name  = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def preprocess(self, face_bgr):
        img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 448, 448)

    def estimate(self, face_bgr):
        inp = self.preprocess(face_bgr)
        outputs = self._session.run(self._output_names,
                                    {self._input_name: inp})
        # L2CS outputs: [yaw_logits, pitch_logits]
        yaw_logits, pitch_logits = outputs[0], outputs[1]

        pp = self.softmax(pitch_logits)
        yp = self.softmax(yaw_logits)

        idx = np.arange(self._bins, dtype=np.float32)

        pitch_deg = float(np.sum(pp * idx) * self._binwidth - self._angle)
        yaw_deg   = float(np.sum(yp * idx) * self._binwidth - self._angle)

        conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        return np.radians(pitch_deg), np.radians(yaw_deg), conf


# ==============================================================================
# Plugin class
# ==============================================================================

class L2CSPlugin(GazePlugin):
    """
    L2CS-Net per-face gaze estimation plugin.

    Wraps either the PyTorch or ONNX L2CS-Net backend, selected automatically
    based on the model file extension.  Preferred over the MGaze fallback
    backend when activated.
    """

    name = "l2cs"
    mode = "per_face"
    is_fallback = False

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        """Per-face estimation via the wrapped backend."""
        return self._engine.estimate(face_bgr)

    def run_pipeline(self, **kwargs):
        """Delegate to the generic pitch/yaw per-face pipeline."""
        from GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    # -- CLI protocol ----------------------------------------------------------

    @classmethod
    def add_arguments(cls, parser):
        """Register L2CS-specific CLI flags."""
        g = parser.add_argument_group("L2CS-Net backend")
        g.add_argument("--l2cs-model", default=None,
                        help="Path to L2CS model weights (.pkl or .onnx)")
        g.add_argument("--l2cs-arch", default="ResNet50",
                        choices=ARCH_CHOICES,
                        help="Architecture (default: ResNet50)")
        g.add_argument("--l2cs-dataset", default="gaze360",
                        help="Dataset config key (default: gaze360)")

    @classmethod
    def from_args(cls, args):
        """Create an L2CS engine from parsed CLI args."""
        model = getattr(args, "l2cs_model", None)
        if not model:
            return None
        model = Path(model)
        if not model.exists():
            raise FileNotFoundError(f"L2CS model not found: {model}")

        arch    = getattr(args, "l2cs_arch", "ResNet50")
        dataset = getattr(args, "l2cs_dataset", "gaze360")

        device = getattr(args, "device", "auto")

        if model.suffix.lower() in (".pkl", ".pt", ".pth"):
            print(f"Backend: L2CS-Net PyTorch  {arch}/{dataset}")
            engine = _L2CSTorch(str(model), arch, dataset, device=device)
        else:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prefs = [
                "CoreMLExecutionProvider", "CUDAExecutionProvider",
                "DirectMLExecutionProvider", "CPUExecutionProvider",
            ]
            prov = [p for p in prefs if p in avail] or ["CPUExecutionProvider"]
            print(f"Backend: L2CS-Net ONNX  {prov}")
            engine = _L2CSONNX(
                session=ort.InferenceSession(str(model), providers=prov),
                dataset=dataset,
            )
        return cls(engine)


# -- Exported symbol consumed by PluginRegistry.discover() ---------------------
PLUGIN_CLASS = L2CSPlugin
