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

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from Plugins import GazePlugin

from .L2CS_Config import (  # noqa: E402
    ARCH_CHOICES,
    ARCH_CONFIGS,
    DATA_CONFIG,
    INPUT_SIZE,
)


# ==============================================================================
# Full-model pickle loader
# ==============================================================================

def _load_pkl_state_dict(weight_path, device, model_cls):
    """Load a .pkl that may be a full model pickle or a bare state dict.

    Some L2CS checkpoints are saved via ``torch.save(model, path)`` which
    embeds the original class reference (e.g. ``nets.L2CS``).  When that
    class is not importable, ``torch.load`` raises ``ModuleNotFoundError``.
    This helper catches that, injects a temporary stub module so the
    unpickler resolves the class to *model_cls*, then extracts the state
    dict.
    """
    import sys
    import types

    try:
        state = torch.load(weight_path, map_location=device, weights_only=False)
    except ModuleNotFoundError as exc:
        # Identify which module is missing (e.g. "nets", "l2cs", "model")
        missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc).split()[-1]
        stub = types.ModuleType(missing)
        # Map any class name the pickle references to our reimplementation
        for attr in ("L2CS", "L2CSModel", "L2CS_Model"):
            setattr(stub, attr, model_cls)
        sys.modules[missing] = stub
        try:
            state = torch.load(weight_path, map_location=device, weights_only=False)
        finally:
            sys.modules.pop(missing, None)

    # Unwrap checkpoint dicts (training saves wrap the state dict)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    # Handle full model objects saved via torch.save(model, path)
    if isinstance(state, nn.Module):
        state = state.state_dict()
    # Strip DataParallel 'module.' prefix if present
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state

# ==============================================================================
# Shared confidence helper
# ==============================================================================

def _softmax_confidence(pitch_probs_max, yaw_probs_max, n_bins):
    """Map average softmax peak from [1/n_bins, 1] onto [0, 1]."""
    uniform = 1.0 / n_bins
    return float(np.clip(((pitch_probs_max + yaw_probs_max) / 2 - uniform)
                         / (1 - uniform), 0, 1))


# ==============================================================================
# L2CS model definition (exact replica of upstream l2cs.model.L2CS)
# ==============================================================================

class _L2CSModel(nn.Module):
    """L2CS-Net: ResNet backbone with two FC classification heads (yaw, pitch).

    Exact replica of the upstream ``l2cs.model.L2CS`` class so state dicts
    (and full-model pickles) from any L2CS-Net release load without key
    mismatches.  Does not depend on the ``l2cs`` pip package.

    Reference: https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/model.py
    """

    def __init__(self, block, layers, num_bins):
        import math
        import torchvision.models.resnet as resnet_mod

        self.inplanes = 64
        super().__init__()
        block_cls = resnet_mod.BasicBlock if block == "BasicBlock" else resnet_mod.Bottleneck

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_cls, 64, layers[0])
        self.layer2 = self._make_layer(block_cls, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 512 * block_cls.expansion
        self.fc_yaw_gaze = nn.Linear(in_features, num_bins)
        self.fc_pitch_gaze = nn.Linear(in_features, num_bins)

        # Vestigial layer from the original L2CS training code; not used in
        # forward() but must exist so checkpoints containing it load cleanly.
        self.fc_finetune = nn.Linear(in_features + 3, 3)

        # Match upstream Kaiming initialization (overwritten by load_state_dict)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)
        yaw = self.fc_yaw_gaze(x)
        pitch = self.fc_pitch_gaze(x)
        return yaw, pitch


# ==============================================================================
# PyTorch estimation backend
# ==============================================================================

class _L2CSTorch:
    """PyTorch L2CS estimator -> (pitch_rad, yaw_rad, confidence).

    NOTE: L2CS uses the standard convention (pitch=vertical, yaw=horizontal)
    but the MindSight pipeline convention (inherited from MGaze) swaps them:
    the first return value controls the horizontal axis and the second
    controls the vertical axis.  The return order is therefore swapped to
    (yaw_rad, pitch_rad) so the shared pipeline projection is correct.
    """

    def __init__(self, weight_path, arch="ResNet50", dataset="gaze360", device="auto"):
        from mindsight.utils.device import resolve_device
        cfg = DATA_CONFIG[dataset]
        self._bins     = cfg["bins"]
        self._binwidth = cfg["binwidth"]
        self._angle    = cfg["angle"]

        self.device = resolve_device(device)
        self.idx_tensor = torch.arange(self._bins, dtype=torch.float32,
                                       device=self.device)

        block, layers = ARCH_CONFIGS[arch]
        model = _L2CSModel(block, layers, self._bins)
        state = _load_pkl_state_dict(weight_path, self.device, _L2CSModel)

        # strict=False: L2CS weights contain an extra fc_finetune layer
        # used during training that is not needed for inference.
        result = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(result.unexpected_keys)
        if loaded == 0:
            raise RuntimeError(
                f"No weights matched the model — checkpoint keys may be "
                f"incompatible. First 5 keys: {list(state.keys())[:5]}"
            )
        if result.missing_keys:
            print(f"  L2CS warning: {len(result.missing_keys)} model keys "
                  f"not found in checkpoint: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"  L2CS info: {len(result.unexpected_keys)} extra "
                  f"checkpoint keys ignored (expected for fc/fc_finetune)")
        print(f"  L2CS: loaded {loaded}/{loaded + len(result.missing_keys)} "
              f"model parameters from checkpoint")
        self.model = model.to(self.device).eval()

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
            yaw_logits, pitch_logits = self.model(t)

        pp = F.softmax(pitch_logits, dim=1)
        yp = F.softmax(yaw_logits, dim=1)

        to_rad = lambda p: float(np.radians(
            (torch.sum(p * self.idx_tensor) * self._binwidth - self._angle).item()))

        conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        # Swap: pipeline expects (horizontal, vertical) per MGaze convention
        return to_rad(yp), to_rad(pp), conf


# ==============================================================================
# ONNX estimation backend
# ==============================================================================

class _L2CSONNX:
    """ONNX L2CS estimator -> (pitch_rad, yaw_rad, confidence).

    Return order is swapped to match the MindSight pipeline convention
    (see _L2CSTorch docstring).
    """

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
        # Swap: pipeline expects (horizontal, vertical) per MGaze convention
        return np.radians(yaw_deg), np.radians(pitch_deg), conf


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
        from mindsight.GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
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
        from mindsight.weights import resolve_weight
        model = getattr(args, "l2cs_model", None)
        if not model:
            return None
        model = Path(resolve_weight("L2CS", str(model)))
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
