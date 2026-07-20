"""
GazeTracking/Backends/MGaze/MGaze_Config.py — Configuration for the MGaze gaze-estimation backend.

Contains dataset parameters, default model paths, and architecture choices
used by the MGaze ONNX and PyTorch estimation backends.
"""
from mindsight.weights import resolve_weight

# Default ONNX model (resolved via Weights/MGaze/).  v1.1 W4C flip
# (ruling R7): resnet50 measures ~12px better than mobileone_s0 on
# every eval config (88px/48% vs 74.5px/62% standalone); mobileone_s0
# stays shipped as the low-power preset's pick.
DEFAULT_ONNX_MODEL = str(resolve_weight("MGaze", "resnet50_gaze.onnx"))

# Architecture choices for PyTorch (.pt) models
ARCH_CHOICES = [
    "resnet18", "resnet34", "resnet50", "mobilenetv2",
    "mobileone_s0", "mobileone_s1", "mobileone_s2",
    "mobileone_s3", "mobileone_s4",
]

# Dataset configuration (bin-based regression parameters)
DATA_CONFIG = {
    "gaze360":  {"bins": 90, "binwidth": 4, "angle": 180},
    "mpiigaze": {"bins": 28, "binwidth": 3, "angle": 42},
}
