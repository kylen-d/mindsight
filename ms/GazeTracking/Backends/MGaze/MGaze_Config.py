"""
GazeTracking/Backends/MGaze/MGaze_Config.py — Configuration for the MGaze gaze-estimation backend.

Contains dataset parameters, default model paths, and architecture choices
used by the MGaze ONNX and PyTorch estimation backends.
"""
from ms.weights import resolve_weight

# Default ONNX model (resolved via Weights/MGaze/)
DEFAULT_ONNX_MODEL = str(resolve_weight("MGaze", "mobileone_s0_gaze.onnx"))

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
