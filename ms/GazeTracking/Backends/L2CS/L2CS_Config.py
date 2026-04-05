"""
GazeTracking/Backends/L2CS/L2CS_Config.py -- Configuration for the L2CS-Net backend.

L2CS-Net uses a ResNet backbone with dual classification branches (pitch + yaw).
Pre-trained weights are .pkl files (PyTorch state dicts) or .onnx exports.

References
----------
Paper  : https://arxiv.org/abs/2203.03339
Repo   : https://github.com/Ahmednull/L2CS-Net
"""
from ms.weights import resolve_weight

# Default model shipped with MindSight (resolved via Weights/L2CS/)
DEFAULT_MODEL = str(resolve_weight("L2CS", "L2CSNet_gaze360.pkl"))

# Architecture choices (ResNet variants supported by L2CS)
ARCH_CHOICES = [
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
]

# ResNet block configs for each architecture
ARCH_CONFIGS = {
    "ResNet18":  ("BasicBlock", [2, 2, 2, 2]),
    "ResNet34":  ("BasicBlock", [3, 4, 6, 3]),
    "ResNet50":  ("Bottleneck", [3, 4, 6, 3]),
    "ResNet101": ("Bottleneck", [3, 4, 23, 3]),
    "ResNet152": ("Bottleneck", [3, 8, 36, 3]),
}

# Dataset / bin configuration (same bin scheme as L2CS paper)
DATA_CONFIG = {
    "gaze360":  {"bins": 90, "binwidth": 4, "angle": 180},
    "mpiigaze": {"bins": 28, "binwidth": 3, "angle": 42},
}

# Input image size (L2CS uses 448x448 after transform)
INPUT_SIZE = 448
