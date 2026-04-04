"""
GazeTracking/Backends/UniGaze/UniGaze_Config.py -- Configuration for UniGaze.

UniGaze uses ViT backbones with MAE pre-training.  Weights are hosted on
HuggingFace and auto-downloaded on first use.

License: Non-commercial (ModelGo NC-RAI-2.0).  The ``unigaze`` package must
be installed separately by the user.

References
----------
Paper : https://arxiv.org/abs/2502.02307  (WACV 2025)
Repo  : https://github.com/ut-vision/UniGaze
"""

# Available model variants (name -> HuggingFace model ID)
MODEL_VARIANTS = {
    "unigaze_b16_joint": "ViT-Base (fastest, ~4 deg MPIIGaze)",
    "unigaze_l16_joint": "ViT-Large (balanced)",
    "unigaze_h14_joint": "ViT-Huge (best accuracy, ~4 deg MPIIGaze, ~9.4 deg Gaze360)",
}

DEFAULT_VARIANT = "unigaze_h14_joint"

# Input image size expected by UniGaze models
INPUT_SIZE = 224
