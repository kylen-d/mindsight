"""
Plugins/GazeTracking/Gazelle/gazelle_backend.py — Gazelle (Gaze-LLE) backend plugin.

Registers the scene-level Gazelle estimator under the name ``"gazelle"``
in the gaze plugin registry.

Supported model variants
------------------------
    gazelle_dinov2_vitb14
    gazelle_dinov2_vitb14_inout
    gazelle_dinov2_vitl14
    gazelle_dinov2_vitl14_inout

Activation
----------
Pass ``--gazelle-model /path/to/checkpoint.pt`` on the command line.
Optionally pair with ``--gazelle-name`` and ``--gazelle-inout-threshold``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure the repo root is on sys.path so sibling package imports resolve when
# this module is loaded by the PluginRegistry discovery loop.
_REPO_ROOT   = Path(__file__).parent.parent.parent.parent   # Plugins/GazeTracking/Gazelle/ → repo root
_GAZELLE_DIR = _REPO_ROOT / "gazelle"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Plugins import GazePlugin  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Plugin class
# ══════════════════════════════════════════════════════════════════════════════

class GazeEstimationGazelle(GazePlugin):
    """
    Gazelle (Gaze-LLE) scene-level gaze estimator.

    All faces in the supplied frame are processed in a single DinoV2
    forward pass, returning a per-face gaze-point heatmap.

    Parameters
    ----------
    model_name      : One of the four supported Gazelle model variants.
    ckpt_path       : Path to the ``*.pt`` checkpoint file.
    inout_threshold : For ``*_inout`` models: faces whose in/out-of-view score
                      falls below this value have their heatmap confidence
                      attenuated proportionally.  No effect on non-inout models.
    """

    name = "gazelle"
    mode = "scene"

    _VALID_MODELS = {
        "gazelle_dinov2_vitb14",
        "gazelle_dinov2_vitl14",
        "gazelle_dinov2_vitb14_inout",
        "gazelle_dinov2_vitl14_inout",
    }

    def __init__(self, model_name: str, ckpt_path: str | Path,
                 inout_threshold: float = 0.5) -> None:
        if model_name not in self._VALID_MODELS:
            raise ValueError(
                f"Unknown Gazelle model '{model_name}'. "
                f"Choose from {sorted(self._VALID_MODELS)}"
            )
        if str(_GAZELLE_DIR) not in sys.path:
            sys.path.insert(0, str(_GAZELLE_DIR))

        from gazelle.model import get_gazelle_model  # noqa: E402
        from PIL import Image as _PIL                # noqa: E402

        self._PIL       = _PIL
        self._inout_thr = inout_threshold
        self._has_inout = model_name.endswith("_inout")
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, tf = get_gazelle_model(model_name)
        model.load_gazelle_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model     = model.to(self.device).eval()
        self.transform = tf

    # ── Estimation ────────────────────────────────────────────────────────────

    def estimate_frame(self, frame_bgr, face_bboxes_px: list) -> list:
        """
        Estimate gaze for all faces in a single forward pass.

        Parameters
        ----------
        frame_bgr      : H×W×3 BGR numpy array (full scene frame).
        face_bboxes_px : list of (x1, y1, x2, y2) pixel bounding boxes.

        Returns
        -------
        list of (gaze_xy_px, confidence)
            One entry per input bounding box.  ``gaze_xy_px`` is a float numpy
            array ``[x, y]`` in pixel coordinates of the estimated gaze target.
        """
        if not face_bboxes_px:
            return []

        h, w = frame_bgr.shape[:2]
        pil  = self._PIL.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        norm = [(x1/w, y1/h, x2/w, y2/h) for x1, y1, x2, y2 in face_bboxes_px]
        inp  = {
            "images": self.transform(pil).unsqueeze(0).to(self.device),
            "bboxes": [norm],
        }

        with torch.no_grad():
            out = self.model(inp)

        heatmaps = out["heatmap"][0]
        inout    = (
            out["inout"][0]
            if self._has_inout and out["inout"] is not None
            else None
        )

        results = []
        for i, hm in enumerate(heatmaps):
            idx  = int(hm.flatten().argmax())
            xy   = np.array([idx % 64 / 64 * w, idx // 64 / 64 * h])
            conf = float(hm.max())
            if inout is not None and (s := float(inout[i])) < self._inout_thr:
                conf *= s
            results.append((xy, conf))

        return results

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Gazelle-specific CLI flags to *parser*."""
        g = parser.add_argument_group("Gazelle backend")
        g.add_argument(
            "--gazelle-model",
            default=None, metavar="PATH",
            help="Path to a Gazelle checkpoint (.pt).  Activates the Gazelle backend.",
        )
        g.add_argument(
            "--gazelle-name",
            default="gazelle_dinov2_vitb14",
            choices=sorted(cls._VALID_MODELS),
            metavar="NAME",
            help=(
                "Gazelle model variant  "
                f"(choices: {', '.join(sorted(cls._VALID_MODELS))};  "
                "default: gazelle_dinov2_vitb14)."
            ),
        )
        g.add_argument(
            "--gazelle-inout-threshold",
            type=float, default=0.5, metavar="F",
            help=(
                "In/out-of-view confidence threshold for *_inout model variants.  "
                "No effect on non-inout models  (default: 0.5)."
            ),
        )

    @classmethod
    def from_args(cls, args):
        """Return an initialized instance if ``--gazelle-model`` was given, else ``None``."""
        ckpt = getattr(args, "gazelle_model", None)
        if not ckpt:
            return None
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Gazelle checkpoint not found: {ckpt}")
        name = getattr(args, "gazelle_name", "gazelle_dinov2_vitb14")
        thr  = getattr(args, "gazelle_inout_threshold", 0.5)
        print(f"Backend: Gazelle  {name}")
        return cls(name, ckpt, thr)


# ── Exported symbol consumed by PluginRegistry.discover() ─────────────────────
PLUGIN_CLASS = GazeEstimationGazelle
