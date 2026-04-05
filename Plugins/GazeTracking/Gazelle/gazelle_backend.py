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

import numpy as np
import torch

_GAZELLE_DIR = Path(__file__).parent / "gazelle"             # Plugins/GazeTracking/Gazelle/gazelle/ (3rd-party gazelle package)

from Plugins import GazePlugin

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
                 inout_threshold: float = 0.5,
                 skip_frames: int = 0,
                 use_fp16: bool = False,
                 use_compile: bool = False,
                 device: str = "auto") -> None:
        if model_name not in self._VALID_MODELS:
            raise ValueError(
                f"Unknown Gazelle model '{model_name}'. "
                f"Choose from {sorted(self._VALID_MODELS)}"
            )
        if str(_GAZELLE_DIR) not in sys.path:
            sys.path.insert(0, str(_GAZELLE_DIR))

        from gazelle.model import get_gazelle_model  # noqa: E402
        from PIL import Image as _PIL  # noqa: E402
        from torchvision import transforms  # noqa: E402

        self._PIL       = _PIL
        self._inout_thr = inout_threshold
        self._has_inout = model_name.endswith("_inout")

        # ── Device selection ─────────────────────────────────────────────
        from ms.utils.device import resolve_device
        self.device = resolve_device(device)

        model, _tf = get_gazelle_model(model_name)
        model.load_gazelle_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True)
        )
        self.model = model.to(self.device).eval()

        # ── Optimised transform: resize PIL first (fast C-level) ─────────
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── Half-precision ───────────────────────────────────────────────
        self._use_fp16 = use_fp16 and self.device.type in ("cuda", "mps")
        if self._use_fp16:
            self.model = self.model.half()

        # ── torch.compile ────────────────────────────────────────────────
        if use_compile:
            if not hasattr(torch, "compile"):
                import warnings
                warnings.warn("--gazelle-compile requires PyTorch 2.0+; ignoring.",
                              RuntimeWarning)
            else:
                self.model = torch.compile(self.model, mode="reduce-overhead")

        # ── Frame-skip cache ─────────────────────────────────────────────
        self._skip_frames       = skip_frames
        self._frame_counter     = 0
        self._cached_result     = None
        self._cached_bbox_count = 0

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
            self._cached_result = []
            self._cached_bbox_count = 0
            return []

        n_faces = len(face_bboxes_px)

        # ── Frame-skip: reuse cached result when possible ────────────────
        if (self._skip_frames > 0
                and self._cached_result is not None
                and self._frame_counter % (self._skip_frames + 1) != 0
                and n_faces == self._cached_bbox_count):
            self._frame_counter += 1
            return self._cached_result

        h, w = frame_bgr.shape[:2]

        # Zero-copy BGR→RGB via numpy view, then PIL
        pil  = self._PIL.fromarray(frame_bgr[:, :, ::-1])
        norm = [(x1/w, y1/h, x2/w, y2/h) for x1, y1, x2, y2 in face_bboxes_px]

        img_tensor = self.transform(pil).unsqueeze(0).to(self.device)
        if self._use_fp16:
            img_tensor = img_tensor.half()

        inp = {"images": img_tensor, "bboxes": [norm]}

        with torch.no_grad():
            out = self.model(inp)

        heatmaps = out["heatmap"][0]
        inout    = (
            out["inout"][0]
            if self._has_inout and out["inout"] is not None
            else None
        )

        # ── Batched heatmap extraction (single GPU→CPU sync) ────────────
        hm_flat = heatmaps.flatten(start_dim=1)          # [N, 4096]
        maxvals, argmaxes = hm_flat.max(dim=1)           # [N], [N]
        argmaxes_np = argmaxes.cpu().numpy()
        maxvals_np  = maxvals.cpu().numpy()
        inout_np    = inout.cpu().numpy() if inout is not None else None

        results = []
        for i in range(n_faces):
            idx  = int(argmaxes_np[i])
            xy   = np.array([idx % 64 / 64 * w, idx // 64 / 64 * h])
            conf = float(maxvals_np[i])
            if inout_np is not None and (s := float(inout_np[i])) < self._inout_thr:
                conf *= s
            results.append((xy, conf))

        self._cached_result     = results
        self._cached_bbox_count = n_faces
        self._frame_counter    += 1
        return results

    def raw_heatmaps(self, frame_bgr, face_bboxes_px: list) -> np.ndarray:
        """Run a Gazelle forward pass and return the raw heatmaps.

        Parameters
        ----------
        frame_bgr      : H×W×3 BGR numpy array.
        face_bboxes_px : list of (x1, y1, x2, y2) pixel bounding boxes.

        Returns
        -------
        numpy array of shape ``[N, 64, 64]`` with sigmoid-activated values
        in [0, 1].  Returns an empty ``(0, 64, 64)`` array when no faces
        are supplied.
        """
        if not face_bboxes_px:
            return np.empty((0, 64, 64), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        pil  = self._PIL.fromarray(frame_bgr[:, :, ::-1])
        norm = [(x1 / w, y1 / h, x2 / w, y2 / h)
                for x1, y1, x2, y2 in face_bboxes_px]

        img_tensor = self.transform(pil).unsqueeze(0).to(self.device)
        if self._use_fp16:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            out = self.model({"images": img_tensor, "bboxes": [norm]})

        return out["heatmap"][0].cpu().numpy()           # [N, 64, 64]

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
        g.add_argument(
            "--gazelle-device",
            default="auto", metavar="DEV",
            help=(
                "Compute device: auto, cpu, cuda, or mps.  "
                "'auto' selects CUDA > MPS > CPU  (default: auto)."
            ),
        )
        g.add_argument(
            "--gazelle-skip-frames",
            type=int, default=0, metavar="N",
            help=(
                "Reuse the previous gaze result for N frames between "
                "inference runs.  0 = no skipping  (default: 0)."
            ),
        )
        g.add_argument(
            "--gazelle-fp16",
            action="store_true", default=False,
            help="Use half-precision (float16) inference on CUDA/MPS (ignored on CPU).",
        )
        g.add_argument(
            "--gazelle-compile",
            action="store_true", default=False,
            help="Use torch.compile() for the Gazelle model (PyTorch 2.0+ only).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an initialized instance if ``--gazelle-model`` was given, else ``None``."""
        from ms.weights import resolve_weight
        ckpt = getattr(args, "gazelle_model", None)
        if not ckpt:
            return None
        ckpt = Path(resolve_weight("Gazelle", str(ckpt)))
        if not ckpt.exists():
            raise FileNotFoundError(f"Gazelle checkpoint not found: {ckpt}")
        name    = getattr(args, "gazelle_name", "gazelle_dinov2_vitb14")
        thr     = getattr(args, "gazelle_inout_threshold", 0.5)
        gazelle_dev = getattr(args, "gazelle_device", "auto")
        dev     = gazelle_dev if gazelle_dev != "auto" else getattr(args, "device", "auto")
        skip    = getattr(args, "gazelle_skip_frames", 0)
        fp16    = getattr(args, "gazelle_fp16", False)
        compile_ = getattr(args, "gazelle_compile", False)
        print(f"Backend: Gazelle  {name}")
        return cls(name, ckpt, thr,
                   skip_frames=skip, use_fp16=fp16,
                   use_compile=compile_, device=dev)


# ── Exported symbol consumed by PluginRegistry.discover() ─────────────────────
PLUGIN_CLASS = GazeEstimationGazelle
