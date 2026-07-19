"""Gazelle/gazelle_onnx_engine.py -- ONNX gaze-target engine (v1.1 W3Z).

Runs the PINTO0309/gazelle-dinov3 ONNX exports (Gaze-LLE successors on
DINOv3 / distilled HGNetV2 backbones) as a drop-in blend-path engine.
The exports embed their own preprocessing: input is a raw float32 BGR
image resized to the model's fixed square resolution, plus normalized
x1y1x2y2 face boxes; outputs are per-face heatmaps (32/48/64 square,
sigmoid-activated) and, on ``*_inout_*`` variants, a per-face in-frame
score.

Contract mirror of ``GazeEstimationGazelle.raw_heatmaps`` exactly:
``raw_heatmaps(frame_bgr, face_bboxes_px) -> [N, 64, 64] float32`` with
``self._last_inout`` set as a side effect -- so GazelleProvider, the
scheduler, the blender, and the W3Y length channel all work unchanged.
Sub-64 heatmaps are bilinearly resized to the blender's 64x64 grid.

Acceleration (measured on this project's clip hardware, 2026-07-18):
- CPU EP everywhere: atto-320 ~11 ms/call for two faces (vs ~88 ms for
  the torch DINOv2-vitb14 engine on MPS); pico-640 ~109 ms.
- Apple GPU via the CoreML EP works ONLY for the ViT-backbone variants
  and ONLY for the static ``*_1x1x4.onnx`` exports (dynamic-N and every
  HGNetV2 graph fail CoreML compilation): tinyplus 151 -> 48 ms/call,
  vits16 ~141 -> 69 ms/call per face (MLProgram, CPUAndGPU units; the
  NeuralEngine is slower than CPU for these).  Static single-face models
  are handled transparently by looping faces.
- NVIDIA: the CUDA EP runs these exports (upstream benchmarks them on
  TensorRT); it needs the onnxruntime-gpu package, which the default
  install does not carry -- absence falls back to CPU with a note.

Device mapping (from the global --device, no extra flag): cuda -> CUDA
EP, mps -> CoreML EP (Apple GPU), cpu -> CPU EP; auto resolves per
machine.  Unavailable/failing providers fall back to CPU with a printed
note, so a CNN-variant model on an Apple machine still loads.
"""
from __future__ import annotations

import cv2
import numpy as np

_COREML_OPTS = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}


def _providers_for_device(device: str) -> list:
    """Map a resolved device string to an EP preference list."""
    if device == "auto":
        try:
            from mindsight.utils.device import resolve_device
            device = str(getattr(resolve_device("auto"), "type", "cpu"))
        except Exception:
            device = "cpu"
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "mps":
        return [("CoreMLExecutionProvider", dict(_COREML_OPTS)),
                "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class GazelleOnnxEngine:
    """onnxruntime engine for gazelle-dinov3 ``*_1xNx4`` / ``*_1x1x4`` exports."""

    def __init__(self, model_path, device="cpu", providers=None, session=None):
        if session is not None:                 # test seam
            self.session = session
        else:
            self.session = self._make_session(
                model_path, providers or _providers_for_device(device))
        self._image_name = None
        self._bbox_name = None
        self._static_one_face = False
        for inp in self.session.get_inputs():
            if "bbox" in inp.name.lower():
                self._bbox_name = inp.name
                # [1, N, 4]: a static middle dim of 1 means a single-face
                # export (the CoreML-compatible kind) -- loop faces then.
                shape = getattr(inp, "shape", None) or []
                dim = shape[1] if len(shape) > 1 else None
                self._static_one_face = isinstance(dim, int) and dim == 1
            else:
                self._image_name = inp.name
                # [1, 3, H, W] with static H == W.
                self._size = int(inp.shape[2])
        if self._image_name is None or self._bbox_name is None:
            raise ValueError(
                f"{model_path}: expected an image input and a bbox input; "
                f"got {[i.name for i in self.session.get_inputs()]}")
        self._n_outputs = len(self.session.get_outputs())
        self._last_inout = None

    @staticmethod
    def _make_session(model_path, providers):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        available = set(ort.get_available_providers())
        wanted = [p for p in providers
                  if (p[0] if isinstance(p, tuple) else p) in available]
        dropped = [p[0] if isinstance(p, tuple) else p
                   for p in providers
                   if (p[0] if isinstance(p, tuple) else p) not in available]
        if dropped and dropped != ["CPUExecutionProvider"]:
            print(f"Gaze-LLE ONNX: provider(s) {dropped} not available in "
                  f"this onnxruntime build -- falling back "
                  f"(CUDA needs the onnxruntime-gpu package)")
        if not wanted:
            wanted = ["CPUExecutionProvider"]
        try:
            return ort.InferenceSession(str(model_path), so, providers=wanted)
        except Exception as exc:
            # e.g. CoreML cannot compile HGNetV2 or dynamic-N graphs.
            first = wanted[0][0] if isinstance(wanted[0], tuple) else wanted[0]
            if first == "CPUExecutionProvider":
                raise
            print(f"Gaze-LLE ONNX: {first} failed to load this model "
                  f"({str(exc)[:80]}...) -- using CPU")
            return ort.InferenceSession(
                str(model_path), so, providers=["CPUExecutionProvider"])

    def raw_heatmaps(self, frame_bgr, face_bboxes_px: list) -> np.ndarray:
        """Forward pass; returns ``[N, 64, 64]`` float32 heatmaps.

        Side effect: ``self._last_inout`` holds the per-face in-frame
        scores ([N] float array) on inout variants, else None -- same
        contract as the torch engine.
        """
        if not face_bboxes_px:
            self._last_inout = None
            return np.empty((0, 64, 64), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self._size, self._size))
        img = img.astype(np.float32).transpose(2, 0, 1)[None]   # BGR, 0-255
        norm = [[x1 / w, y1 / h, x2 / w, y2 / h]
                for x1, y1, x2, y2 in face_bboxes_px]

        if self._static_one_face:
            # Single-face export (CoreML-compatible): one call per face.
            hm_parts, io_parts = [], []
            for box in norm:
                boxes = np.array([[box]], dtype=np.float32)
                outs = self.session.run(
                    None, {self._image_name: img, self._bbox_name: boxes})
                hm_parts.append(np.asarray(outs[0], dtype=np.float32))
                if self._n_outputs > 1:
                    io_parts.append(
                        np.asarray(outs[1], dtype=np.float32).reshape(-1))
            heatmaps = np.concatenate(hm_parts, axis=0)
            inout = np.concatenate(io_parts) if io_parts else None
        else:
            boxes = np.array([norm], dtype=np.float32)
            outs = self.session.run(
                None, {self._image_name: img, self._bbox_name: boxes})
            heatmaps = np.asarray(outs[0], dtype=np.float32)    # [N, S, S]
            inout = (np.asarray(outs[1], dtype=np.float32).reshape(-1)
                     if self._n_outputs > 1 else None)

        if heatmaps.shape[-1] != 64:
            heatmaps = np.stack(
                [cv2.resize(hm, (64, 64), interpolation=cv2.INTER_LINEAR)
                 for hm in heatmaps]).astype(np.float32)
        self._last_inout = inout
        return heatmaps
