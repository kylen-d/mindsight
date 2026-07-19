"""Gazelle/gazelle_onnx_engine.py -- ONNX gaze-target engine (v1.1 W3Z).

Runs the PINTO0309/gazelle-dinov3 ONNX exports (Gaze-LLE successors on
DINOv3 / distilled HGNetV2 backbones) as a drop-in blend-path engine.
The exports embed their own preprocessing: input is a raw float32 BGR
image resized to the model's fixed square resolution, plus normalized
``[1, N, 4]`` x1y1x2y2 face boxes; outputs are per-face heatmaps
(32/48/64 square, sigmoid-activated) and, on ``*_inout_*`` variants, a
per-face in-frame score.

Contract mirror of ``GazeEstimationGazelle.raw_heatmaps`` exactly:
``raw_heatmaps(frame_bgr, face_bboxes_px) -> [N, 64, 64] float32`` with
``self._last_inout`` set as a side effect -- so GazelleProvider, the
scheduler, the blender, and the W3Y length channel all work unchanged.
Sub-64 heatmaps are bilinearly resized to the blender's 64x64 grid.

Measured on this project's 869-frame clip hardware (Apple, CPU execution
provider -- the CoreML EP rejects these exports): the atto-320 variant
runs ~11 ms/call for two faces vs ~88 ms for the torch DINOv2-vitb14
engine on MPS.
"""
from __future__ import annotations

import cv2
import numpy as np


class GazelleOnnxEngine:
    """onnxruntime engine for gazelle-dinov3 ``*_1xNx4.onnx`` exports."""

    def __init__(self, model_path, providers=None, session=None):
        if session is not None:                 # test seam
            self.session = session
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.session = ort.InferenceSession(
                str(model_path), so,
                providers=providers or ["CPUExecutionProvider"])
        self._image_name = None
        self._bbox_name = None
        for inp in self.session.get_inputs():
            if "bbox" in inp.name.lower():
                self._bbox_name = inp.name
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
        boxes = np.array(
            [[[x1 / w, y1 / h, x2 / w, y2 / h]
              for x1, y1, x2, y2 in face_bboxes_px]], dtype=np.float32)

        outs = self.session.run(
            None, {self._image_name: img, self._bbox_name: boxes})
        heatmaps = np.asarray(outs[0], dtype=np.float32)        # [N, S, S]
        if heatmaps.shape[-1] != 64:
            heatmaps = np.stack(
                [cv2.resize(hm, (64, 64), interpolation=cv2.INTER_LINEAR)
                 for hm in heatmaps]).astype(np.float32)
        self._last_inout = (
            np.asarray(outs[1], dtype=np.float32).reshape(-1)
            if self._n_outputs > 1 else None)
        return heatmaps
