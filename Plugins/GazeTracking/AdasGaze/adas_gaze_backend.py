"""
Plugins/GazeTracking/AdasGaze/adas_gaze_backend.py — Intel adas-0002 gaze backend.

Head-pose-normalized per-face gaze estimation with Intel's
gaze-estimation-adas-0002 network (Apache-2.0, provenance-clean;
converted offline from the OpenVINO IR to ONNX and numerically verified
against the OpenVINO runtime).  Per face: MediaPipe 468-point landmarks
on the RetinaFace crop -> solvePnP head pose -> two 60x60 eye crops
(square, 1.8x the eye-corner distance, de-rolled) + head pose angles ->
gaze vector -> the MindSight pitch/yaw ray convention.

The preprocessing mirrors the OMZ gaze_estimation_demo with its default
roll alignment: eye crops are rotated so the eye line is horizontal, the
network sees roll = 0, and the predicted vector is rotated back.

Activation
----------
Pass ``--adas-gaze-model gaze-estimation-adas-0002.onnx`` (bare
filenames resolve through ``Weights/AdasGaze/``).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from Plugins import GazePlugin

_EYE_BOX_SCALE = 1.8            # OMZ createEyeBoundingBox default
_EYE_INPUT = 60                 # network input side


def _eye_crop(frame, corner_a, corner_b, roll_deg):
    """Square de-rolled 60x60 eye crop around the corner midpoint.

    Returns None when the corners collapse or the box falls outside the
    frame (the OMZ demo treats a zero-area eye box as a closed eye).
    """
    h, w = frame.shape[:2]
    size = float(np.linalg.norm(np.asarray(corner_a) - np.asarray(corner_b)))
    side = int(_EYE_BOX_SCALE * size)
    if side < 2:
        return None
    mx = int((corner_a[0] + corner_b[0]) / 2)
    my = int((corner_a[1] + corner_b[1]) / 2)
    x1, y1 = mx - side // 2, my - side // 2
    x2, y2 = x1 + side, y1 + side
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
    crop = frame[y1:y2, x1:x2]
    if roll_deg:
        ch, cw = crop.shape[:2]
        rot = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), float(roll_deg), 1.0)
        crop = cv2.warpAffine(crop, rot, (cw, ch), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (_EYE_INPUT, _EYE_INPUT))


class AdasGazeEngine:
    """(frame, bbox) -> (pitch_rad, yaw_rad, confidence) estimator."""

    def __init__(self, model_path, landmarker=None):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._out_name = self._sess.get_outputs()[0].name

        if landmarker is None:
            from mindsight.GazeTracking.normalized.landmarks import (
                CropFaceLandmarker,
            )
            landmarker = CropFaceLandmarker()
        self._landmarker = landmarker
        self._camera_matrix = None
        self._camera_shape = None

    def _camera_for(self, shape):
        if self._camera_shape != shape:
            h, w = shape
            self._camera_matrix = np.array(
                [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
            self._camera_shape = shape
        return self._camera_matrix

    def estimate_in_frame(self, frame_bgr, bbox):
        from mindsight.GazeTracking.normalized import (
            LEYE_INDICES,
            REYE_INDICES,
            estimate_head_pose,
            head_pose_angles_adas,
            vector_to_pipeline_pitchyaw,
        )

        landmarks = self._landmarker.detect(frame_bgr, bbox)
        if landmarks is None:
            return 0.0, 0.0, 0.0

        rot, _tvec, _model3d = estimate_head_pose(
            landmarks, self._camera_for(frame_bgr.shape[:2]))
        yaw_deg, pitch_deg, roll_deg = head_pose_angles_adas(rot)

        # Network naming is subject-relative: left_eye_image is the
        # subject's LEFT eye (image right; mediapipe LEYE indices).
        left = _eye_crop(frame_bgr, landmarks[LEYE_INDICES[0]],
                         landmarks[LEYE_INDICES[1]], roll_deg)
        right = _eye_crop(frame_bgr, landmarks[REYE_INDICES[0]],
                          landmarks[REYE_INDICES[1]], roll_deg)
        if left is None or right is None:
            return 0.0, 0.0, 0.0

        def _tensor(img):
            return img.astype(np.float32).transpose(2, 0, 1)[None]

        # Roll alignment: crops are de-rolled above, the net sees roll=0,
        # and the output vector is rotated back (OMZ demo default).
        g = self._sess.run([self._out_name], {
            "left_eye_image": _tensor(left),
            "right_eye_image": _tensor(right),
            "head_pose_angles": np.array(
                [[yaw_deg, pitch_deg, 0.0]], dtype=np.float32),
        })[0][0].astype(float)
        g /= max(np.linalg.norm(g), 1e-9)
        rr = np.radians(roll_deg)
        gx = g[0] * np.cos(rr) + g[1] * np.sin(rr)
        gy = -g[0] * np.sin(rr) + g[1] * np.cos(rr)

        # OMZ gaze frame (x image-right, y up, z toward camera) ->
        # camera frame (x right, y down, z away).
        g_cam = np.array([gx, -gy, -g[2]])
        pitch, yaw = vector_to_pipeline_pitchyaw(g_cam)
        return pitch, yaw, 1.0

    def estimate(self, face_bgr):
        """Crop-only fallback; the pipeline prefers estimate_in_frame."""
        h, w = face_bgr.shape[:2]
        return self.estimate_in_frame(face_bgr, (0, 0, w, h))


class AdasGazePlugin(GazePlugin):
    """Intel gaze-estimation-adas-0002 per-face gaze plugin."""

    name = "adas_gaze"
    mode = "per_face"
    is_fallback = False

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        return self._engine.estimate(face_bgr)

    def estimate_in_frame(self, frame_bgr, bbox):
        return self._engine.estimate_in_frame(frame_bgr, bbox)

    def run_pipeline(self, **kwargs):
        from mindsight.GazeTracking.pitchyaw_pipeline import (
            run_pitchyaw_pipeline,
        )
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    # ── CLI protocol ─────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("Adas Gaze backend")
        g.add_argument(
            "--adas-gaze-model", default=None, metavar="PATH",
            help=(
                "Path to the Intel gaze-estimation-adas-0002 ONNX model.  "
                "Activates the head-pose-normalized adas gaze backend "
                "(Apache-2.0, provenance-clean; requires the MediaPipe "
                "face_landmarker.task asset)."
            ),
        )

    @classmethod
    def from_args(cls, args):
        from mindsight.weights import resolve_weight
        model = getattr(args, "adas_gaze_model", None)
        if not model:
            return None
        path = Path(resolve_weight("AdasGaze", str(model)))
        if not path.exists():
            raise FileNotFoundError(
                f"adas gaze model not found: {path}\n"
                "Install it with: mindsight-weights --backend AdasGaze")
        print("Backend: AdasGaze ONNX  gaze-estimation-adas-0002")
        return cls(AdasGazeEngine(path))


PLUGIN_CLASS = AdasGazePlugin
