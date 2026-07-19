"""
GazeTracking/normalized/landmarks.py — 468-point landmarks on RetinaFace crops.

Wraps the MediaPipe FaceLandmarker tasks API (0.10.x ships no legacy
solutions) for the head-pose-normalized backends.  The landmarker's face
detector is SHORT-RANGE: on a full lab frame it finds zero faces, but it
is reliable on padded face crops — so this wrapper only ever runs on the
pipeline's existing RetinaFace boxes (padded, then mapped back to frame
coordinates).  Do not feed it full frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _create_landmarker(task_path: str, max_faces: int):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        num_faces=max_faces)
    return mp, vision.FaceLandmarker.create_from_options(options)


class CropFaceLandmarker:
    """468-point face landmarks from a frame + face bbox.

    Parameters
    ----------
    task_path : Path to ``face_landmarker.task``.  ``None`` resolves the
                manifest asset via the weights vault
                (``Weights/Mediapipe/face_landmarker.task``).
    max_faces : Faces the landmarker may return per crop.  Padded crops
                can catch a neighboring face, so keep this >= 2 and let
                ``detect`` pick the candidate nearest the requested box.
    """

    def __init__(self, task_path=None, max_faces: int = 2, _detector=None):
        if task_path is None:
            from mindsight.weights import resolve_weight
            task_path = resolve_weight("Mediapipe", "face_landmarker.task")
        task_path = Path(task_path)
        if _detector is not None:                 # test seam
            self._mp, self._detector = None, _detector
        else:
            if not task_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe landmarker asset not found: {task_path}\n"
                    "Install it with: mindsight-weights --backend Mediapipe")
            self._mp, self._detector = _create_landmarker(
                str(task_path), max_faces)

    @staticmethod
    def padded_crop_box(bbox, frame_w: int, frame_h: int):
        """Expand *bbox* by half its size on every side, clamped to the frame.

        The half-size padding (2x total crop extent) is what the W4B
        prototype validated: tight RetinaFace boxes crop away the chin
        and forehead the landmarker needs.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        pw, ph = x2 - x1, y2 - y1
        return (max(0, x1 - pw // 2), max(0, y1 - ph // 2),
                min(frame_w, x2 + pw // 2), min(frame_h, y2 + ph // 2))

    def detect(self, frame_bgr, bbox):
        """Landmark the face in *bbox*; return (468, 2) pixel coords or None.

        Runs the landmarker on the padded crop and maps the result back
        to frame coordinates.  When the padded crop catches more than one
        face, the candidate whose center lies closest to the center of
        the REQUESTED box wins.
        """
        h, w = frame_bgr.shape[:2]
        cx1, cy1, cx2, cy2 = self.padded_crop_box(bbox, w, h)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            return None
        crop = np.ascontiguousarray(frame_bgr[cy1:cy2, cx1:cx2, ::-1])
        if self._mp is not None:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=crop)
        else:
            image = crop                          # test seam
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None

        ch, cw = crop.shape[:2]
        candidates = [
            np.array([[cx1 + lm.x * cw, cy1 + lm.y * ch]
                      for lm in lms[:468]])
            for lms in result.face_landmarks
        ]
        target = np.array([(bbox[0] + bbox[2]) / 2.0,
                           (bbox[1] + bbox[3]) / 2.0])
        return min(candidates,
                   key=lambda pts: float(
                       np.linalg.norm(pts.mean(axis=0) - target)))

    def close(self):
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
