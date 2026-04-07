"""
Plugins/GazeTracking/IrisRefinedGaze/iris_refined_gaze.py -- Iris-based gaze refinement.

Decorator-pattern GazePlugin that wraps any existing gaze backend and applies
iris-landmark-based corrections to its pitch/yaw estimates.  Activated only
when ``--iris-refine`` is passed; without it, the backend runs unmodified.

The iris center position relative to the eye socket center indicates actual
eye rotation independent of head pose.  This signal refines the backend's
coarse head-based gaze estimates, especially for fine eye movements within
a stable head pose.

Usage: ``--gaze-backend l2cs --iris-refine``
"""

from __future__ import annotations

import cv2
import numpy as np

from Plugins import GazePlugin


class IrisRefinedGaze(GazePlugin):
    """
    Wraps an inner gaze backend and blends iris-based corrections into its
    pitch/yaw output.
    """

    name = "iris_refined"
    mode = "per_face"
    is_fallback = False

    def __init__(self, inner_engine, *, weight: float = 0.3,
                 upscale: float = 2.0) -> None:
        self._inner = inner_engine
        self._weight = weight
        self._upscale = upscale

    def estimate(self, face_bgr):
        """Delegate to inner engine."""
        return self._inner.estimate(face_bgr)

    def run_pipeline(self, **kwargs):
        """
        Delegate to the inner backend's pipeline, then apply iris corrections.

        Returns the standard 7-tuple:
        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended)
        """
        frame = kwargs.get('frame')

        # Check if inner engine has run_pipeline
        inner_has_pipeline = (
            hasattr(self._inner, 'run_pipeline')
            and callable(self._inner.run_pipeline)
            and type(self._inner).run_pipeline is not GazePlugin.run_pipeline
        )

        if inner_has_pipeline:
            result = self._inner.run_pipeline(**kwargs)
        else:
            # Fallback: use default scene pipeline from gaze_pipeline module
            from ms.GazeTracking.gaze_pipeline import _default_scene_pipeline
            result = _default_scene_pipeline(
                frame, kwargs.get('faces', []), self._inner
            )

        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended) = result

        if frame is None or not persons_gaze:
            return result

        # Apply iris-based corrections
        refined_gaze = self._apply_iris_corrections(
            frame, persons_gaze, face_bboxes
        )

        return (refined_gaze, face_confs, face_bboxes, face_track_ids,
                face_objs, ray_snapped, ray_extended)

    def _apply_iris_corrections(self, frame, persons_gaze, face_bboxes):
        """Apply iris-position-based corrections to each face's gaze estimate."""
        try:
            from ms.utils.mediapipe_face import extract_iris_data
        except ImportError:
            return persons_gaze

        h, w = frame.shape[:2]
        refined = []

        for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
            if fi >= len(face_bboxes):
                refined.append((origin, ray_end, angles))
                continue

            # Get face crop
            x1, y1, x2, y2 = face_bboxes[fi]
            fx1, fy1 = max(0, int(x1)), max(0, int(y1))
            fx2, fy2 = min(w, int(x2)), min(h, int(y2))

            if fx2 - fx1 < 20 or fy2 - fy1 < 20:
                refined.append((origin, ray_end, angles))
                continue

            crop = frame[fy1:fy2, fx1:fx2]

            # Upscale for better iris detection
            if self._upscale != 1.0:
                ch, cw = crop.shape[:2]
                crop = cv2.resize(
                    crop,
                    (int(cw * self._upscale), int(ch * self._upscale)),
                    interpolation=cv2.INTER_CUBIC,
                )

            try:
                iris_data = extract_iris_data(crop)
            except (ImportError, AttributeError):
                # mediapipe not installed or incompatible version
                return persons_gaze
            if iris_data is None:
                refined.append((origin, ray_end, angles))
                continue

            # Compute iris offset from eye center (average of both eyes)
            offsets = []
            for side in ('right', 'left'):
                if not getattr(iris_data, f'{side}_valid'):
                    continue
                iris_c = getattr(iris_data, f'{side}_iris_center')
                eye_pts = getattr(iris_data, f'{side}_eye_contour')
                if iris_c is None or eye_pts is None:
                    continue
                eye_center = np.mean(eye_pts, axis=0)
                eye_width = np.linalg.norm(eye_pts[0] - eye_pts[1])
                if eye_width < 1:
                    continue
                # Normalize offset by eye width for scale invariance
                offset = (iris_c - eye_center) / eye_width
                offsets.append(offset)

            if not offsets:
                refined.append((origin, ray_end, angles))
                continue

            mean_offset = np.mean(offsets, axis=0)

            # Convert iris offset to ray correction
            # Horizontal offset -> yaw correction, vertical -> pitch correction
            # Scale by face width for pixel-space correction magnitude
            face_w = fx2 - fx1
            correction_px = mean_offset * face_w * self._weight

            new_ray_end = np.array([
                float(ray_end[0]) + correction_px[0],
                float(ray_end[1]) + correction_px[1],
            ])

            # Also correct angles if available
            new_angles = angles
            if angles is not None:
                pitch, yaw = float(angles[0]), float(angles[1])
                # Small angular corrections from normalized offset
                yaw_corr = mean_offset[0] * self._weight * 0.5
                pitch_corr = mean_offset[1] * self._weight * 0.5
                new_angles = (pitch + pitch_corr, yaw + yaw_corr)

            refined.append((origin, new_ray_end, new_angles))

        return refined

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        g = parser.add_argument_group("Iris-Refined Gaze")
        g.add_argument("--iris-refine", action="store_true",
                        help="Enable iris-based gaze refinement (wraps active backend).")
        g.add_argument("--iris-refine-weight", type=float, default=0.3, metavar="F",
                        help="Blending weight for iris correction (default: 0.3).")
        g.add_argument("--iris-refine-upscale", type=float, default=2.0, metavar="F",
                        help="Upscale face crops before iris extraction (default: 2.0).")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "iris_refine", False):
            return None

        # Load the inner backend by creating the engine without iris-refine
        # to avoid infinite recursion.
        from ms.GazeTracking.gaze_factory import create_gaze_engine

        # Temporarily disable iris_refine to get the actual backend
        args.iris_refine = False
        try:
            inner = create_gaze_engine(plugin_args=args)
        finally:
            args.iris_refine = True

        weight = getattr(args, "iris_refine_weight", 0.3)
        upscale = getattr(args, "iris_refine_upscale", 2.0)

        inst = cls(inner, weight=weight, upscale=upscale)
        print(
            f"IrisRefinedGaze: wrapping '{inner.name}'"
            f"  weight={weight}  upscale={upscale}x"
        )
        return inst


PLUGIN_CLASS = IrisRefinedGaze
