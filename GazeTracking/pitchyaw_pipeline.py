"""
GazeTracking/pitchyaw_pipeline.py — Generic per-face pitch/yaw gaze pipeline.

Handles face cropping, per-face estimation, temporal smoothing, ray
construction, and adaptive-snap logic for any per-face pitch/yaw gaze backend.

Any backend that implements ``estimate(face_bgr) -> (pitch, yaw, confidence)``
can use this pipeline directly via its ``run_pipeline()`` method::

    def run_pipeline(self, **kwargs):
        from GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)
"""
import numpy as np

from constants import CR_MAX, CR_MIN, EYE_CONF_THRESH
from GazeTracking.gaze_processing import (
    _faces_as_objects,
    _get_eye_center,
    adaptive_snap,
)
from utils.geometry import pitch_yaw_to_2d


def run_pitchyaw_pipeline(*, frame, faces, gaze_eng, objects, gaze_cfg,
                          smoother=None, snap_hysteresis=None, **kwargs):
    """
    Generic per-face pitch/yaw gaze estimation pipeline step.

    Works with any gaze backend that provides an
    ``estimate(face_bgr) -> (pitch_rad, yaw_rad, confidence)`` method.

    Parameters
    ----------
    frame           : BGR numpy array at display resolution.
    faces           : List of face detection dicts (from RetinaFace).
    gaze_eng        : Gaze estimation engine with estimate(crop) method.
    objects         : Non-person detection list.
    gaze_cfg        : GazeConfig with ray parameters.
    smoother        : Optional GazeSmootherReID instance.
    snap_hysteresis : Optional SnapHysteresisTracker instance.
    **kwargs        : Additional context (ignored — future-proofing).

    Returns
    -------
    persons_gaze    : list of (origin, ray_end, (pitch, yaw)).
    face_confs      : list[float] per-face gaze confidence.
    face_bboxes     : list of (x1, y1, x2, y2) in display pixels.
    face_track_ids  : list[int] stable smoother track IDs.
    face_objs       : list[Detection] face-as-object targets.
    ray_snapped     : list[bool] per-face adaptive-snap flag.
    ray_extended    : list[bool] per-face ray-extension flag.
    """
    h, w = frame.shape[:2]
    face_confs:  list = []
    face_bboxes: list = []

    # -- Per-face estimation --------------------------------------------------
    raw_faces, face_widths, gaze_confs, raw_face_bboxes = [], [], [], []
    for f in faces:
        x1, y1 = max(0, int(f["bbox"][0])), max(0, int(f["bbox"][1]))
        x2, y2 = min(w, int(f["bbox"][2])), min(h, int(f["bbox"][3]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pitch, yaw, gc = gaze_eng.estimate(crop)

        face_score = f["bbox"][4] if len(f["bbox"]) > 4 else 1.0
        ec = _get_eye_center(f, inv_scale=1.0) if face_score >= EYE_CONF_THRESH else None
        center = ec if ec is not None else np.array([(x1+x2)/2, (y1+y2)/2], float)

        raw_faces.append((center, pitch, yaw, crop))
        face_widths.append(x2 - x1)
        gaze_confs.append(gc)
        raw_face_bboxes.append((x1, y1, x2, y2))

    # -- Sort detections left-to-right for deterministic track-ID assignment --
    if raw_faces:
        ltr = sorted(range(len(raw_faces)),
                     key=lambda i: raw_face_bboxes[i][0])
        raw_faces       = [raw_faces[i]       for i in ltr]
        face_widths     = [face_widths[i]     for i in ltr]
        gaze_confs      = [gaze_confs[i]      for i in ltr]
        raw_face_bboxes = [raw_face_bboxes[i] for i in ltr]

    # -- Temporal smoothing ---------------------------------------------------
    if smoother:
        sm    = smoother.update(raw_faces)
        order = sorted(range(len(raw_faces)), key=lambda i: sm[i][2])
        raw_faces       = [raw_faces[i]       for i in order]
        face_widths     = [face_widths[i]     for i in order]
        gaze_confs      = [gaze_confs[i]      for i in order]
        raw_face_bboxes = [raw_face_bboxes[i] for i in order]
        smoothed        = [(sm[i][0], sm[i][1]) for i in order]
        face_track_ids  = [sm[i][2]            for i in order]
    else:
        smoothed       = [(entry[1], entry[2]) for entry in raw_faces]
        face_track_ids = list(range(len(raw_faces)))

    face_objs = _faces_as_objects(raw_face_bboxes)

    # -- Ray construction & adaptive snap -------------------------------------
    ray_length      = gaze_cfg.ray_length
    conf_ray        = gaze_cfg.conf_ray
    adaptive_ray    = gaze_cfg.adaptive_ray
    snap_dist       = gaze_cfg.snap_dist

    fwd_thresh_rad = np.radians(gaze_cfg.forward_gaze_threshold)

    persons_gaze, ray_snapped, ray_extended = [], [], []
    for fi_loc, (entry, (pitch, yaw), fw, gc, bbox) in enumerate(zip(
            raw_faces, smoothed, face_widths, gaze_confs, raw_face_bboxes)):
        c  = entry[0]

        # Forward-gaze dead zone: when both angles are tiny the 2D projection
        # degenerates to near-zero and normalisation amplifies noise.  Produce
        # a short stub ray instead of a long errant one.
        if fwd_thresh_rad > 0 and abs(pitch) < fwd_thresh_rad and abs(yaw) < fwd_thresh_rad:
            d_raw = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(yaw)])
            end = c + d_raw * (fw * 0.25)
            persons_gaze.append((c, end, (pitch, yaw)))
            ray_snapped.append(False)
            ray_extended.append(False)
            face_confs.append(gc)
            face_bboxes.append(bbox)
            continue

        d  = pitch_yaw_to_2d(pitch, yaw)
        rl = ray_length * (CR_MIN + gc * (CR_MAX - CR_MIN)) if conf_ray else ray_length
        fb = c + d * (fw * rl)
        snap, extended = False, False
        other_faces = [fo for fo in face_objs if fo['_face_idx'] != fi_loc]
        adaptive_targets = objects + other_faces
        if adaptive_ray != "off" and adaptive_targets:
            raw_ctr, raw_snap, _ = adaptive_snap(
                c, d, adaptive_targets, fb, snap_dist,
                gaze_conf=gc,
                bbox_scale=gaze_cfg.snap_bbox_scale,
                w_dist=gaze_cfg.snap_w_dist,
                w_size=gaze_cfg.snap_w_size,
                w_intersect=gaze_cfg.snap_w_intersect)
            if snap_hysteresis is not None:
                obj_ctr, _ = snap_hysteresis.update(face_track_ids[fi_loc], raw_ctr, raw_snap)
                if obj_ctr is None:
                    end = fb
                elif adaptive_ray == "snap":
                    end, snap = obj_ctr, True
                else:
                    t = float(np.dot(obj_ctr - c, d))
                    end, extended = ((c + d * t), True) if t > 0 else (fb, False)
            else:
                if raw_snap:
                    if adaptive_ray == "snap":
                        end, snap = raw_ctr, True
                    else:
                        t = float(np.dot(raw_ctr - c, d))
                        end, extended = ((c + d * t), True) if t > 0 else (fb, False)
                else:
                    end = fb
        else:
            if snap_hysteresis is not None:
                snap_hysteresis.update(face_track_ids[fi_loc], None, False)
            end = fb
        persons_gaze.append((c, end, (pitch, yaw)))
        ray_snapped.append(snap)
        ray_extended.append(extended)
        face_confs.append(gc)
        face_bboxes.append(bbox)

    return (persons_gaze, face_confs, face_bboxes, face_track_ids,
            face_objs, ray_snapped, ray_extended)
