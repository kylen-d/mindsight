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

from mindsight.constants import CR_MAX, CR_MIN, EYE_CONF_THRESH
from mindsight.GazeTracking.gaze_processing import (
    _faces_as_objects,
    _get_eye_center,
    snap_score,
)
from mindsight.utils.geometry import pitch_yaw_to_2d


def run_pitchyaw_pipeline(*, frame, faces, gaze_eng, objects, gaze_cfg,
                          smoother=None, snap_temporal=None,
                          smooth_snap_tracker=None, **kwargs):
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
    snap_temporal   : Optional SnapTemporalState instance.
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

    fwd_thresh_rad = np.radians(gaze_cfg.forward_gaze_threshold)
    frame_diag = float(np.sqrt(h * h + w * w))

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
            if snap_temporal is not None:
                snap_temporal.update(face_track_ids[fi_loc], None, False,
                                    gaze_conf=gc)
            continue

        d  = pitch_yaw_to_2d(pitch, yaw)
        rl = ray_length * (CR_MIN + gc * (CR_MAX - CR_MIN)) if conf_ray else ray_length
        fb = c + d * (fw * rl)
        snap, extended = False, False

        other_faces = [fo for fo in face_objs if fo['_face_idx'] != fi_loc]
        adaptive_targets = objects + other_faces

        if adaptive_ray != "off" and adaptive_targets:
            # Face bbox center for head-blend angular scoring
            bx1, by1, bx2, by2 = bbox
            face_ctr = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])

            # Get previous target key for temporal stickiness
            prev_key = None
            key_fn = None
            if snap_temporal is not None:
                prev_key = snap_temporal.prev_target_key(face_track_ids[fi_loc])
                key_fn = snap_temporal.key_for

            # Depth-aware scoring (opt-in via depth_cfg)
            _depth_map = kwargs.get('depth_map')
            _depth_cfg = kwargs.get('depth_cfg')
            _w_depth = (_depth_cfg.snap_w_depth
                        if _depth_cfg is not None
                        and _depth_cfg.depth_aware_scoring
                        and _depth_map is not None else 0.0)
            _sample_r = (_depth_cfg.gaze_sample_radius
                         if _depth_cfg is not None else 2)

            raw_ctr, raw_found, _, _ = snap_score(
                c, d, adaptive_targets, fb,
                snap_dist=gaze_cfg.snap_dist,
                gaze_conf=gc,
                bbox_scale=gaze_cfg.snap_bbox_scale,
                w_dist=gaze_cfg.snap_w_dist,
                w_angle=gaze_cfg.snap_w_angle,
                w_size=gaze_cfg.snap_w_size,
                w_intersect=gaze_cfg.snap_w_intersect,
                w_temporal=gaze_cfg.snap_w_temporal,
                gate_angle_deg=gaze_cfg.snap_gate_angle,
                head_blend=gaze_cfg.snap_head_blend,
                quality_thresh=gaze_cfg.snap_quality_thresh,
                face_center=face_ctr,
                prev_target_key=prev_key,
                frame_diag=frame_diag,
                _key_fn=key_fn,
                depth_map=_depth_map,
                w_depth=_w_depth,
                gaze_endpoint=fb,
                gaze_sample_radius=_sample_r)

            if snap_temporal is not None:
                obj_ctr, did_snap = snap_temporal.update(
                    face_track_ids[fi_loc], raw_ctr, raw_found, gaze_conf=gc)
                if obj_ctr is None or not did_snap:
                    end = fb
                elif adaptive_ray == "snap":
                    end, snap = obj_ctr, True
                else:
                    t = float(np.dot(obj_ctr - c, d))
                    end, extended = ((c + d * t), True) if t > 0 else (fb, False)
            else:
                if raw_found:
                    if adaptive_ray == "snap":
                        end, snap = raw_ctr, True
                    else:
                        t = float(np.dot(raw_ctr - c, d))
                        end, extended = ((c + d * t), True) if t > 0 else (fb, False)
                else:
                    end = fb
        else:
            if snap_temporal is not None:
                snap_temporal.update(face_track_ids[fi_loc], None, False,
                                    gaze_conf=gc)
            end = fb

        # ── Smooth snap (objects) ───────────────────────────────────────
        # Always update the tracker so the state stays fresh — this
        # ensures smooth transitions both into AND out of snaps.
        if (smooth_snap_tracker is not None
                and gaze_cfg.smooth_snap in ("objects", "all")):
            end = smooth_snap_tracker.update(
                face_track_ids[fi_loc], end)
        persons_gaze.append((c, end, (pitch, yaw)))
        ray_snapped.append(snap)
        ray_extended.append(extended)
        face_confs.append(gc)
        face_bboxes.append(bbox)

    return (persons_gaze, face_confs, face_bboxes, face_track_ids,
            face_objs, ray_snapped, ray_extended)
