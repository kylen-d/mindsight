"""
RayForming/hit_detection.py — Ray-bounding box (or cone) intersection testing.

Moved from ``GazeTracking/gaze_processing.py``.  Determines which objects
each participant is looking at based on gaze ray geometry.
"""
from __future__ import annotations

import numpy as np

from mindsight.utils.geometry import ray_hits_box, ray_hits_cone, sample_depth_patch


def compute_ray_intersections(persons_gaze, face_confs, face_track_ids,
                              face_objs, objects, gaze_cfg,
                              depth_map=None, gaze_sample_radius=2,
                              ray_snapped=None, ray_extended=None):
    """Compute ray-bbox (or cone) intersections with confidence gating.

    Parameters
    ----------
    persons_gaze    : list of (origin, ray_end, angles)
    face_confs      : list[float] per-face gaze confidence
    face_track_ids  : list[int] stable track IDs (used in hit_events)
    face_objs       : list[Detection] face-as-object targets
    objects         : non-person detection list
    gaze_cfg        : config with hit_conf_gate, detect_extend, gaze_cone_angle, etc.
    ray_snapped     : optional list[bool] per face (object-snap applied)
    ray_extended    : optional list[bool] per face (ray reach extended)

    Returns
    -------
    all_targets : list[dict]  objects + face_objs
    hits        : set of (face_track_id, target_idx) pairs.  Keyed by the
                  stable track ID since v1.1 (W1.1) so every consumer shares
                  one identity convention with ``hit_events`` -- previously
                  list-position, which churned when face order changed and
                  attached pid_map labels to the wrong people downstream.
    hit_events  : list[dict] per-hit records with face_idx = track ID
    """
    hit_conf_gate   = getattr(gaze_cfg, 'hit_conf_gate', 0.0)
    scope           = getattr(gaze_cfg, 'detect_extend_scope', 'objects')
    detect_extend   = (getattr(gaze_cfg, 'detect_extend', 0.0)
                       if scope in ('objects', 'both') else 0.0)
    gaze_cone_angle = getattr(gaze_cfg, 'gaze_cone_angle', 0.0)
    gaze_tips       = getattr(gaze_cfg, 'gaze_tips', False)
    tip_radius      = getattr(gaze_cfg, 'tip_radius', 80)
    fwd_thresh_rad  = np.radians(getattr(gaze_cfg, 'forward_gaze_threshold', 5.0))
    all_targets = objects + face_objs
    hits, hit_events = set(), []
    for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
        if hit_conf_gate > 0.0 and fi < len(face_confs) and face_confs[fi] < hit_conf_gate:
            continue
        if fwd_thresh_rad > 0 and angles:
            if abs(angles[0]) < fwd_thresh_rad and abs(angles[1]) < fwd_thresh_rad:
                continue
        o_arr  = np.asarray(origin, float)
        re_arr = np.asarray(ray_end, float)
        dv     = re_arr - o_arr
        dl     = np.linalg.norm(dv)
        udir   = dv / dl if dl > 1e-6 else np.array([0., 1.])
        detect_range = dl + detect_extend

        if detect_extend > 0:
            det_end = o_arr + udir * detect_range
        else:
            det_end = re_arr

        for oi, obj in enumerate(all_targets):
            if obj.get('_face_idx') == fi:
                continue
            if gaze_cone_angle > 0.0:
                hit = ray_hits_cone(o_arr, udir,
                                    obj['x1'], obj['y1'], obj['x2'], obj['y2'],
                                    gaze_cone_angle, ray_length=detect_range)
            else:
                hit = ray_hits_box(o_arr, det_end,
                                   obj['x1'], obj['y1'], obj['x2'], obj['y2'])
            if not hit and gaze_tips:
                cx = np.clip(re_arr[0], obj['x1'], obj['x2'])
                cy = np.clip(re_arr[1], obj['y1'], obj['y2'])
                hit = (cx - re_arr[0])**2 + (cy - re_arr[1])**2 <= tip_radius**2
            if hit:
                hits.add((face_track_ids[fi] if face_track_ids else fi, oi))
                ev = dict(
                    face_idx=face_track_ids[fi] if face_track_ids else fi,
                    object=obj['class_name'],
                    object_conf=obj['conf'],
                    bbox=(obj['x1'], obj['y1'], obj['x2'], obj['y2']),
                    gaze_conf=(face_confs[fi]
                               if fi < len(face_confs) else None),
                    gaze_pitch=angles[0] if angles else None,
                    gaze_yaw=angles[1] if angles else None,
                    ray_end=(float(re_arr[0]), float(re_arr[1])),
                    ray_snapped=bool(ray_snapped[fi]) if (
                        ray_snapped and fi < len(ray_snapped)) else False,
                    ray_extended=bool(ray_extended[fi]) if (
                        ray_extended and fi < len(ray_extended)) else False)
                if depth_map is not None:
                    ev['depth_median'] = obj.get('depth_median', 0.5)
                    ev['depth_at_gaze'] = sample_depth_patch(
                        depth_map, re_arr[0], re_arr[1],
                        radius=gaze_sample_radius)
                hit_events.append(ev)
    return all_targets, hits, hit_events
