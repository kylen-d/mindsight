"""
GazeTracking/gaze_pipeline.py — Run-loop gaze step (plugin coordinator).

Delegates gaze estimation to the active plugin's ``run_pipeline()`` method,
then applies generic post-processing (tip-snapping, lock-on, ray-bbox / cone
intersection).  Plugins that do not implement ``run_pipeline()`` fall back to
a built-in default handler based on their ``mode`` attribute.

Usage
-----
    from GazeTracking.gaze_pipeline import run_gaze_step

    run_gaze_step(ctx, face_det=face_det, gaze_eng=gaze_eng, gaze_cfg=gaze_cfg)
    # smoother, locker, snap_hysteresis can be set on ctx or passed as kwargs.
    # Results written to ctx: 'persons_gaze', 'face_confs', 'face_bboxes',
    #   'face_track_ids', 'all_targets', 'hits', 'hit_events',
    #   'lock_info', 'ray_snapped', 'ray_extended'
"""

import numpy as np

from ms.constants import EYE_CONF_THRESH
from ms.GazeTracking.gaze_processing import (
    _faces_as_objects,
    _get_eye_center,
    apply_lock_on,
    apply_tip_snapping,
    compute_ray_intersections,
)
from ms.pipeline_config import GazeConfig
from Plugins import GazePlugin

# ══════════════════════════════════════════════════════════════════════════════
# Default scene-level pipeline (for plugins without run_pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _default_scene_pipeline(frame, faces, gaze_eng):
    """Fallback pipeline for scene-level backends (e.g. Gazelle) that do not
    implement their own ``run_pipeline()``."""
    h, w = frame.shape[:2]
    face_confs:  list = []
    face_bboxes: list = []

    bboxes_raw   = []
    valid_faces  = []
    for f in faces:
        x1 = max(0, int(f["bbox"][0])); y1 = max(0, int(f["bbox"][1]))
        x2 = min(w, int(f["bbox"][2])); y2 = min(h, int(f["bbox"][3]))
        if x2 > x1 and y2 > y1:
            bboxes_raw.append((x1, y1, x2, y2))
            valid_faces.append(f)
    gz = gaze_eng.estimate_frame(frame, bboxes_raw)

    # Sort detections left-to-right for deterministic track-ID assignment
    if bboxes_raw:
        ltr = sorted(range(len(bboxes_raw)), key=lambda i: bboxes_raw[i][0])
        bboxes_raw  = [bboxes_raw[i]  for i in ltr]
        valid_faces = [valid_faces[i] for i in ltr]
        gz          = [gz[i]          for i in ltr]

    persons_gaze = []
    for f, (x1, y1, x2, y2), (xy, gc) in zip(valid_faces, bboxes_raw, gz):
        face_score = f["bbox"][4] if len(f["bbox"]) > 4 else 1.0
        ec = _get_eye_center(f) if face_score >= EYE_CONF_THRESH else None
        origin = ec if ec is not None else np.array([(x1+x2)/2, (y1+y2)/2], float)
        persons_gaze.append((origin, xy, None))
        face_confs.append(gc)
        face_bboxes.append((x1, y1, x2, y2))

    ray_snapped    = [False] * len(persons_gaze)
    ray_extended   = [False] * len(persons_gaze)
    face_objs      = _faces_as_objects(face_bboxes)
    face_track_ids = list(range(len(persons_gaze)))

    return (persons_gaze, face_confs, face_bboxes, face_track_ids,
            face_objs, ray_snapped, ray_extended)


# ══════════════════════════════════════════════════════════════════════════════
# Main coordinator pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_gaze_step(ctx, *, face_det, gaze_eng, gaze_cfg: GazeConfig, **kwargs):
    """
    Run face detection, gaze estimation, and ray-bbox intersection for one frame.

    Reads from ctx
    --------------
    frame           : BGR numpy array at full display resolution.
    fdet            : Detection-scale frame (from run_detection_step).
    inv             : Inverse of detect_scale (from run_detection_step).
    objects         : Non-person detection dicts (from run_detection_step).
    cached_faces    : Pre-detected face list to skip RetinaFace this frame.  Optional.
    smoother        : GazeSmootherReID instance or None.  Optional.
    locker          : GazeLockTracker instance or None.  Optional.
    snap_hysteresis : SnapHysteresisTracker instance or None.  Optional.

    kwargs override any ctx value with the same key.

    Writes to ctx
    -------------
    persons_gaze, face_confs, face_bboxes, face_track_ids,
    all_targets, hits, hit_events, lock_info, ray_snapped, ray_extended, faces.
    """
    frame = ctx['frame']
    fdet = ctx['detection_frame']
    inv = ctx['inverse_scale']
    objects = ctx['objects']
    cached_faces = kwargs.get('cached_faces', ctx.get('cached_faces'))
    smoother = kwargs.get('smoother', ctx.get('smoother'))
    locker = kwargs.get('locker', ctx.get('locker'))
    snap_hysteresis = kwargs.get('snap_hysteresis', ctx.get('snap_hysteresis'))
    do_cache = ctx.get('do_cache', False)

    # ── Face detection (generic) ─────────────────────────────────────────────
    if cached_faces is not None:
        faces = cached_faces
    else:
        raw = face_det.detect(fdet)
        faces = (
            [{**f,
              "bbox": [c * inv for c in f["bbox"][:4]] + list(f["bbox"][4:]),
              "kps":  [[kp[0] * inv, kp[1] * inv] for kp in f["kps"]]
                      if f.get("kps") is not None else None}
             for f in raw]
            if inv != 1.0 else raw
        )

    if do_cache:
        ctx['faces'] = faces

    # ── Delegate to plugin pipeline or use default ───────────────────────────
    has_pipeline = (hasattr(gaze_eng, 'run_pipeline')
                     and callable(gaze_eng.run_pipeline)
                     and type(gaze_eng).run_pipeline is not GazePlugin.run_pipeline)

    if has_pipeline:
        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended) = gaze_eng.run_pipeline(
            frame=frame, faces=faces, objects=objects, gaze_cfg=gaze_cfg,
            smoother=smoother, snap_hysteresis=snap_hysteresis,
        )
    else:
        # Fallback for plugins without run_pipeline (e.g. scene-level Gazelle)
        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended) = _default_scene_pipeline(
            frame, faces, gaze_eng,
        )

    # ── Tip-snapping between gaze rays (per-face backends only) ──────────────
    persons_gaze, ray_snapped, ray_extended = apply_tip_snapping(
        persons_gaze, ray_snapped, ray_extended, gaze_eng, gaze_cfg)

    # ── Lock-on ──────────────────────────────────────────────────────────────
    persons_gaze, lock_info = apply_lock_on(persons_gaze, locker, objects)

    # ── Ray-bbox (or cone) intersection + confidence gate ────────────────────
    all_targets, hits, hit_events = compute_ray_intersections(
        persons_gaze, face_confs, face_track_ids, face_objs, objects, gaze_cfg)

    ctx['persons_gaze'] = persons_gaze
    ctx['face_confs'] = face_confs
    ctx['face_bboxes'] = face_bboxes
    ctx['face_track_ids'] = face_track_ids
    ctx['all_targets'] = all_targets
    ctx['hits'] = hits
    ctx['hit_events'] = hit_events
    ctx['lock_info'] = lock_info
    ctx['ray_snapped'] = ray_snapped
    ctx['ray_extended'] = ray_extended
