"""
GazeTracking/gaze_pipeline.py — Run-loop gaze step (plugin coordinator).

Delegates gaze estimation to the active plugin, then applies unified ray
forming via ``mindsight.PostProcessing.RayForming``.  When a ``GazelleProvider``
is available on the context, periodic Gaze-LLE heatmap inference and belief
blending are applied as a core pipeline feature.

Plugins that implement ``run_pipeline()`` still work for backward compatibility,
but the recommended path is:
  - pitch/yaw plugin provides raw ``estimate(face_bgr)`` per face
  - GazelleProvider handles Gaze-LLE model + heatmap scheduling
  - RayForming module fuses both signals into finalized gaze rays

Usage
-----
    from GazeTracking.gaze_pipeline import run_gaze_step

    run_gaze_step(ctx, face_det=face_det, gaze_eng=gaze_eng, gaze_cfg=gaze_cfg)
"""

import numpy as np

from mindsight.constants import EYE_CONF_THRESH
from mindsight.GazeTracking.gaze_processing import (
    _faces_as_objects,
    _get_eye_center,
)
from mindsight.PostProcessing.RayForming.fixation import apply_lock_on
from mindsight.PostProcessing.RayForming.hit_detection import compute_ray_intersections
from mindsight.PostProcessing.RayForming.object_snap import apply_tip_snapping
from mindsight.pipeline_config import GazeConfig
from mindsight.PostProcessing.RayForming import (
    RawGaze,
    run_ray_forming,
)
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
# Core estimation: per-face pitch/yaw + temporal smoothing
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_pitchyaw(frame, faces, gaze_eng, smoother):
    """Run per-face pitch/yaw estimation and temporal smoothing.

    Returns (raw_faces, smoothed, face_widths, gaze_confs,
             raw_face_bboxes, face_track_ids, face_objs).
    """
    h, w = frame.shape[:2]
    raw_faces, face_widths, gaze_confs, raw_face_bboxes = [], [], [], []
    for f in faces:
        x1, y1 = max(0, int(f["bbox"][0])), max(0, int(f["bbox"][1]))
        x2, y2 = min(w, int(f["bbox"][2])), min(h, int(f["bbox"][3]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pitch, yaw, gc = gaze_eng.estimate(crop)

        face_score = f["bbox"][4] if len(f["bbox"]) > 4 else 1.0
        ec = (_get_eye_center(f, inv_scale=1.0)
              if face_score >= EYE_CONF_THRESH else None)
        center = ec if ec is not None else np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2], float)

        raw_faces.append((center, pitch, yaw, crop))
        face_widths.append(x2 - x1)
        gaze_confs.append(gc)
        raw_face_bboxes.append((x1, y1, x2, y2))

    # Sort left-to-right for deterministic track-ID assignment
    if raw_faces:
        ltr = sorted(range(len(raw_faces)),
                     key=lambda i: raw_face_bboxes[i][0])
        raw_faces       = [raw_faces[i]       for i in ltr]
        face_widths     = [face_widths[i]     for i in ltr]
        gaze_confs      = [gaze_confs[i]      for i in ltr]
        raw_face_bboxes = [raw_face_bboxes[i] for i in ltr]

    # Temporal smoothing
    if smoother:
        sm = smoother.update(raw_faces)
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
    return (raw_faces, smoothed, face_widths, gaze_confs,
            raw_face_bboxes, face_track_ids, face_objs)


# ══════════════════════════════════════════════════════════════════════════════
# Main coordinator pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_gaze_step(ctx, *, face_det, gaze_eng, gaze_cfg: GazeConfig, **kwargs):
    """
    Run face detection, gaze estimation, and ray-bbox intersection for one frame.

    Reads from ctx
    --------------
    frame, detection_frame, inverse_scale, objects, cached_faces,
    smoother, locker, snap_temporal, smooth_snap_tracker,
    gazelle_provider, ray_cfg.

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
    snap_temporal = kwargs.get('snap_temporal', ctx.get('snap_temporal'))
    smooth_snap_tracker = kwargs.get('smooth_snap_tracker',
                                     ctx.get('smooth_snap_tracker'))
    do_cache = ctx.get('do_cache', False)

    # Ray forming state (core Gazelle blend path)
    gazelle_provider = ctx.get('gazelle_provider')
    ray_cfg = ctx.get('ray_cfg')

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

    # ── Determine pipeline path ──────────────────────────────────────────────
    depth_map = ctx.get('depth_map')
    depth_cfg = ctx.get('depth_cfg')
    h, w = frame.shape[:2]

    has_plugin_pipeline = (hasattr(gaze_eng, 'run_pipeline')
                           and callable(gaze_eng.run_pipeline)
                           and type(gaze_eng).run_pipeline is not GazePlugin.run_pipeline)

    # ── Path A: Core ray forming (per-face PY + optional Gazelle blend) ─────
    # Used when the engine is a per-face backend.
    use_core_ray_forming = (
        ray_cfg is not None
        and getattr(gaze_eng, 'mode', 'per_face') == 'per_face'
    )

    if use_core_ray_forming:
        # Per-face pitch/yaw estimation
        (raw_faces, smoothed, face_widths, gaze_confs,
         raw_face_bboxes, face_track_ids, face_objs) = _estimate_pitchyaw(
            frame, faces, gaze_eng, smoother)

        # Gazelle heatmap inference (core, not plugin)
        if gazelle_provider is not None:
            gazelle_provider.step(frame, raw_face_bboxes, face_track_ids)

        # Build RawGaze list
        raw_gazes = []
        for fi_loc, (entry, (pitch, yaw), fw, gc, bbox) in enumerate(zip(
                raw_faces, smoothed, face_widths, gaze_confs,
                raw_face_bboxes)):
            raw_gazes.append(RawGaze(
                origin=entry[0], pitch=pitch, yaw=yaw,
                confidence=gc, face_width=fw,
                track_id=face_track_ids[fi_loc], face_bbox=bbox))

        # Persistent ray forming state objects (created once in cli.py/run())
        gazelle_blender = ctx.get('gazelle_blender')
        ray_object_snap = ctx.get('ray_object_snap')

        # One Euro time-constant must track real fps -- a 60fps clip needs
        # half the per-sample smoothing of a 30fps clip.
        _vfps = ctx.get('video_fps')
        dt = (1.0 / _vfps
              if _vfps is not None and _vfps > 0 and _vfps == _vfps
              else 1.0 / 30.0)

        result = run_ray_forming(
            raw_gazes=raw_gazes,
            objects=objects,
            face_objs=face_objs,
            frame_h=h, frame_w=w,
            cfg=ray_cfg,
            gazelle_provider=gazelle_provider,
            gazelle_blender=gazelle_blender,
            object_snap=ray_object_snap,
            depth_map=depth_map,
            dt=dt,
        )
        persons_gaze = result.persons_gaze
        face_confs = result.face_confs
        face_bboxes = result.face_bboxes
        face_track_ids = result.face_track_ids
        face_objs = result.face_objs
        ray_snapped = result.ray_snapped
        ray_extended = result.ray_extended

    # ── Path B: Custom plugin pipeline ──────────────────────────────────────
    elif has_plugin_pipeline:
        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended) = gaze_eng.run_pipeline(
            frame=frame, faces=faces, objects=objects, gaze_cfg=gaze_cfg,
            smoother=smoother, snap_temporal=snap_temporal,
            smooth_snap_tracker=smooth_snap_tracker,
            depth_map=depth_map, depth_cfg=depth_cfg,
        )

    # ── Path C: Default scene-level pipeline (standalone Gazelle) ───────────
    else:
        (persons_gaze, face_confs, face_bboxes, face_track_ids,
         face_objs, ray_snapped, ray_extended) = _default_scene_pipeline(
            frame, faces, gaze_eng,
        )

    # ── Tip-snapping between gaze rays (per-face backends only) ──────────────
    _tip_cfg = ray_cfg if ray_cfg is not None else gaze_cfg
    persons_gaze, ray_snapped, ray_extended = apply_tip_snapping(
        persons_gaze, ray_snapped, ray_extended, gaze_eng, _tip_cfg,
        face_track_ids=face_track_ids,
        smooth_snap_tracker=smooth_snap_tracker)

    # ── Lock-on ──────────────────────────────────────────────────────────────
    persons_gaze, lock_info = apply_lock_on(persons_gaze, locker, objects)

    # ── Ray-bbox (or cone) intersection + confidence gate ────────────────────
    _sample_r = depth_cfg.gaze_sample_radius if depth_cfg is not None else 2
    _hit_cfg = ray_cfg if ray_cfg is not None else gaze_cfg
    all_targets, hits, hit_events = compute_ray_intersections(
        persons_gaze, face_confs, face_track_ids, face_objs, objects, _hit_cfg,
        depth_map=depth_map, gaze_sample_radius=_sample_r)

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
