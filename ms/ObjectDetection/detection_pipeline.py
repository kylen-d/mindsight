"""
ObjectDetection/detection_pipeline.py — Run-loop detection step.

Extracts the YOLO object-detection pass from the main frame loop so it can
be swapped, extended, or mocked by user plugins without touching MindSight.py.

Usage
-----
    from ObjectDetection.detection_pipeline import run_detection_step

    run_detection_step(ctx, yolo=yolo, det_cfg=det_cfg, obj_cache=obj_cache)
    # Results written to ctx: 'all_dets', 'persons', 'objects', 'fdet', 'inv'
"""

import cv2

from ms.ObjectDetection.object_detection import parse_dets
from ms.pipeline_config import DetectionConfig


def run_detection_step(ctx, *, yolo, det_cfg: DetectionConfig,
                       obj_cache=None, detection_plugins=None, **kwargs):
    """
    Run object detection for one frame.

    Uses YOLO as the default/fallback detector.  If *detection_plugins* are
    provided, each plugin's ``detect()`` method is called after YOLO and may
    augment, filter, or replace the detection list.

    Reads from ctx
    --------------
    frame           : BGR numpy array at full display resolution.
    cached_all_dets : Pre-computed detection list to skip detection this frame
                      (used when skip_frames > 1).  Optional.

    Writes to ctx
    -------------
    all_dets : list[dict]   All detections in full-resolution frame coordinates.
    persons  : list[dict]   Subset where class_name == 'person'.
    objects  : list[dict]   Subset where class_name != 'person', after cache.
    detection_frame : np.ndarray  Frame actually fed to the detector (may be downscaled).
    inverse_scale   : float      Inverse of detect_scale (for coordinate mapping downstream).
    """
    frame = ctx['frame']
    cached_all_dets = ctx.get('cached_all_dets')

    h, w = frame.shape[:2]
    detect_scale = det_cfg.detect_scale

    if detect_scale != 1.0:
        dw = max(1, int(w * detect_scale))
        dh = max(1, int(h * detect_scale))
        detection_frame = cv2.resize(frame, (dw, dh))
        inverse_scale = 1.0 / detect_scale
    else:
        detection_frame, inverse_scale = frame, 1.0

    if cached_all_dets is not None:
        all_dets = cached_all_dets
    else:
        # When detection plugins declare a min_conf, lower the YOLO threshold
        # so sub-threshold candidates are available for boosting.
        effective_conf = det_cfg.conf
        if detection_plugins:
            for p in detection_plugins:
                mc = getattr(p, 'min_conf', None)
                if mc is not None:
                    effective_conf = min(effective_conf, mc)

        # Default detection: YOLO
        all_dets = parse_dets(
            yolo(detection_frame, conf=effective_conf, classes=det_cfg.class_ids, verbose=False),
            yolo.names, effective_conf, det_cfg.blacklist)
        if detect_scale != 1.0:
            for d in all_dets:
                d.update(x1=int(d['x1'] * inverse_scale), y1=int(d['y1'] * inverse_scale),
                         x2=int(d['x2'] * inverse_scale), y2=int(d['y2'] * inverse_scale))

        # Plugin hook: let detection plugins augment/filter/replace detections
        for plugin in (detection_plugins or []):
            if hasattr(plugin, 'detect'):
                all_dets = plugin.detect(
                    frame=frame, detection_frame=detection_frame,
                    all_dets=all_dets, det_cfg=det_cfg,
                    prev_persons_gaze=ctx.get('prev_persons_gaze', []),
                    prev_face_track_ids=ctx.get('prev_face_track_ids', []),
                ) or all_dets

        # Post-plugin confidence gate: discard anything plugins didn't boost
        # above the user's configured threshold.
        if effective_conf < det_cfg.conf:
            all_dets = [d for d in all_dets if d['conf'] >= det_cfg.conf]

    persons = [d for d in all_dets if d['class_name'].lower() == 'person']
    objects = [d for d in all_dets if d['class_name'].lower() != 'person']

    if obj_cache is not None:
        objects = obj_cache.update(objects)

    ctx['all_dets'] = all_dets
    ctx['persons'] = persons
    ctx['objects'] = objects
    ctx['detection_frame'] = detection_frame
    ctx['inverse_scale'] = inverse_scale
