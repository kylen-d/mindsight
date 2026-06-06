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

import dataclasses
from collections import defaultdict

import cv2

from mindsight.ObjectDetection.detection import Detection
from mindsight.ObjectDetection.object_detection import parse_dets
from mindsight.pipeline_config import DetectionConfig


# ══════════════════════════════════════════════════════════════════════════════
# Overlap merging
# ══════════════════════════════════════════════════════════════════════════════

def _overlap(a: Detection, b: Detection, metric: str) -> float:
    """Compute overlap between two detections using *metric* ('iou' or 'iomin')."""
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    if metric == 'iomin':
        denom = min(area_a, area_b)
    else:  # iou
        denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _cluster(dets: list[Detection], metric: str, threshold: float) -> list[list[int]]:
    """Group detection indices into overlap clusters (union-find)."""
    n = len(dets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(dets[i], dets[j], metric) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _det_area(d: Detection) -> int:
    return (d.x2 - d.x1) * (d.y2 - d.y1)


def _filter_cluster(cluster: list[Detection]) -> Detection:
    """Keep the highest-confidence detection from the cluster."""
    return max(cluster, key=lambda d: d.conf)


def _merge_cluster(cluster: list[Detection]) -> Detection:
    """Create an encompassing bounding box with the best detection's metadata."""
    best = max(cluster, key=lambda d: d.conf)
    return dataclasses.replace(
        best,
        x1=min(d.x1 for d in cluster),
        y1=min(d.y1 for d in cluster),
        x2=max(d.x2 for d in cluster),
        y2=max(d.y2 for d in cluster),
    )


def _dynamic_resolve(cluster: list[Detection]) -> Detection:
    """Decide per-cluster whether to filter or merge.

    Heuristics
    ----------
    1. **Confidence dominance** — if the top detection is >=1.5x the
       confidence of the second-best, one box is clearly the "real" detection
       and the rest are duplicates → filter.
    2. **Area expansion** — if the merged bounding box would be >=1.5x the
       area of the largest individual box, the boxes are spread apart and
       merging would create an unreasonably large result → filter.
    3. Otherwise the boxes are similar in confidence and tightly overlapping
       → merge.
    """
    sorted_by_conf = sorted(cluster, key=lambda d: d.conf, reverse=True)
    conf_ratio = (sorted_by_conf[0].conf / sorted_by_conf[1].conf
                  if sorted_by_conf[1].conf > 0 else float('inf'))
    if conf_ratio >= 1.5:
        return _filter_cluster(cluster)

    max_area = max(_det_area(d) for d in cluster)
    merged = _merge_cluster(cluster)
    merged_area = _det_area(merged)
    expansion = merged_area / max_area if max_area > 0 else float('inf')
    if expansion >= 1.5:
        return _filter_cluster(cluster)

    return merged


def merge_overlaps(
    dets: list[Detection],
    strategy: str = 'filter',
    threshold: float = 0.7,
) -> list[Detection]:
    """Merge or filter overlapping same-class detections.

    Parameters
    ----------
    strategy : 'filter', 'merge', or 'dynamic'
        filter  — keep highest-confidence detection per cluster.  Uses IoMin
        (intersection / min area) to catch small duplicates inside larger boxes.
        merge   — union bounding box encompassing all cluster members.  Uses IoU
        (intersection / union) to merge similar-sized overlapping boxes.
        dynamic — choose per-cluster based on relative confidence and area
        expansion.  Uses IoMin for clustering.
    threshold : float
        Minimum overlap to consider two detections as overlapping.
    """
    if not dets:
        return dets

    metric = 'iou' if strategy == 'merge' else 'iomin'

    by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in dets:
        by_class[d.class_name].append(d)

    resolve = {
        'filter': _filter_cluster,
        'merge': _merge_cluster,
        'dynamic': _dynamic_resolve,
    }[strategy]

    result: list[Detection] = []
    for class_dets in by_class.values():
        if len(class_dets) == 1:
            result.append(class_dets[0])
            continue

        for cluster_idxs in _cluster(class_dets, metric, threshold):
            cluster = [class_dets[i] for i in cluster_idxs]
            if len(cluster) == 1:
                result.append(cluster[0])
            else:
                result.append(resolve(cluster))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Detection step
# ══════════════════════════════════════════════════════════════════════════════

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

    if det_cfg.merge_overlaps:
        all_dets = merge_overlaps(
            all_dets,
            strategy=det_cfg.merge_overlap_strategy,
            threshold=det_cfg.merge_overlap_threshold,
        )

    persons = [d for d in all_dets if d['class_name'].lower() == 'person']
    objects = [d for d in all_dets if d['class_name'].lower() != 'person']

    if obj_cache is not None:
        objects = obj_cache.update(objects)

    ctx['all_dets'] = all_dets
    ctx['persons'] = persons
    ctx['objects'] = objects
    ctx['detection_frame'] = detection_frame
    ctx['inverse_scale'] = inverse_scale
