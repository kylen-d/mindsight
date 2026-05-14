"""
depth_pipeline.py -- Per-frame depth estimation pipeline stage.

Sits between the detection and gaze stages in ``process_frame()``.
Writes ``ctx['depth_map']`` and per-object ``depth_median`` values.
"""

from __future__ import annotations

import cv2
import numpy as np

from ms.DepthEstimation.depth_backend import DepthBackend
from ms.pipeline_config import DepthConfig


def run_depth_step(ctx, *, depth_cfg: DepthConfig, depth_backend: DepthBackend):
    """Estimate a per-pixel depth map and attach per-object depth metadata.

    Reads ``ctx['frame']``, ``ctx['objects']``, ``ctx['persons']``.
    Writes ``ctx['depth_map']`` (HxW float32 normalised to [0, 1]) and
    sets ``depth_median`` on each Detection in objects/persons.
    """
    frame = ctx['frame']
    h, w = frame.shape[:2]

    # -- Downsample for inference (the backend handles its own transforms,
    #    but we pre-resize to bound memory / compute) --
    sz = depth_cfg.input_size
    if max(h, w) > sz:
        scale = sz / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame

    # -- Run depth model --
    raw_depth = depth_backend.estimate(small)

    # -- Resize back to original resolution --
    if raw_depth.shape[:2] != (h, w):
        depth_map = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_map = raw_depth

    # -- Normalise to [0, 1] --
    d_min, d_max = depth_map.min(), depth_map.max()
    d_range = d_max - d_min
    if d_range > 1e-8:
        depth_map = (depth_map - d_min) / d_range
    else:
        depth_map = np.zeros_like(depth_map)

    ctx['depth_map'] = depth_map.astype(np.float32)

    # -- Attach per-object depth_median --
    for obj in list(ctx.get('objects', [])) + list(ctx.get('persons', [])):
        x1 = max(0, int(obj['x1']))
        y1 = max(0, int(obj['y1']))
        x2 = min(w, int(obj['x2']))
        y2 = min(h, int(obj['y2']))
        if x2 > x1 and y2 > y1:
            patch = depth_map[y1:y2, x1:x2]
            obj['depth_median'] = float(np.median(patch))
        else:
            obj['depth_median'] = 0.5  # fallback for degenerate bboxes
