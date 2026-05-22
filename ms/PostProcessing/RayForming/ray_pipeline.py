"""
RayForming/ray_pipeline.py — Unified ray forming orchestrator.

Converts raw gaze estimates (pitch/yaw angles + optional Gaze-LLE heatmaps)
into finalized gaze rays.  Replaces the ray construction loops previously
embedded in ``pitchyaw_pipeline.py``.

Pipeline order per face:
  1. Construct base ray from pitch/yaw (direction + confidence-scaled length).
  2. Forward-gaze dead zone check (stub ray if looking at camera).
  3. Gaze-LLE blend (fixation-aware scheduler trust + One Euro smoothing).
  4. Depth-adjusted ray length (when depth map available).
  5. Object snap (snap/extend to YOLO objects).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ms.constants import CR_MAX, CR_MIN
from ms.PostProcessing.RayForming.ray_config import RayFormingConfig
from ms.PostProcessing.RayForming.gazelle_blender import GazeLLEBlender
from ms.PostProcessing.RayForming.depth_ray import depth_adjusted_length
from ms.PostProcessing.RayForming.object_snap import ObjectSnap
from ms.utils.geometry import pitch_yaw_to_2d


class RawGaze(NamedTuple):
    """Raw per-face gaze estimate before ray forming."""
    origin: np.ndarray        # eye center (x, y) in pixels
    pitch: float              # gaze pitch in radians
    yaw: float                # gaze yaw in radians
    confidence: float         # gaze confidence [0, 1]
    face_width: float         # face bbox width in pixels
    track_id: int             # stable face track ID
    face_bbox: tuple          # (x1, y1, x2, y2)


@dataclass
class RayFormingResult:
    """Output of the ray forming pipeline."""
    persons_gaze: list        # [(origin, ray_end, (pitch, yaw)), ...]
    face_confs: list           # [float, ...]
    face_bboxes: list          # [(x1, y1, x2, y2), ...]
    face_track_ids: list       # [int, ...]
    face_objs: list            # [Detection, ...]
    ray_snapped: list          # [bool, ...]
    ray_extended: list         # [bool, ...]


def run_ray_forming(
    raw_gazes: list[RawGaze],
    objects: list,
    face_objs: list,
    frame_h: int,
    frame_w: int,
    cfg: RayFormingConfig,
    *,
    gazelle_provider=None,
    gazelle_blender: GazeLLEBlender | None = None,
    object_snap: ObjectSnap | None = None,
    depth_map: np.ndarray | None = None,
    dt: float = 1.0 / 30.0,
) -> RayFormingResult:
    """Run the full ray forming pipeline on raw gaze estimates.

    Parameters
    ----------
    raw_gazes      : per-face raw estimates from pitch/yaw + optional heatmap.
    objects        : non-person detection list.
    face_objs      : face-as-object Detection list.
    frame_h, frame_w : frame dimensions in pixels.
    cfg            : RayFormingConfig with all tuning parameters.
    gazelle_provider : optional GazelleProvider owning the scheduler +
                       heatmap cache; supplies accept/trust signals.
    gazelle_blender  : optional GazeLLEBlender turning (accept, trust) into
                       a smoothed endpoint.
    object_snap    : optional ObjectSnap for snap/extend/smooth.
    depth_map      : optional HxW normalized depth map.
    dt             : seconds between frames; used by the One Euro smoothers.

    Returns
    -------
    RayFormingResult with finalized gaze rays and metadata.
    """
    fwd_thresh_rad = np.radians(cfg.forward_gaze_threshold)
    frame_diag = float(np.sqrt(frame_h ** 2 + frame_w ** 2))

    persons_gaze = []
    face_confs = []
    face_bboxes = []
    face_track_ids = []
    ray_snapped = []
    ray_extended = []

    for rg in raw_gazes:
        c = np.asarray(rg.origin, float)
        pitch, yaw = rg.pitch, rg.yaw
        gc = rg.confidence
        fw = rg.face_width
        tid = rg.track_id
        bbox = rg.face_bbox

        # ── 1. Forward-gaze dead zone ──────────────────────────────────────
        if (fwd_thresh_rad > 0
                and abs(pitch) < fwd_thresh_rad
                and abs(yaw) < fwd_thresh_rad):
            d_raw = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(yaw)])
            end = c + d_raw * (fw * 0.25)
            persons_gaze.append((c, end, (pitch, yaw)))
            face_confs.append(gc)
            face_bboxes.append(bbox)
            face_track_ids.append(tid)
            ray_snapped.append(False)
            ray_extended.append(False)
            # Update snap temporal state so it doesn't hold stale targets
            if object_snap is not None and object_snap.temporal is not None:
                object_snap.temporal.update(tid, None, False, gaze_conf=gc)
            continue

        # ── 2. Construct base ray from pitch/yaw ──────────────────────────
        d = pitch_yaw_to_2d(pitch, yaw)
        rl = (cfg.ray_length * (CR_MIN + gc * (CR_MAX - CR_MIN))
              if cfg.conf_ray else cfg.ray_length)
        base_length = fw * rl
        fb = c + d * base_length  # fallback endpoint (pure pitch/yaw)

        # ── 3. Gaze-LLE blend (fixation-aware scheduler + One Euro) ──────
        # The provider's scheduler decides which faces accept fresh
        # heatmaps; the blender turns (accept, trust) into a smoothed
        # endpoint.  observe_face() feeds THIS frame's PY signal into the
        # scheduler for the NEXT frame's fire decision (one-frame lag,
        # documented in gazelle_provider.py).
        if gazelle_provider is not None and gazelle_blender is not None:
            gazelle_provider.observe_face(
                track_id=tid, py_dir=d, py_conf=gc)
            hm, age, inout, wanted = gazelle_provider.heatmap_cache.get(tid)
            trust = gazelle_provider.likelihood(tid)
            accept = bool(wanted and age == 0)
            endpoint = gazelle_blender.update(
                track_id=tid,
                pitch=pitch, yaw=yaw, gaze_conf=gc,
                origin=c, face_width=fw,
                frame_h=frame_h, frame_w=frame_w,
                gazelle_hm=(hm if accept else None),
                accept_heatmap=accept, trust=trust, dt=dt)
            fb = endpoint

        # ── 4. Depth-adjusted ray length ──────────────────────────────────
        if cfg.depth_ray_length and depth_map is not None:
            new_length = depth_adjusted_length(
                depth_map, fb, base_length,
                length_min=cfg.depth_length_min,
                length_max=cfg.depth_length_max,
                sample_radius=cfg.gaze_sample_radius,
            )
            # Rescale endpoint to new length while preserving direction
            cur_vec = fb - c
            cur_len = float(np.linalg.norm(cur_vec))
            if cur_len > 1e-6:
                fb = c + (cur_vec / cur_len) * new_length

        # ── 5. Object snap ────────────────────────────────────────────────
        snap, extended = False, False
        end = fb
        if object_snap is not None:
            fi_loc = len(persons_gaze)  # current index in the output list
            end, snap, extended = object_snap.snap_ray(
                origin=c, endpoint=fb, direction=d,
                gaze_conf=gc, face_width=fw,
                objects=objects, face_objs=face_objs,
                track_id=tid, fi_loc=fi_loc, bbox=bbox,
                frame_diag=frame_diag,
                depth_map=depth_map,
                smooth_mode=cfg.smooth_snap,
            )
        elif cfg.smooth_snap != "off":
            # No object snap but smooth is requested -- just smooth the endpoint
            pass

        persons_gaze.append((c, end, (pitch, yaw)))
        face_confs.append(gc)
        face_bboxes.append(bbox)
        face_track_ids.append(tid)
        ray_snapped.append(snap)
        ray_extended.append(extended)

    return RayFormingResult(
        persons_gaze=persons_gaze,
        face_confs=face_confs,
        face_bboxes=face_bboxes,
        face_track_ids=face_track_ids,
        face_objs=face_objs,
        ray_snapped=ray_snapped,
        ray_extended=ray_extended,
    )
