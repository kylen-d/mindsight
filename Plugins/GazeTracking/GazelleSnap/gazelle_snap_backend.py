"""
Plugins/GazeTracking/GazelleSnap/gazelle_snap_backend.py — Gazelle-Snap composite backend.

Combines any per-face pitch/yaw gaze backend (MGaze, L2CS, UniGaze) with
periodic Gazelle heatmap inference.  The pitch/yaw backend provides fast
per-frame gaze rays; every *N* frames a Gazelle forward pass produces a
64×64 heatmap whose high-confidence centroid biases the ray endpoint before
the standard adaptive-snap logic is applied.

Activation
----------
Pass ``--gazelle-snap`` together with a pitch/yaw backend flag (e.g.
``--mgaze-model``) **and** ``--gazelle-model``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ms.constants import CR_MAX, CR_MIN, EYE_CONF_THRESH
from ms.GazeTracking.gaze_processing import (
    _faces_as_objects,
    _get_eye_center,
    adaptive_snap,
)
from Plugins import GazePlugin
from ms.utils.geometry import pitch_yaw_to_2d

# ══════════════════════════════════════════════════════════════════════════════
# Heatmap utilities
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_to_target(hm, h, w, threshold_frac):
    """Convert a 64×64 heatmap to a pixel-space snap target.

    Parameters
    ----------
    hm              : (64, 64) numpy float array with values in [0, 1].
    h, w            : Frame dimensions in pixels.
    threshold_frac  : Fraction of peak value used for thresholding.

    Returns
    -------
    (centroid_xy_px, confidence) or (None, 0.0) if below threshold.
    """
    peak = hm.max()
    if peak < 0.05:
        return None, 0.0

    mask = hm >= (peak * threshold_frac)
    if mask.sum() == 0:
        return None, 0.0

    ys, xs = np.where(mask)
    weights = hm[mask]
    cx = np.average(xs, weights=weights) / 64.0 * w
    cy = np.average(ys, weights=weights) / 64.0 * h
    confidence = float(weights.mean())
    return np.array([cx, cy], dtype=float), confidence


# ══════════════════════════════════════════════════════════════════════════════
# Plugin class
# ══════════════════════════════════════════════════════════════════════════════

class GazelleSnapPlugin(GazePlugin):
    """
    Composite gaze plugin: per-face pitch/yaw rays biased by periodic
    Gazelle heatmap inference.
    """

    name = "gazelle_snap"
    mode = "per_face"
    is_fallback = False

    def __init__(self, pitchyaw_engine, gazelle_engine, *,
                 snap_interval=30,
                 heatmap_threshold=0.5,
                 heatmap_weight=1.0,
                 heatmap_decay=0.85,
                 obj_snap="all"):
        self._py_engine = pitchyaw_engine
        self._gz_engine = gazelle_engine
        self._snap_interval = max(1, snap_interval)
        self._hm_threshold = heatmap_threshold
        self._hm_weight = heatmap_weight
        self._hm_decay = heatmap_decay
        self._obj_snap = obj_snap       # "all", "faces_only", or "off"

        # Per-track cached heatmaps and ages
        self._cached_heatmaps: dict[int, np.ndarray] = {}
        self._heatmap_ages: dict[int, int] = {}
        self._frame_counter = 0

    # ── Compatibility: delegate per-face estimate ────────────────────────────

    def estimate(self, face_bgr):
        return self._py_engine.estimate(face_bgr)

    # ── Raw heatmap extraction ───────────────────────────────────────────────

    def _get_raw_heatmaps(self, frame_bgr, face_bboxes_px):
        """Run Gazelle and return raw [N, 64, 64] numpy heatmaps."""
        return self._gz_engine.raw_heatmaps(frame_bgr, face_bboxes_px)

    # ── Core pipeline ────────────────────────────────────────────────────────

    def run_pipeline(self, *, frame, faces, objects, gaze_cfg,
                     smoother=None, snap_hysteresis=None, **kwargs):
        h, w = frame.shape[:2]
        face_confs: list = []
        face_bboxes: list = []

        # ── 1. Per-face pitch/yaw estimation (every frame) ──────────────────
        raw_faces, face_widths, gaze_confs, raw_face_bboxes = [], [], [], []
        for f in faces:
            x1, y1 = max(0, int(f["bbox"][0])), max(0, int(f["bbox"][1]))
            x2, y2 = min(w, int(f["bbox"][2])), min(h, int(f["bbox"][3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pitch, yaw, gc = self._py_engine.estimate(crop)

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

        # ── 2. Conditional Gazelle heatmap inference ────────────────────────
        run_gazelle = (
            raw_face_bboxes
            and self._frame_counter % self._snap_interval == 0
        )
        if run_gazelle:
            heatmaps = self._get_raw_heatmaps(frame, raw_face_bboxes)
            refreshed = set()
            for fi, tid in enumerate(face_track_ids):
                if fi < heatmaps.shape[0]:
                    self._cached_heatmaps[tid] = heatmaps[fi]
                    self._heatmap_ages[tid] = 0
                    refreshed.add(tid)
            # Age heatmaps for tracks not refreshed this frame
            for tid in self._heatmap_ages:
                if tid not in refreshed:
                    self._heatmap_ages[tid] += 1
        else:
            for tid in self._heatmap_ages:
                self._heatmap_ages[tid] += 1

        # Prune stale entries for tracks no longer present
        active_tids = set(face_track_ids)
        for tid in list(self._cached_heatmaps):
            if tid not in active_tids:
                del self._cached_heatmaps[tid]
                self._heatmap_ages.pop(tid, None)

        self._frame_counter += 1

        # ── 3. Ray construction with heatmap snap ───────────────────────────
        ray_length   = gaze_cfg.ray_length
        conf_ray     = gaze_cfg.conf_ray
        adaptive_ray = gaze_cfg.adaptive_ray
        snap_dist    = gaze_cfg.snap_dist

        fwd_thresh_rad = np.radians(gaze_cfg.forward_gaze_threshold)

        persons_gaze, ray_snapped, ray_extended = [], [], []
        for fi_loc, (entry, (pitch, yaw), fw, gc, bbox) in enumerate(zip(
                raw_faces, smoothed, face_widths, gaze_confs,
                raw_face_bboxes)):
            c = entry[0]
            tid = face_track_ids[fi_loc]

            # Forward-gaze dead zone
            if (fwd_thresh_rad > 0
                    and abs(pitch) < fwd_thresh_rad
                    and abs(yaw) < fwd_thresh_rad):
                d_raw = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(yaw)])
                end = c + d_raw * (fw * 0.25)
                persons_gaze.append((c, end, (pitch, yaw)))
                ray_snapped.append(False)
                ray_extended.append(False)
                face_confs.append(gc)
                face_bboxes.append(bbox)
                continue

            d  = pitch_yaw_to_2d(pitch, yaw)
            rl = (ray_length * (CR_MIN + gc * (CR_MAX - CR_MIN))
                  if conf_ray else ray_length)
            fb = c + d * (fw * rl)

            # ── Heatmap snap ────────────────────────────────────────────────
            # When the heatmap confidence is high, snap the ray endpoint
            # directly to the heatmap target (like standalone Gazelle).
            # As confidence decays with age, blend back toward the
            # pitch/yaw ray endpoint.
            hm = self._cached_heatmaps.get(tid)
            if hm is not None:
                ht, hm_conf = _heatmap_to_target(
                    hm, h, w, self._hm_threshold)
                if ht is not None:
                    age = self._heatmap_ages.get(tid, 0)
                    hm_conf *= self._hm_decay ** age
                    blend = min(1.0, self._hm_weight * hm_conf * 2.0)
                    # Only snap if target is in front of the face
                    if np.dot(ht - c, d) > 0 and blend > 0.05:
                        fb = fb * (1.0 - blend) + ht * blend

            # ── Standard adaptive snap with objects ─────────────────────────
            snap, extended = False, False
            other_faces = [fo for fo in face_objs if fo['_face_idx'] != fi_loc]
            if self._obj_snap == "off":
                adaptive_targets = []
            elif self._obj_snap == "faces_only":
                adaptive_targets = other_faces
            else:
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
                    obj_ctr, _ = snap_hysteresis.update(
                        tid, raw_ctr, raw_snap)
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
                    snap_hysteresis.update(tid, None, False)
                end = fb

            persons_gaze.append((c, end, (pitch, yaw)))
            ray_snapped.append(snap)
            ray_extended.append(extended)
            face_confs.append(gc)
            face_bboxes.append(bbox)

        return (persons_gaze, face_confs, face_bboxes, face_track_ids,
                face_objs, ray_snapped, ray_extended)

    # ── CLI protocol ─────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("Gazelle-Snap composite backend")
        g.add_argument(
            "--gazelle-snap", action="store_true", default=False,
            help="Enable the gazelle-snap composite backend (requires a "
                 "pitch/yaw backend AND --gs-gazelle-model).",
        )
        g.add_argument(
            "--gs-gazelle-model", default=None, metavar="PATH",
            help="Path to a Gazelle checkpoint (.pt) for heatmap inference.",
        )
        g.add_argument(
            "--gs-gazelle-name",
            default="gazelle_dinov2_vitb14",
            choices=sorted([
                "gazelle_dinov2_vitb14", "gazelle_dinov2_vitl14",
                "gazelle_dinov2_vitb14_inout", "gazelle_dinov2_vitl14_inout",
            ]),
            metavar="NAME",
            help="Gazelle model variant (default: gazelle_dinov2_vitb14).",
        )
        g.add_argument(
            "--gs-snap-interval", type=int, default=30, metavar="N",
            help="Run Gazelle heatmap inference every N frames (default: 30).",
        )
        g.add_argument(
            "--gs-heatmap-threshold", type=float, default=0.5, metavar="F",
            help="Fraction of heatmap peak for thresholding (default: 0.5).",
        )
        g.add_argument(
            "--gs-heatmap-weight", type=float, default=1.0, metavar="F",
            help="Blend weight toward heatmap target, 0=ignore 1=full "
                 "(default: 1.0).",
        )
        g.add_argument(
            "--gs-heatmap-decay", type=float, default=0.85, metavar="F",
            help="Per-frame confidence decay for stale heatmaps (default: 0.85).",
        )
        g.add_argument(
            "--gs-obj-snap",
            default="all",
            choices=["all", "faces_only", "off"],
            help="Object snap targets: 'all' = YOLO objects + faces, "
                 "'faces_only' = faces and gaze tips only, "
                 "'off' = disable all object snapping (default: all).",
        )

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "gazelle_snap", False):
            return None

        gazelle_ckpt = getattr(args, "gs_gazelle_model", None)
        if not gazelle_ckpt:
            return None

        # ── Instantiate the Gazelle engine ──────────────────────────────────
        from Plugins.GazeTracking.Gazelle.gazelle_backend import (
            GazeEstimationGazelle,
        )
        from ms.weights import resolve_weight
        gazelle_ckpt = Path(resolve_weight("Gazelle", str(gazelle_ckpt)))
        if not gazelle_ckpt.exists():
            raise FileNotFoundError(
                f"Gazelle-Snap checkpoint not found: {gazelle_ckpt}"
            )
        gz_name = getattr(args, "gs_gazelle_name", "gazelle_dinov2_vitb14")
        gz_dev  = getattr(args, "device", "auto")
        gazelle_engine = GazeEstimationGazelle(
            gz_name, gazelle_ckpt, inout_threshold=0.5,
            skip_frames=0, use_fp16=False,
            use_compile=False, device=gz_dev,
        )
        print(f"Backend: Gazelle-Snap  (heatmap: {gz_name})")

        # ── Instantiate the pitch/yaw engine via factory ────────────────────
        # Temporarily clear --gazelle-snap so the factory doesn't recurse
        # back to this plugin.
        from ms.GazeTracking.gaze_factory import create_gaze_engine
        args.gazelle_snap = False
        try:
            pitchyaw_engine = create_gaze_engine(plugin_args=args)
        finally:
            args.gazelle_snap = True

        if getattr(pitchyaw_engine, "mode", None) != "per_face":
            raise ValueError(
                "--gazelle-snap requires a per-face pitch/yaw backend "
                f"(got mode={getattr(pitchyaw_engine, 'mode', '?')})"
            )
        print(f"  pitch/yaw inner backend: {getattr(pitchyaw_engine, 'name', '?')}")

        return cls(
            pitchyaw_engine, gazelle_engine,
            snap_interval=getattr(args, "gs_snap_interval", 30),
            heatmap_threshold=getattr(args, "gs_heatmap_threshold", 0.5),
            heatmap_weight=getattr(args, "gs_heatmap_weight", 1.0),
            heatmap_decay=getattr(args, "gs_heatmap_decay", 0.85),
            obj_snap=getattr(args, "gs_obj_snap", "all"),
        )


# ── Exported symbol consumed by PluginRegistry.discover() ────────────────────
PLUGIN_CLASS = GazelleSnapPlugin
