"""
RayForming/gazelle_blender.py -- Fixation-aware Gaze-LLE blender.

Consumes accept/trust signals from the InferenceScheduler and produces a
smoothed ray endpoint per face per frame.  Preserves the belief-map
Bayesian accumulation (translate + diffuse + multiply-on-accepted-heatmap)
that lets prolonged fixations sharpen their scene anchor across
successive inferences.  Applies a One Euro filter to the direction and
length output channels for adaptive smoothing that responds to real
motion without amplifying jitter.

The blender does NOT decide when to fire Gaze-LLE or when to accept a
heatmap -- those are the scheduler's job.  The blender is a pure
consumer: given (accept: bool, trust: float, dt: float, heatmap: array),
it produces an endpoint.

See docs/superpowers/specs/2026-05-15-gazelle-blend-redesign-design.md.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from mindsight.PostProcessing.RayForming.ray_config import RayFormingConfig
from mindsight.utils.geometry import pitch_yaw_to_2d
from mindsight.utils.one_euro import OneEuroFilter


_GRID = 64
_DIFFUSION_SIGMA = 0.4
"""Fixed belief-map diffusion sigma.

Not user-tunable -- with the InferenceScheduler ensuring inferences
arrive at good times, the exact diffusion rate is not a knob users need
to touch.  0.4 is empirically balanced.
"""

_log = logging.getLogger(__name__)


def _make_gaussian(cx: float, cy: float, sigma: float,
                   size: int = _GRID) -> np.ndarray:
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    g = np.exp(-0.5 * (dy[:, None] ** 2 + dx[None, :] ** 2) / max(sigma, 0.1) ** 2)
    s = g.sum()
    return g / s if s > 1e-12 else np.full((size, size), 1.0 / (size * size), dtype=np.float32)


def _translate_belief(belief: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return belief
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(belief, M, (_GRID, _GRID),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _diffuse(belief: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.1:
        return belief
    ksize = max(3, int(np.ceil(sigma * 3)) | 1)
    return cv2.GaussianBlur(belief, (ksize, ksize), sigma)


def _normalize(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    if s > 1e-12:
        return arr / s
    return np.full_like(arr, 1.0 / arr.size)


def _project_to_grid(pitch: float, yaw: float, origin: np.ndarray,
                     face_width: float, ray_length: float,
                     frame_h: int, frame_w: int) -> tuple[float, float]:
    d = pitch_yaw_to_2d(pitch, yaw)
    px_target = origin + d * (face_width * ray_length)
    gx = float(np.clip(px_target[0] / frame_w * _GRID, 0, _GRID - 1))
    gy = float(np.clip(px_target[1] / frame_h * _GRID, 0, _GRID - 1))
    return gx, gy


def _heatmap_centroid_pixel(hm: np.ndarray,
                            frame_h: int, frame_w: int) -> np.ndarray:
    s = float(hm.sum())
    if s < 1e-12:
        return np.array([frame_w * 0.5, frame_h * 0.5], dtype=float)
    xs = np.arange(_GRID, dtype=np.float32)
    ys = np.arange(_GRID, dtype=np.float32)
    cx = float(np.sum(hm * xs[None, :])) / s
    cy = float(np.sum(hm * ys[:, None])) / s
    return np.array([cx / _GRID * frame_w, cy / _GRID * frame_h], dtype=float)


class GazeLLEBlender:
    """Per-track Gaze-LLE blender with scheduler-driven trust.

    The blender's caller (``ray_pipeline.run_ray_forming``) supplies:
      - ``accept_heatmap``: True iff the scheduler flagged this track as
        wanting the current fresh inference.  Only accepted heatmaps
        update the belief map and re-latch the length target.
      - ``trust``: fixation_likelihood in [0, 1] from the scheduler,
        weighting belief-anchored output vs pure-PY output.
      - ``dt``: seconds between samples; the One Euro filters use it.

    Per-track state:
      - ``_beliefs[tid]``: 64x64 posterior probability map
      - ``_prev_grid[tid]``: last frame's PY-projected grid coords
      - ``_latched_lle_length[tid]``: LLE-derived length latched on the
        last accepted inference
      - ``_latch_age_s[tid]``: seconds since that latch; drives the slow
        exp(-age/len_hold_tau) decay of length back toward the PY baseline
      - ``_dir_x_filter / _dir_y_filter / _len_filter``: One Euro filters
        for the output channels
    """

    def __init__(self, cfg: RayFormingConfig):
        self._cfg = cfg
        self._beliefs: dict[int, np.ndarray] = {}
        self._prev_grid: dict[int, tuple[float, float]] = {}
        self._latched_lle_length: dict[int, float] = {}
        self._latch_age_s: dict[int, float] = {}
        # OneEuroFilter is 1D; direction needs two (x, y).
        self._dir_x_filter: dict[int, OneEuroFilter] = {}
        self._dir_y_filter: dict[int, OneEuroFilter] = {}
        self._len_filter: dict[int, OneEuroFilter] = {}
        self._antipodal_reported: set[int] = set()

    def update(self, *, track_id: int,
               pitch: float, yaw: float, gaze_conf: float,
               origin: np.ndarray, face_width: float,
               frame_h: int, frame_w: int,
               gazelle_hm: np.ndarray | None,
               accept_heatmap: bool, trust: float, dt: float) -> np.ndarray:
        """Update one track for one frame; return the smoothed endpoint."""
        cfg = self._cfg
        py_dir = pitch_yaw_to_2d(pitch, yaw)
        gx, gy = _project_to_grid(pitch, yaw, origin, face_width,
                                  cfg.ray_length, frame_h, frame_w)

        # === Belief map evolution (direction anchor) ===
        belief = self._beliefs.get(track_id)
        if belief is None:
            # No belief yet -- initialize with a PY-centered Gaussian.
            # If trust stays 0 forever the belief centroid never
            # influences the output, so this prior is inert until the
            # first accepted heatmap sharpens it.
            belief = _make_gaussian(gx, gy, sigma=8.0)
        else:
            prev_gp = self._prev_grid.get(track_id, (gx, gy))
            belief = _translate_belief(belief, gx - prev_gp[0], gy - prev_gp[1])
            belief = _diffuse(belief, _DIFFUSION_SIGMA)
            belief = _normalize(belief)

        # Only accepted heatmaps update the belief and re-latch length.
        if accept_heatmap and gazelle_hm is not None:
            belief = belief * (gazelle_hm.astype(np.float32) + 1e-6)
            belief = _normalize(belief)
            # Re-latch length from the raw heatmap centroid (not the
            # belief centroid -- avoids past-frame contamination).
            lle_pixel = _heatmap_centroid_pixel(
                gazelle_hm.astype(np.float32), frame_h, frame_w)
            lle_vec = lle_pixel - origin
            lle_norm = float(np.linalg.norm(lle_vec))
            if lle_norm > 1e-6:
                self._latched_lle_length[track_id] = lle_norm
                self._latch_age_s[track_id] = 0.0

        self._beliefs[track_id] = belief
        self._prev_grid[track_id] = (gx, gy)

        # === Direction target = blend(py_dir, belief_dir, trust) ===
        xs = np.arange(_GRID, dtype=np.float32)
        ys = np.arange(_GRID, dtype=np.float32)
        cx_grid = float(np.sum(belief * xs[None, :]))
        cy_grid = float(np.sum(belief * ys[:, None]))
        belief_target = np.array([cx_grid / _GRID * frame_w,
                                  cy_grid / _GRID * frame_h], dtype=float)
        belief_vec = belief_target - origin
        belief_dist = float(np.linalg.norm(belief_vec))
        raw_belief_dir = belief_vec / belief_dist if belief_dist > 1e-6 else py_dir

        t = float(np.clip(trust, 0.0, 1.0))
        raw_dir_target = py_dir + t * (raw_belief_dir - py_dir)
        rd_norm = float(np.linalg.norm(raw_dir_target))
        if rd_norm > 1e-6:
            raw_dir_target = raw_dir_target / rd_norm
        else:
            # Antipodal pathological case -- fall back to PY, log once per track.
            if track_id not in self._antipodal_reported:
                _log.warning(
                    "GazeLLEBlender: antipodal belief vs PY on track %d; "
                    "falling back to PY", track_id)
                self._antipodal_reported.add(track_id)
            raw_dir_target = py_dir

        # === Length target = hold latched LLE length, slow decay to PY ===
        # Length is deliberately DECOUPLED from per-frame trust.  Direction
        # reverts to PY quickly as trust drops (above); length must not --
        # ray reach is the main pathology the blend fixes, and per-face
        # models carry zero depth information to fall back on.  Instead,
        # each accepted heatmap re-latches the length and resets an age
        # clock; the target then decays latched -> PY with time constant
        # cfg.len_hold_tau (seconds), so reach persists across trust dips
        # and between fixations.
        #
        # PY baseline matches the fallback ray's confidence-scaled length
        # so enabling the blend does not silently change length semantics
        # when conf_ray is on.  Mirrors ray_pipeline's base-ray computation.
        from mindsight.constants import CR_MIN, CR_MAX
        rl = (cfg.ray_length * (CR_MIN + gaze_conf * (CR_MAX - CR_MIN))
              if cfg.conf_ray else cfg.ray_length)
        py_length = float(face_width * rl)
        latched = self._latched_lle_length.get(track_id)
        if latched is None:
            raw_length_target = py_length
        else:
            age = self._latch_age_s.get(track_id, 0.0)
            if not accept_heatmap:
                age += max(dt, 0.0)
                self._latch_age_s[track_id] = age
            tau = max(float(cfg.len_hold_tau), 1e-6)
            hold = float(np.exp(-age / tau))
            raw_length_target = py_length + hold * (latched - py_length)

        # === One Euro smoothing ===
        if track_id not in self._dir_x_filter:
            self._dir_x_filter[track_id] = OneEuroFilter(cfg.dir_min_cutoff, cfg.dir_beta, dt=dt)
            self._dir_y_filter[track_id] = OneEuroFilter(cfg.dir_min_cutoff, cfg.dir_beta, dt=dt)
            self._len_filter[track_id] = OneEuroFilter(cfg.len_min_cutoff, cfg.len_beta, dt=dt)
        dir_x = self._dir_x_filter[track_id]
        dir_y = self._dir_y_filter[track_id]
        len_f = self._len_filter[track_id]

        # Direction x/y are smoothed independently then renormalized.
        # During a rapid gaze reversal the components can briefly desync,
        # producing a short transient swing -- acceptable for a jitter
        # smoother and bounded by the renormalize + py_dir fallback below.
        sx = dir_x.update(float(raw_dir_target[0]))
        sy = dir_y.update(float(raw_dir_target[1]))
        smoothed_dir = np.array([sx, sy], dtype=float)
        n = float(np.linalg.norm(smoothed_dir))
        smoothed_dir = smoothed_dir / n if n > 1e-6 else py_dir

        smoothed_length = len_f.update(float(raw_length_target))

        return origin + smoothed_dir * smoothed_length

    def prune(self, active_tids: set[int]) -> None:
        """Remove state for tracks no longer present."""
        for tid in list(self._beliefs):
            if tid not in active_tids:
                self._beliefs.pop(tid, None)
                self._prev_grid.pop(tid, None)
                self._latched_lle_length.pop(tid, None)
                self._latch_age_s.pop(tid, None)
                self._dir_x_filter.pop(tid, None)
                self._dir_y_filter.pop(tid, None)
                self._len_filter.pop(tid, None)
                self._antipodal_reported.discard(tid)
