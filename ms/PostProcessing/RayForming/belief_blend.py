"""
RayForming/belief_blend.py — Bayesian belief map blending.

Maintains a per-track 64x64 belief map that fuses continuous pitch/yaw
temporal tracking with periodic Gaze-LLE spatial correction using principled
spatial fusion.

Algorithm overview (per frame, per track):
  A. Temporal propagation: shift belief map by pitch/yaw delta, diffuse.
  B. Pitch/yaw prior: generate confidence-scaled Gaussian at PY target.
  C. Combine prior into propagated belief.
  D. Gaze-LLE update (every N frames): multiply belief by heatmap likelihood.
  E. Optional depth modulation of heatmap before update.
  F. Extract target centroid and confidence from belief map.
"""
from __future__ import annotations

import cv2
import numpy as np

from ms.PostProcessing.RayForming.ray_config import RayFormingConfig
from ms.utils.geometry import pitch_yaw_to_2d, sample_depth_patch

_GRID = 64  # Gaze-LLE heatmap resolution


def _make_gaussian(cx: float, cy: float, sigma: float, size: int = _GRID) -> np.ndarray:
    """Create a 2D Gaussian centered at (cx, cy) on a size x size grid."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    g = np.exp(-0.5 * (dy[:, None] ** 2 + dx[None, :] ** 2) / max(sigma, 0.1) ** 2)
    s = g.sum()
    return g / s if s > 1e-12 else np.full((size, size), 1.0 / (size * size), dtype=np.float32)


def _translate_belief(belief: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate the belief map by (dx, dy) grid cells using affine warp."""
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return belief
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(belief, M, (_GRID, _GRID),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def _diffuse(belief: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to model growing spatial uncertainty."""
    if sigma < 0.1:
        return belief
    ksize = max(3, int(np.ceil(sigma * 3)) | 1)
    return cv2.GaussianBlur(belief, (ksize, ksize), sigma)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to sum to 1, or uniform if near-zero."""
    s = arr.sum()
    if s > 1e-12:
        return arr / s
    return np.full_like(arr, 1.0 / arr.size)


def _project_to_grid(pitch: float, yaw: float, origin: np.ndarray,
                     face_width: float, ray_length: float,
                     frame_h: int, frame_w: int,
                     conf_ray: bool, gaze_conf: float,
                     cr_min: float, cr_max: float) -> tuple[float, float]:
    """Project a pitch/yaw gaze direction to a point on the 64x64 grid."""
    d = pitch_yaw_to_2d(pitch, yaw)
    rl = ray_length * (cr_min + gaze_conf * (cr_max - cr_min)) if conf_ray else ray_length
    px_target = origin + d * (face_width * rl)
    gx = float(np.clip(px_target[0] / frame_w * _GRID, 0, _GRID - 1))
    gy = float(np.clip(px_target[1] / frame_h * _GRID, 0, _GRID - 1))
    return gx, gy


class BeliefBlender:
    """Per-track belief map fusing pitch/yaw prior with Gaze-LLE likelihood.

    The belief map is a 64x64 array representing the system's current
    spatial estimate of where the person is looking.  Each frame it is
    updated from two signals:

    - **Pitch/yaw** (every frame): temporal tracking via a translated
      Gaussian prior.
    - **Gaze-LLE heatmap** (every N frames): scene-aware spatial
      correction via element-wise likelihood product.

    Config parameters used:
      - ``direction_decay`` — controls how quickly the PY prior steers the
        belief between Gazelle updates (0 = no PY influence, 1 = PY dominates).
      - ``length_decay`` — diffusion sigma for inter-frame uncertainty growth.
        Higher = belief spreads faster between updates.
      - ``blend_conf_scale`` — scales PY prior sigma by gaze confidence.
      - ``belief_min_peak`` — minimum heatmap peak to accept.
      - ``inout_threshold`` — suppress heatmap when in/out score is below this.
    """

    def __init__(self, cfg: RayFormingConfig):
        self._cfg = cfg
        self._beliefs: dict[int, np.ndarray] = {}
        self._prev_grid: dict[int, tuple[float, float]] = {}
        self._has_gazelle: dict[int, bool] = {}  # whether track has had a Gazelle update
        self._smooth_target: dict[int, np.ndarray] = {}  # EMA-smoothed output target

        # Map config fields to algorithm parameters:
        # direction_decay (0-1): PY prior blend strength.
        # Only used before first Gazelle update to shape the initial belief.
        # After Gazelle updates, PY influence comes via translation (step A).
        self._py_blend = cfg.direction_decay
        # Belief map diffusion: per-frame Gaussian blur sigma.
        # Controls how fast the belief spreads between Gazelle updates,
        # which drives confidence decay via the log-ratio metric.
        self._diffusion_sigma = cfg.diffusion_sigma
        # blend_conf_scale: how much gaze confidence tightens the PY prior.
        self._conf_scale = cfg.blend_conf_scale

    def update(self, track_id: int,
               pitch: float, yaw: float, gaze_conf: float,
               origin: np.ndarray, face_width: float,
               frame_h: int, frame_w: int,
               gazelle_hm: np.ndarray | None = None,
               inout_score: float = 1.0,
               depth_map: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Update belief for one track and return ``(target_xy_px, confidence)``.

        Parameters
        ----------
        track_id   : stable face track ID
        pitch, yaw : smoothed gaze angles in radians
        gaze_conf  : pitch/yaw confidence in [0, 1]
        origin     : eye center in pixel coordinates
        face_width : detected face width in pixels
        frame_h/w  : frame dimensions
        gazelle_hm : fresh 64x64 Gaze-LLE heatmap, or None if not this frame
        inout_score : Gaze-LLE in/out-of-frame score (1.0 = in frame)
        depth_map  : optional HxW normalized depth map

        Returns
        -------
        target_xy : pixel-space (x, y) target from belief centroid
        confidence : peak value of the belief map (0-1)
        """
        cfg = self._cfg
        from ms.constants import CR_MIN, CR_MAX

        # Project current pitch/yaw to grid coordinates
        gx, gy = _project_to_grid(
            pitch, yaw, origin, face_width, cfg.ray_length,
            frame_h, frame_w, cfg.conf_ray, gaze_conf, CR_MIN, CR_MAX)

        belief = self._beliefs.get(track_id)

        has_gazelle = self._has_gazelle.get(track_id, False)

        if belief is None:
            # Initialize belief from pitch/yaw prior
            sigma = 8.0 * (1.0 - gaze_conf * self._conf_scale)
            belief = _make_gaussian(gx, gy, sigma)
        else:
            # A. Temporal propagation: shift belief by PY delta.
            # This is how PY provides temporal tracking -- the Gazelle-
            # corrected distribution moves along with PY gaze movement.
            prev = self._prev_grid.get(track_id, (gx, gy))
            dx = gx - prev[0]
            dy = gy - prev[1]
            belief = _translate_belief(belief, dx, dy)
            belief = _diffuse(belief, self._diffusion_sigma)
            belief = _normalize(belief)

            # B. PY prior blend: only before the first Gazelle update.
            # Once Gazelle has corrected the belief, we don't want PY to
            # pull the centroid back every frame -- translation (step A)
            # is sufficient for temporal tracking.  Before any Gazelle data,
            # the PY prior shapes and steers the belief.
            if not has_gazelle:
                sigma = 8.0 * (1.0 - gaze_conf * self._conf_scale)
                py_prior = _make_gaussian(gx, gy, sigma)
                a = self._py_blend
                belief = _normalize(belief ** (1.0 - a) * py_prior ** a)

        # C. Gaze-LLE likelihood update (when fresh heatmap available)
        if gazelle_hm is not None:
            if inout_score >= cfg.inout_threshold:
                peak = float(gazelle_hm.max())
                if peak >= cfg.belief_min_peak:
                    hm = gazelle_hm.astype(np.float32)

                    # D. Optional depth modulation
                    if (depth_map is not None
                            and cfg.depth_belief_boost > 0):
                        px_target = origin + pitch_yaw_to_2d(pitch, yaw) * face_width
                        d_at_py = sample_depth_patch(
                            depth_map, float(px_target[0]), float(px_target[1]),
                            radius=cfg.gaze_sample_radius)
                        depth_grid = cv2.resize(depth_map, (_GRID, _GRID),
                                                interpolation=cv2.INTER_LINEAR)
                        diff = (depth_grid.astype(np.float32) - d_at_py) ** 2
                        agreement = np.exp(-diff / max(0.04, 2.0 * 0.1 ** 2))
                        hm = hm * (1.0 + cfg.depth_belief_boost * agreement)

                    # Bayesian update: belief x likelihood, then renormalize
                    belief = belief * (hm + 1e-6)
                    belief = _normalize(belief)
                    self._has_gazelle[track_id] = True

        # Store state
        self._beliefs[track_id] = belief
        self._prev_grid[track_id] = (gx, gy)

        # E. Extract target: weighted centroid of belief map
        xs = np.arange(_GRID, dtype=np.float32)
        ys = np.arange(_GRID, dtype=np.float32)
        cx_grid = float(np.sum(belief * xs[None, :]))
        cy_grid = float(np.sum(belief * ys[:, None]))
        target_x = cx_grid / _GRID * frame_w
        target_y = cy_grid / _GRID * frame_h
        # Confidence: how much sharper the peak is vs. a uniform distribution.
        uniform_peak = 1.0 / (_GRID * _GRID)
        raw_peak = float(belief.max())
        ratio = raw_peak / uniform_peak  # 1.0 = uniform, ~72 = very sharp
        if ratio > 1.0:
            confidence = min(1.0, np.log(ratio) / np.log(70.0))
        else:
            confidence = 0.0

        # F. Smooth the output with independent direction/length EMA.
        # Decompose the belief centroid into direction and distance from
        # origin, smooth each independently, then reconstruct.  This lets
        # ray reach persist longer than directional correction.
        #   direction_decay → direction EMA alpha (lower = smoother direction)
        #   length_decay    → length EMA alpha (lower = smoother length)
        raw_target = np.array([target_x, target_y], dtype=float)
        raw_vec = raw_target - origin
        raw_dist = float(np.linalg.norm(raw_vec))
        raw_dir = raw_vec / raw_dist if raw_dist > 1e-6 else raw_vec

        prev = self._smooth_target.get(track_id)
        if prev is not None:
            prev_vec = prev - origin
            prev_dist = float(np.linalg.norm(prev_vec))
            prev_dir = prev_vec / prev_dist if prev_dist > 1e-6 else prev_vec

            # Smooth direction (higher direction_decay = faster response)
            dir_alpha = self._py_blend
            sm_dir = prev_dir + dir_alpha * (raw_dir - prev_dir)
            sm_dir_n = float(np.linalg.norm(sm_dir))
            sm_dir = sm_dir / sm_dir_n if sm_dir_n > 1e-6 else raw_dir

            # Smooth length (higher length_decay = faster response)
            # Use a smaller alpha so length persists longer
            len_alpha = cfg.length_decay
            sm_dist = prev_dist + len_alpha * (raw_dist - prev_dist)

            smoothed = origin + sm_dir * sm_dist
        else:
            smoothed = raw_target.copy()
        self._smooth_target[track_id] = smoothed.copy()

        return smoothed, confidence

    def prune(self, active_tids: set[int]) -> None:
        """Remove state for tracks no longer present."""
        for tid in list(self._beliefs):
            if tid not in active_tids:
                del self._beliefs[tid]
                self._prev_grid.pop(tid, None)
                self._has_gazelle.pop(tid, None)
                self._smooth_target.pop(tid, None)
