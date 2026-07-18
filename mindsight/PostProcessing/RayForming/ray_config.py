"""
RayForming/ray_config.py — Unified configuration for the ray forming pipeline.

Consolidates all ray-related parameters previously scattered across GazeConfig
and DepthConfig into a single dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


def resolve_min_call_gap(ns) -> int:
    """Resolve min_call_gap from a namespace with legacy fallback.

    Precedence: --min-call-gap wins; else the legacy --rf-gazelle-interval
    alias; else the default of 30.  Both flags default to None in argparse
    so "unset" is distinguishable from an explicit value.
    """
    v = getattr(ns, 'min_call_gap', None)
    if v is None:
        v = getattr(ns, 'rf_gazelle_interval', None)
    return 30 if v is None else int(v)


@dataclass
class RayFormingConfig:
    """All ray forming, belief blending, snap, fixation, and hit detection params."""

    # ── Ray geometry ────────────────────────────────────────────────────────
    ray_length: float = 1.0
    conf_ray: bool = False
    forward_gaze_threshold: float = 5.0

    # ── Gazelle blend (fixation-aware scheduler + One Euro smoother) ──────
    # A per-face InferenceScheduler decides when Gaze-LLE inferences are
    # applied to each track's belief map, based on fixation_likelihood
    # computed from smoothed PY velocity and windowed dispersion.  A One
    # Euro Filter adaptively smooths the output direction and length
    # channels -- heavy smoothing at rest (jitter kill), light smoothing
    # during real motion (no lag).  See gazelle_blender.py for the
    # algorithm and docs/superpowers/specs/ for the design rationale.
    #
    # Scheduler knobs.
    fixation_v_threshold: float = 0.04       # rad/frame at 50% velocity fit
    fixation_d_threshold: float = 0.15       # rad windowed dispersion at 50% fit
    min_call_gap: int = 30                   # min frames between Gaze-LLE calls
    # One Euro smoother knobs (per channel; direction and length).
    dir_min_cutoff: float = 1.0              # Hz floor cutoff for direction
    dir_beta: float = 0.5                    # direction speed responsiveness
    len_min_cutoff: float = 1.0              # Hz floor cutoff for length
    len_beta: float = 0.3                    # length speed responsiveness
    # Length-hold decay: after an accepted Gaze-LLE inference latches a
    # length, the length target decays from the latched value back toward
    # the PY baseline via exp(-age / len_hold_tau).  Direction reverts to
    # PY quickly (per-frame trust); length deliberately does NOT -- ray
    # reach is the main pathology the blend fixes, so it persists on a
    # much longer timescale than the instantaneous fixation signal.
    len_hold_tau: float = 5.0                # seconds; length-hold time constant
    # In/out-of-frame gating (v1.1 W3.1).  0.0 = fully inert (1.0.0 behavior:
    # the non-inout architecture is constructed and inout never consulted).
    # > 0 activates the checkpoint's in/out head when present: accepts are
    # VETOED when the fresh inout score falls below the gate (protecting the
    # belief map and length latch from off-screen garbage), and blend trust
    # is attenuated by the cached inout score (PY fixation likelihood is
    # blind to off-screen gaze; this is exactly the missing signal).
    rf_inout_gate: float = 0.0
    # W3X fire-decision knobs (all 0 = off = 1.0.0 behavior).  Consumed by
    # GazelleProvider/InferenceScheduler at construction (read off the ns in
    # from_namespace there); mirrored here so the schema section stays a
    # faithful transcript of the runtime config surface.
    rf_reuse_eps: float = 0.0        # perceptual refire suppression (frame MAD)
    rf_onset_samples: int = 3        # bootstrap fixation warmup override (3.8 default)
    rf_onset_gap: int = 5            # relaxed global gap for new faces (3.8 default)
    # W3Y cheap length channel (0 = off): every N frames a cheap Gaze-LLE
    # pass refreshes ray LENGTH only; direction stays with the
    # fixation-gated fp32 channel.  Consumed by GazelleProvider.
    # Default 10 since the W3Y flip (eval-validated; second re-bless).
    rf_len_refresh_gap: int = 10

    # ── Object snap ─────────────────────────────────────────────────────────
    snap_mode: str = "off"              # "off" | "extend" | "snap"
    snap_dist: float = 150.0
    snap_bbox_scale: float = 0.0
    snap_w_dist: float = 1.0
    snap_w_angle: float = 0.8
    snap_w_size: float = 0.0
    snap_w_intersect: float = 0.5
    snap_w_temporal: float = 0.3
    snap_gate_angle: float = 60.0
    snap_head_blend: float = 0.3
    snap_quality_thresh: float = 0.8
    snap_release_frames: int = 5
    snap_engage_frames: int = 0
    snap_tip_dist: float = -1.0
    snap_tip_quality: float = -1.0
    smooth_snap: str = "off"            # "off" | "objects" | "gaze_tips" | "all"
    smooth_snap_alpha: float = 0.20
    obj_snap_targets: str = "all"       # "all" | "faces_only" | "off"

    # ── Depth integration ───────────────────────────────────────────────────
    depth_ray_length: bool = False
    depth_length_min: float = 0.5
    depth_length_max: float = 3.0
    depth_belief_boost: float = 0.0
    depth_aware_scoring: bool = False
    snap_w_depth: float = 0.0
    gaze_sample_radius: int = 2

    # ── Hit detection ───────────────────────────────────────────────────────
    gaze_tips: bool = False
    tip_radius: int = 80
    gaze_cone_angle: float = 0.0
    hit_conf_gate: float = 0.0
    detect_extend: float = 0.0
    detect_extend_scope: str = "objects"

    @classmethod
    def from_gaze_config(cls, gaze_cfg, depth_cfg=None) -> RayFormingConfig:
        """Construct from legacy GazeConfig (+ optional DepthConfig).

        Maps all existing GazeConfig fields to their RayFormingConfig
        equivalents so that pipelines using the old config continue to work.
        """
        kw = dict(
            ray_length=gaze_cfg.ray_length,
            conf_ray=gaze_cfg.conf_ray,
            forward_gaze_threshold=gaze_cfg.forward_gaze_threshold,
            snap_mode=gaze_cfg.adaptive_ray,
            snap_dist=gaze_cfg.snap_dist,
            snap_bbox_scale=gaze_cfg.snap_bbox_scale,
            snap_w_dist=gaze_cfg.snap_w_dist,
            snap_w_angle=gaze_cfg.snap_w_angle,
            snap_w_size=gaze_cfg.snap_w_size,
            snap_w_intersect=gaze_cfg.snap_w_intersect,
            snap_w_temporal=gaze_cfg.snap_w_temporal,
            snap_gate_angle=gaze_cfg.snap_gate_angle,
            snap_head_blend=gaze_cfg.snap_head_blend,
            snap_quality_thresh=gaze_cfg.snap_quality_thresh,
            snap_tip_dist=gaze_cfg.snap_tip_dist,
            snap_tip_quality=gaze_cfg.snap_tip_quality,
            smooth_snap=gaze_cfg.smooth_snap,
            smooth_snap_alpha=gaze_cfg.smooth_snap_alpha,
            gaze_tips=gaze_cfg.gaze_tips,
            tip_radius=gaze_cfg.tip_radius,
            gaze_cone_angle=gaze_cfg.gaze_cone_angle,
            hit_conf_gate=gaze_cfg.hit_conf_gate,
            detect_extend=gaze_cfg.detect_extend,
            detect_extend_scope=gaze_cfg.detect_extend_scope,
        )
        if depth_cfg is not None:
            kw.update(
                depth_ray_length=getattr(depth_cfg, 'depth_ray_length', False),
                depth_aware_scoring=getattr(depth_cfg, 'depth_aware_scoring', False),
                snap_w_depth=getattr(depth_cfg, 'snap_w_depth', 0.0),
                gaze_sample_radius=getattr(depth_cfg, 'gaze_sample_radius', 2),
            )
        return cls(**kw)

    @classmethod
    def from_namespace(cls, ns) -> RayFormingConfig:
        """Construct from an argparse.Namespace, reading both legacy and new flags."""
        kw = dict(
            ray_length=getattr(ns, 'ray_length', 1.0),
            conf_ray=getattr(ns, 'conf_ray', False),
            forward_gaze_threshold=getattr(ns, 'forward_gaze_threshold', 5.0),
            fixation_v_threshold=getattr(ns, 'fixation_v_threshold', 0.04),
            fixation_d_threshold=getattr(ns, 'fixation_d_threshold', 0.15),
            min_call_gap=resolve_min_call_gap(ns),
            dir_min_cutoff=getattr(ns, 'dir_min_cutoff', 1.0),
            dir_beta=getattr(ns, 'dir_beta', 0.5),
            len_min_cutoff=getattr(ns, 'len_min_cutoff', 1.0),
            len_beta=getattr(ns, 'len_beta', 0.3),
            len_hold_tau=getattr(ns, 'len_hold_tau', 5.0),
            rf_inout_gate=getattr(ns, 'rf_inout_gate', 0.0),
            rf_reuse_eps=getattr(ns, 'rf_reuse_eps', 0.0) or 0.0,
            rf_onset_samples=getattr(ns, 'rf_onset_samples', 3) or 0,
            rf_onset_gap=getattr(ns, 'rf_onset_gap', 5) or 0,
            rf_len_refresh_gap=getattr(ns, 'rf_len_refresh_gap', 10) or 0,
            snap_mode=getattr(ns, 'adaptive_ray', 'off'),
            snap_dist=getattr(ns, 'snap_dist', 150.0),
            snap_bbox_scale=getattr(ns, 'snap_bbox_scale', 0.0),
            snap_w_dist=getattr(ns, 'snap_w_dist', 1.0),
            snap_w_angle=getattr(ns, 'snap_w_angle', 0.8),
            snap_w_size=getattr(ns, 'snap_w_size', 0.0),
            snap_w_intersect=getattr(ns, 'snap_w_intersect', 0.5),
            snap_w_temporal=getattr(ns, 'snap_w_temporal', 0.3),
            snap_gate_angle=getattr(ns, 'snap_gate_angle', 60.0),
            snap_head_blend=getattr(ns, 'snap_head_blend', 0.3),
            snap_quality_thresh=getattr(ns, 'snap_quality_thresh', 0.8),
            snap_release_frames=getattr(ns, 'snap_release_frames', 5),
            snap_engage_frames=getattr(ns, 'snap_engage_frames', 0),
            snap_tip_dist=getattr(ns, 'snap_tip_dist', -1.0),
            snap_tip_quality=getattr(ns, 'snap_tip_quality', -1.0),
            smooth_snap=getattr(ns, 'smooth_snap', 'off'),
            smooth_snap_alpha=getattr(ns, 'smooth_snap_alpha', 0.20),
            obj_snap_targets=getattr(ns, 'obj_snap_targets', 'all'),
            depth_ray_length=getattr(ns, 'depth_ray_length', False),
            depth_length_min=getattr(ns, 'depth_length_min', 0.5),
            depth_length_max=getattr(ns, 'depth_length_max', 3.0),
            depth_belief_boost=getattr(ns, 'depth_belief_boost', 0.0),
            depth_aware_scoring=getattr(ns, 'depth_aware_scoring', False),
            snap_w_depth=getattr(ns, 'snap_w_depth', 0.0),
            gaze_sample_radius=getattr(ns, 'gaze_sample_radius', 2),
            gaze_tips=getattr(ns, 'gaze_tips', False),
            tip_radius=getattr(ns, 'tip_radius', 80),
            gaze_cone_angle=getattr(ns, 'gaze_cone', 0.0),
            hit_conf_gate=getattr(ns, 'hit_conf_gate', 0.0),
            detect_extend=getattr(ns, 'detect_extend', 0.0),
            detect_extend_scope=getattr(ns, 'detect_extend_scope', 'objects'),
        )
        return cls(**kw)
