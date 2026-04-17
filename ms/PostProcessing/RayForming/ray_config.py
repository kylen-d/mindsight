"""
RayForming/ray_config.py — Unified configuration for the ray forming pipeline.

Consolidates all ray-related parameters previously scattered across GazeConfig,
GazelleSnap constructor args, and DepthConfig into a single dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RayFormingConfig:
    """All ray forming, belief blending, snap, fixation, and hit detection params."""

    # ── Ray geometry ────────────────────────────────────────────────────────
    ray_length: float = 1.0
    conf_ray: bool = False
    forward_gaze_threshold: float = 5.0

    # ── Gazelle blend ───────────────────────────────────────────────────────
    # Independent direction and length blend strengths.
    # 0.0 = pure pitch/yaw, 1.0 = full Gazelle correction.
    # When no Gazelle model is loaded these are effectively 0.0.
    blend_strength: float = 1.0         # legacy: sets both if dir/len not specified
    direction_blend: float = 1.0        # direction blend strength
    length_blend: float = 1.0           # length/reach blend strength
    length_only: bool = False           # shortcut: sets direction_blend=0
    direction_decay: float = 0.30       # direction EMA response rate
    length_decay: float = 0.15          # length EMA response rate (lower = more persistent)
    diffusion_sigma: float = 0.40       # per-frame belief map blur sigma
    blend_conf_scale: float = 0.70      # PY prior tightening by gaze confidence
    belief_min_peak: float = 0.05       # min Gaze-LLE heatmap peak to accept
    inout_threshold: float = 0.5        # suppress heatmap when in/out < this

    # ── Gazelle scheduling ──────────────────────────────────────────────────
    gazelle_interval: int = 30          # frames between Gaze-LLE inferences

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
            blend_strength=getattr(ns, 'blend_strength', 1.0),
            direction_blend=getattr(ns, 'direction_blend',
                             getattr(ns, 'blend_strength', 1.0)),
            length_blend=getattr(ns, 'length_blend',
                          getattr(ns, 'blend_strength', 1.0)),
            length_only=getattr(ns, 'length_only', False),
            direction_decay=getattr(ns, 'direction_decay', 0.30),
            length_decay=getattr(ns, 'length_decay', 0.15),
            diffusion_sigma=getattr(ns, 'diffusion_sigma', 0.40),
            blend_conf_scale=getattr(ns, 'blend_conf_scale', 0.7),
            belief_min_peak=getattr(ns, 'belief_min_peak', 0.05),
            inout_threshold=getattr(ns, 'inout_threshold', 0.5),
            gazelle_interval=getattr(ns, 'rf_gazelle_interval',
                              getattr(ns, 'gs_snap_interval', 30)),
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
            obj_snap_targets=getattr(ns, 'gs_obj_snap',
                              getattr(ns, 'obj_snap_targets', 'all')),
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
