"""
Phenomena/phenomena_pipeline.py — Run-loop phenomena step.

Extracts tracker initialisation, per-frame updates, and post-run console
summaries from the main run loop so they can be extended or replaced by
user plugins without touching MindSight.py.

All built-in trackers are PhenomenaPlugin subclasses, so they are treated
identically to external plugins.  The pipeline simply iterates a flat list
of tracker instances.

JointAttentionTracker is always first in the list because subsequent
trackers depend on ``confirmed_objs`` that JA computes.

Usage
-----
    from Phenomena.phenomena_pipeline import (
        init_phenomena_trackers,
        update_phenomena_step,
        post_run_summary,
        joint_attention, gaze_convergence,
    )

    builtin_trackers = init_phenomena_trackers(phenomena_cfg)
    all_trackers = builtin_trackers + active_plugins

    ctx['all_trackers'] = all_trackers

    # inside the frame loop:
    update_phenomena_step(ctx)

    # after the loop:
    post_run_summary(all_trackers, total_frames)
"""

from Phenomena.Default import (
    AttentionSpanTracker,
    GazeAversionTracker,
    GazeFollowingTracker,
    GazeLeadershipTracker,
    JointAttentionTracker,
    MutualGazeTracker,
    ScanpathTracker,
    SocialReferenceTracker,
)
from Phenomena.phenomena_config import PhenomenaConfig


def init_phenomena_trackers(cfg: PhenomenaConfig):
    """
    Instantiate all phenomena trackers based on the provided config.

    Returns
    -------
    builtin_trackers : list[PhenomenaPlugin]
        Active built-in tracker instances.  JointAttentionTracker is always
        first (when enabled) so that confirmed_objs is available to
        subsequent trackers.  The list order determines dashboard panel
        rendering order.
    """
    builtin_trackers = []

    # JA tracker is always first — other trackers depend on confirmed_objs.
    if cfg.joint_attention:
        ja = JointAttentionTracker(
            window=cfg.ja_window,
            threshold=cfg.ja_window_thresh,
            quorum=cfg.ja_quorum,
        )
        builtin_trackers.append(ja)

    # Left-panel trackers
    if cfg.mutual_gaze:
        builtin_trackers.append(MutualGazeTracker())
    if cfg.social_ref:
        builtin_trackers.append(SocialReferenceTracker(cfg.social_ref_window))
    if cfg.gaze_follow:
        builtin_trackers.append(GazeFollowingTracker(cfg.gaze_follow_lag))

    # Right-panel trackers
    if cfg.attn_span:
        builtin_trackers.append(AttentionSpanTracker())
    if cfg.gaze_aversion:
        builtin_trackers.append(GazeAversionTracker(cfg.aversion_window, cfg.aversion_conf))
    if cfg.scanpath:
        builtin_trackers.append(ScanpathTracker(cfg.scanpath_dwell))
    if cfg.gaze_leader:
        builtin_trackers.append(GazeLeadershipTracker(
            tip_mode=cfg.gaze_leader_tips,
            tip_lag=cfg.gaze_leader_tip_lag,
        ))

    return builtin_trackers


def update_phenomena_step(ctx, **kwargs):
    """
    Update all active phenomena trackers for one frame.

    Reads from ctx
    --------------
    frame_no, persons_gaze, face_bboxes, hit_events, joint_objs,
    objects, face_track_ids, hits, all_trackers.

    Writes to ctx
    -------------
    confirmed_objs : set — joint attention after temporal confirmation.
    extra_hud      : str | None — window-fill status for HUD.
    joint_pct      : float — running JA percentage.
    """
    frame_no = ctx['frame_no']
    persons_gaze = ctx['persons_gaze']
    face_bboxes = ctx['face_bboxes']
    hit_events = ctx['hit_events']
    joint_objs = ctx['joint_objs']
    dets = ctx['objects']
    face_track_ids = ctx.get('face_track_ids')
    hits = ctx.get('hits')
    all_trackers = ctx.get('all_trackers', [])

    n_faces = len(persons_gaze)

    # Use pre-computed hits when available; fall back to reconstruction.
    if hits is not None:
        hits_set = hits
    else:
        hits_set = set()
        for ev in hit_events:
            for oi, d in enumerate(dets):
                if (d['class_name'] == ev['object'] and
                        d['x1'] == ev['bbox'][0] and d['y1'] == ev['bbox'][1]):
                    hits_set.add((ev['face_idx'], oi))
                    break

    # Update all trackers uniformly (built-in + external plugins).
    # JA tracker (if present) is first and sets confirmed_objs/extra_hud.
    tracker_kwargs = dict(
        frame_no=frame_no, persons_gaze=persons_gaze, face_bboxes=face_bboxes,
        hit_events=hit_events, joint_objs=joint_objs, dets=dets,
        n_faces=n_faces, face_track_ids=face_track_ids, hits=hits_set,
        tip_convergences=ctx.get('tip_convergences', []),
        tip_radius=ctx.get('tip_radius', 50),
        detect_extend=ctx.get('detect_extend', 0.0),
        detect_extend_scope=ctx.get('detect_extend_scope', 'objects'),
        pid_map=ctx.get('pid_map'),
        fps=ctx.get('fps', 0.0),
        joint_pct=ctx.get('joint_pct', 0.0),
        n_dets=ctx.get('n_dets', 0),
        _all_trackers=all_trackers,
    )

    confirmed_objs = joint_objs  # default: no temporal filter
    extra_hud = None
    joint_pct = ctx.get('joint_pct', 0.0)

    for tracker in all_trackers:
        result = tracker.update(**tracker_kwargs)

        # JA tracker writes confirmed_objs and extra_hud via its return value
        if result and 'confirmed_objs' in result:
            confirmed_objs = result['confirmed_objs']
            extra_hud = result.get('extra_hud')
            joint_pct = result.get('joint_pct', joint_pct)
            # Update kwargs so subsequent trackers see confirmed JA
            tracker_kwargs['joint_objs'] = confirmed_objs

    ctx['confirmed_objs'] = confirmed_objs
    ctx['extra_hud'] = extra_hud
    ctx['joint_pct'] = joint_pct


def post_run_summary(all_trackers: list, total_frames: int,
                     pid_map=None) -> None:
    """
    Print post-run phenomena summaries to stdout for all active trackers.
    """
    for tracker in all_trackers:
        summary = tracker.console_summary(total_frames, pid_map=pid_map)
        if summary:
            print(summary)
