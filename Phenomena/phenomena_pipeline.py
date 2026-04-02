"""
Phenomena/phenomena_pipeline.py — Run-loop phenomena step.

Extracts tracker initialisation, per-frame updates, and post-run console
summaries from the main run loop so they can be extended or replaced by
user plugins without touching MindSight.py.

All built-in trackers are PhenomenaPlugin subclasses, so they are treated
identically to external plugins.  The pipeline simply iterates a flat list
of tracker instances.

Usage
-----
    from Phenomena.phenomena_pipeline import (
        init_phenomena_trackers,
        update_phenomena_step,
        post_run_summary,
        joint_attention, gaze_convergence,
    )

    ja_tracker, builtin_trackers = init_phenomena_trackers(phenomena_cfg)
    all_trackers = builtin_trackers + active_plugins

    ctx['ja_tracker'] = ja_tracker
    ctx['ja_mode_str'] = ja_mode_str
    ctx['all_trackers'] = all_trackers

    # inside the frame loop:
    update_phenomena_step(ctx)

    # after the loop:
    post_run_summary(all_trackers, total_frames)
"""

from Phenomena.Default import (
    JointAttentionTemporalTracker,
    MutualGazeTracker, SocialReferenceTracker, GazeFollowingTracker,
    GazeAversionTracker, ScanpathTracker, GazeLeadershipTracker,
    AttentionSpanTracker,
)
from Phenomena.helpers import joint_attention, gaze_convergence
from Phenomena.phenomena_config import PhenomenaConfig


def init_phenomena_trackers(cfg: PhenomenaConfig):
    """
    Instantiate all phenomena trackers based on the provided config.

    Returns
    -------
    ja_tracker : JointAttentionTemporalTracker | None
        Temporal joint-attention filter (not a PhenomenaPlugin — it gates
        joint_objs before other trackers see them).
    builtin_trackers : list[PhenomenaPlugin]
        Active built-in tracker instances.  Disabled trackers are omitted.
        The list order determines dashboard panel rendering order.
    """
    ja_tracker = (JointAttentionTemporalTracker(cfg.ja_window, cfg.ja_window_thresh)
                  if cfg.joint_attention and cfg.ja_window > 0 else None)

    # Build list in display order: left-panel trackers first, then right-panel.
    builtin_trackers = []

    if cfg.mutual_gaze:
        builtin_trackers.append(MutualGazeTracker())
    if cfg.social_ref:
        builtin_trackers.append(SocialReferenceTracker(cfg.social_ref_window))
    if cfg.gaze_follow:
        builtin_trackers.append(GazeFollowingTracker(cfg.gaze_follow_lag))
    if cfg.attn_span:
        builtin_trackers.append(AttentionSpanTracker())
    if cfg.gaze_aversion:
        builtin_trackers.append(GazeAversionTracker(cfg.aversion_window, cfg.aversion_conf))
    if cfg.scanpath:
        builtin_trackers.append(ScanpathTracker(cfg.scanpath_dwell))
    if cfg.gaze_leader:
        builtin_trackers.append(GazeLeadershipTracker())

    return ja_tracker, builtin_trackers


def update_phenomena_step(ctx, **kwargs):
    """
    Update all active phenomena trackers for one frame.

    Reads from ctx
    --------------
    frame_no, persons_gaze, face_bboxes, hit_events, joint_objs,
    objects, face_track_ids, hits, ja_tracker, ja_mode_str, all_trackers.

    Writes to ctx
    -------------
    confirmed_objs : set — joint attention after temporal confirmation.
    extra_hud      : str | None — window-fill status for HUD.
    """
    frame_no = ctx['frame_no']
    persons_gaze = ctx['persons_gaze']
    face_bboxes = ctx['face_bboxes']
    hit_events = ctx['hit_events']
    joint_objs = ctx['joint_objs']
    dets = ctx['objects']
    face_track_ids = ctx.get('face_track_ids')
    hits = ctx.get('hits')
    ja_tracker = ctx.get('ja_tracker')
    ja_mode_str = ctx.get('ja_mode_str')
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

    # Temporal joint-attention confirmation
    if ja_tracker is not None:
        confirmed_objs = ja_tracker.update(joint_objs)
        win_fill_pct   = ja_tracker.fill * 100
        extra_hud = (f"{ja_mode_str}  win:{win_fill_pct:.0f}%"
                     if ja_mode_str else f"win:{win_fill_pct:.0f}%")
    else:
        confirmed_objs = joint_objs
        extra_hud      = ja_mode_str

    # Update all trackers uniformly (built-in + external plugins)
    tracker_kwargs = dict(
        frame_no=frame_no, persons_gaze=persons_gaze, face_bboxes=face_bboxes,
        hit_events=hit_events, joint_objs=joint_objs, dets=dets,
        n_faces=n_faces, face_track_ids=face_track_ids, hits=hits_set,
    )
    for tracker in all_trackers:
        tracker.update(**tracker_kwargs)

    ctx['confirmed_objs'] = confirmed_objs
    ctx['extra_hud'] = extra_hud


def post_run_summary(all_trackers: list, total_frames: int) -> None:
    """
    Print post-run phenomena summaries to stdout for all active trackers.
    """
    for tracker in all_trackers:
        summary = tracker.console_summary(total_frames)
        if summary:
            print(summary)
