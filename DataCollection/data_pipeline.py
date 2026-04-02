"""
DataCollection/data_pipeline.py — Run-loop data collection step.

Extracts per-frame event logging, look-count accumulation, heatmap data
gathering, and post-run file output (summary CSV, heatmap images) from the
main run loop so they can be extended or replaced by user plugins without
touching MindSight.py.

Usage
-----
    from DataCollection.data_pipeline import collect_frame_data, finalize_run

    # Inside the frame loop:
    collect_frame_data(ctx, log_csv=log_csv, frame_no=frame_no,
                       hit_events=hit_events, face_track_ids=face_track_ids,
                       persons_gaze=persons_gaze)

    # After the loop:
    finalize_run(ctx)
"""

from DataCollection.csv_output import write_summary_csv, resolve_summary_path
from DataCollection.heatmap_output import extract_mid_frame, save_heatmaps, resolve_heatmap_path


def collect_frame_data(ctx, *, log_csv, frame_no: int,
                       hit_events: list, face_track_ids: list,
                       persons_gaze: list, **kwargs) -> None:
    """
    Accumulate per-frame data and write to open log.

    Parameters
    ----------
    ctx            : FrameContext (reads is_joint, is_confirmed, look_counts,
                     heatmap_path, heatmap_gaze).
    log_csv        : csv.writer instance or None.
    frame_no       : Current frame index.
    hit_events     : list[dict]  per-hit records (face_idx, object, bbox, …).
    face_track_ids : list[int]  stable track IDs in same order as persons_gaze.
    persons_gaze   : list of (origin, ray_end, angles) tuples.
    """
    is_joint = ctx.get('is_joint', False)
    is_confirmed = ctx.get('is_confirmed', False)
    look_counts = ctx.get('look_counts', {})
    heatmap_path = ctx.get('heatmap_path')
    heatmap_gaze = ctx.get('heatmap_gaze', {})

    for face_idx, obj_cls in {(ev['face_idx'], ev['object']) for ev in hit_events}:
        look_counts[(face_idx, obj_cls)] = look_counts.get((face_idx, obj_cls), 0) + 1

    if log_csv is not None:
        for ev in hit_events:
            b = ev['bbox']
            log_csv.writerow([frame_no, ev['face_idx'], ev['object'],
                              f"{ev['object_conf']:.3f}",
                              b[0], b[1], b[2], b[3],
                              1 if is_joint else 0,
                              1 if is_confirmed else 0])

    if heatmap_path:
        for tid, (_, ray_end, _) in zip(face_track_ids, persons_gaze):
            heatmap_gaze.setdefault(tid, []).append(
                (float(ray_end[0]), float(ray_end[1])))


def finalize_run(ctx, **kwargs) -> None:
    """
    Print run statistics and write post-run output files.

    Reads from ctx
    --------------
    summary_path, total_frames, joint_frames, confirmed_frames,
    frame_no, total_hits, look_counts, all_trackers,
    heatmap_path, heatmap_gaze, source, ja_tracker.
    """
    summary_path = ctx.get('summary_path')
    total_frames = ctx.get('total_frames', 0)
    joint_frames = ctx.get('joint_frames', 0)
    confirmed_frames = ctx.get('confirmed_frames', 0)
    frame_no = ctx.get('frame_no', 0)
    total_hits = ctx.get('total_hits', 0)
    look_counts = ctx.get('look_counts', {})
    all_trackers = ctx.get('all_trackers', [])
    heatmap_path = ctx.get('heatmap_path')
    heatmap_gaze = ctx.get('heatmap_gaze', {})
    source = ctx.get('source')
    ja_tracker = ctx.get('ja_tracker')

    pct      = joint_frames     / total_frames * 100 if total_frames else 0.0
    conf_pct = confirmed_frames / total_frames * 100 if total_frames else 0.0

    print(f"\nDone \u2014 {frame_no} frames, {total_hits} hit events.")
    print(f"Joint attention (raw):       {joint_frames}/{total_frames} frames = {pct:.1f}%")
    if ja_tracker is not None:
        print(f"Joint attention (confirmed): "
              f"{confirmed_frames}/{total_frames} frames = {conf_pct:.1f}%")

    resolved_summary = resolve_summary_path(summary_path, source)
    if resolved_summary:
        write_summary_csv(
            resolved_summary, total_frames, confirmed_frames, look_counts,
            all_trackers=all_trackers)

    resolved_heatmap = resolve_heatmap_path(heatmap_path, source)
    if resolved_heatmap and heatmap_gaze:
        bg = extract_mid_frame(source)
        if bg is None:
            print("Warning: could not extract background frame for heatmap.")
        else:
            save_heatmaps(resolved_heatmap, source, bg, heatmap_gaze)
