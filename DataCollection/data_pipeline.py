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

from DataCollection.chart_output import generate_run_charts, resolve_chart_path
from DataCollection.csv_output import resolve_summary_path, write_summary_csv
from DataCollection.dashboard_output import apply_face_anonymization
from DataCollection.heatmap_output import extract_mid_frame, resolve_heatmap_path, save_heatmaps
from pipeline_config import resolve_display_pid


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
        pid_map = ctx.get('pid_map')
        video_name = ctx.get('video_name')
        conditions = ctx.get('conditions', '')
        for ev in hit_events:
            b = ev['bbox']
            row = [frame_no, ev['face_idx'], ev['object'],
                   f"{ev['object_conf']:.3f}",
                   b[0], b[1], b[2], b[3],
                   1 if is_joint else 0,
                   1 if is_confirmed else 0,
                   resolve_display_pid(ev['face_idx'], pid_map)]
            if video_name is not None:
                row = [video_name, conditions] + row
            log_csv.writerow(row)

    if heatmap_path:
        for tid, (_, ray_end, _) in zip(face_track_ids, persons_gaze):
            heatmap_gaze.setdefault(tid, []).append(
                (float(ray_end[0]), float(ray_end[1])))


def finalize_run(ctx, **kwargs) -> None:
    """
    Print run statistics and write post-run output files.

    Reads from ctx
    --------------
    summary_path, total_frames, frame_no, total_hits, look_counts,
    all_trackers, heatmap_path, heatmap_gaze, source, charts_path.

    JA statistics are now printed by JointAttentionTracker.console_summary()
    via post_run_summary(), not here.
    """
    summary_path = ctx.get('summary_path')
    total_frames = ctx.get('total_frames', 0)
    frame_no = ctx.get('frame_no', 0)
    total_hits = ctx.get('total_hits', 0)
    look_counts = ctx.get('look_counts', {})
    all_trackers = ctx.get('all_trackers', [])
    heatmap_path = ctx.get('heatmap_path')
    heatmap_gaze = ctx.get('heatmap_gaze', {})
    source = ctx.get('source')
    pid_map = ctx.get('pid_map')

    print(f"\nDone \u2014 {frame_no} frames, {total_hits} hit events.")

    resolved_summary = resolve_summary_path(summary_path, source)
    if resolved_summary:
        write_summary_csv(
            resolved_summary, total_frames, look_counts,
            all_trackers=all_trackers, pid_map=pid_map,
            video_name=ctx.get('video_name'),
            conditions=ctx.get('conditions', ''))

    resolved_heatmap = resolve_heatmap_path(heatmap_path, source)
    if resolved_heatmap and heatmap_gaze:
        bg = extract_mid_frame(source)
        if bg is None:
            print("Warning: could not extract background frame for heatmap.")
        else:
            anon_mode = ctx.get('anonymize')
            if anon_mode:
                face_det = ctx.get('face_det')
                if face_det is not None:
                    raw_faces = face_det.detect(bg)
                    bboxes = [tuple(int(c) for c in f["bbox"][:4])
                              for f in raw_faces]
                    apply_face_anonymization(
                        bg, bboxes, anon_mode,
                        ctx.get('anonymize_padding', 0.3))
            save_heatmaps(resolved_heatmap, source, bg, heatmap_gaze,
                          pid_map=pid_map)

    # Post-run chart generation
    charts_path = ctx.get('charts_path')
    resolved_charts = resolve_chart_path(charts_path, source)
    if resolved_charts:
        fps = ctx.get('fps', 30.0)
        data_plugins = ctx.get('data_plugins', [])
        generate_run_charts(
            resolved_charts, all_trackers, total_frames, fps,
            pid_map=pid_map, data_plugins=data_plugins,
        )
