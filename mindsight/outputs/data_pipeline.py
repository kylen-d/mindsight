"""
outputs/data_pipeline.py — Run-loop data collection step.

Extracts per-frame event logging, look-count accumulation, heatmap data
gathering, and post-run file output (summary CSV, heatmap images) from the
main run loop so they can be extended or replaced by user plugins without
touching MindSight.py.

Usage
-----
    from mindsight.outputs.data_pipeline import collect_frame_data, finalize_run

    # Inside the frame loop:
    collect_frame_data(ctx, log_csv=log_csv, frame_no=frame_no,
                       hit_events=hit_events, face_track_ids=face_track_ids,
                       persons_gaze=persons_gaze)

    # After the loop:
    finalize_run(ctx)
"""

import math

from mindsight.utils.geometry import sample_depth_patch

from mindsight.outputs.chart_output import generate_run_charts, resolve_chart_path
from mindsight.outputs.csv_output import resolve_summary_path, write_summary_tables
from mindsight.outputs.dashboard_output import apply_face_anonymization
from mindsight.outputs.heatmap_output import extract_mid_frame, resolve_heatmap_path, save_heatmaps
from mindsight.pipeline_config import resolve_display_pid


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
        fps = ctx.get('video_fps') or 0.0
        t_seconds = f"{frame_no / fps:.3f}" if fps else ""
        for ev in hit_events:
            b = ev['bbox']
            gaze_conf = ev.get('gaze_conf')
            pitch = ev.get('gaze_pitch')
            yaw = ev.get('gaze_yaw')
            ray_end = ev.get('ray_end')
            depth_at_hit = ev.get('depth_at_gaze')
            row = [frame_no, t_seconds, ev['face_idx'], ev['object'],
                   f"{ev['object_conf']:.3f}",
                   b[0], b[1], b[2], b[3],
                   1 if is_joint else 0,
                   1 if is_confirmed else 0,
                   resolve_display_pid(ev['face_idx'], pid_map),
                   # v1.1 W1.3 additive columns (angles in degrees).
                   f"{gaze_conf:.3f}" if gaze_conf is not None else "",
                   f"{math.degrees(pitch):.2f}" if pitch is not None else "",
                   f"{math.degrees(yaw):.2f}" if yaw is not None else "",
                   f"{ray_end[0]:.1f}" if ray_end is not None else "",
                   f"{ray_end[1]:.1f}" if ray_end is not None else "",
                   f"{depth_at_hit:.4f}" if depth_at_hit is not None else "",
                   1 if ev.get('ray_snapped') else 0,
                   1 if ev.get('ray_extended') else 0]
            if video_name is not None:
                row = [video_name, conditions] + row
            log_csv.writerow(row)

    if heatmap_path:
        for tid, (_, ray_end, _) in zip(face_track_ids, persons_gaze):
            heatmap_gaze.setdefault(tid, []).append(
                (float(ray_end[0]), float(ray_end[1])))

    # Per-frame gaze stream (v1.1 W1.4): one row per face per frame, hits or
    # not -- feeds {stem}_gaze.csv and the eval harness.  Accumulates into
    # the run-level list seeded in run_ctx_base; written by finalize_run.
    gaze_rows = ctx.get('gaze_stream_rows')
    if gaze_rows is not None and persons_gaze:
        pid_map = ctx.get('pid_map')
        fps = ctx.get('video_fps') or 0.0
        t_seconds = f"{frame_no / fps:.3f}" if fps else ""
        face_confs = ctx.get('face_confs', [])
        blend_info = ctx.get('blend_info', [])
        ray_snapped = ctx.get('ray_snapped', [])
        ray_extended = ctx.get('ray_extended', [])
        depth_map = ctx.get('depth_map')
        hit_names: dict = {}
        for ev in hit_events:
            hit_names.setdefault(ev['face_idx'], set()).add(ev['object'])
        for pos, (origin, ray_end, angles) in enumerate(persons_gaze):
            tid = (face_track_ids[pos]
                   if pos < len(face_track_ids) else pos)
            gc = face_confs[pos] if pos < len(face_confs) else None
            blend = blend_info[pos] if pos < len(blend_info) else None
            depth_at_end = (
                sample_depth_patch(depth_map, ray_end[0], ray_end[1])
                if depth_map is not None else None)
            gaze_rows.append([
                frame_no, t_seconds, tid,
                resolve_display_pid(tid, pid_map),
                f"{gc:.3f}" if gc is not None else "",
                f"{math.degrees(angles[0]):.2f}" if angles else "",
                f"{math.degrees(angles[1]):.2f}" if angles else "",
                f"{float(origin[0]):.1f}", f"{float(origin[1]):.1f}",
                f"{float(ray_end[0]):.1f}", f"{float(ray_end[1]):.1f}",
                1 if (pos < len(ray_snapped) and ray_snapped[pos]) else 0,
                1 if (pos < len(ray_extended) and ray_extended[pos]) else 0,
                f"{blend['trust']:.3f}" if blend else "",
                (1 if blend['accepted'] else 0) if blend else "",
                f"{blend['inout']:.3f}" if blend else "",
                f"{depth_at_end:.4f}" if depth_at_end is not None else "",
                ";".join(sorted(hit_names.get(tid, ()))),
            ])

    # Detections side stream (v1.1 W4B validation suite, opt-in via
    # --save-detections): one row per detection per frame, feeding the
    # object-IoU metric.  The list is None unless the flag is on.
    det_rows = ctx.get('detections_stream_rows')
    if det_rows is not None:
        fps = ctx.get('video_fps') or 0.0
        t_seconds = f"{frame_no / fps:.3f}" if fps else ""
        for d in ctx.get('all_dets', []):
            det_rows.append([
                frame_no, t_seconds, d['class_name'],
                f"{float(d['conf']):.3f}",
                d['x1'], d['y1'], d['x2'], d['y2'],
            ])

    # DataCollection plugin per-frame hook (dead until v1.1: instances were
    # built and seeded into ctx but on_frame had no call site anywhere).
    data_plugins = ctx.get('data_plugins', [])
    if data_plugins:
        payload = {**ctx.data, 'frame_no': frame_no,
                   'hit_events': hit_events,
                   'face_track_ids': face_track_ids,
                   'persons_gaze': persons_gaze}
        for plugin in data_plugins:
            plugin.on_frame(**payload)


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

    # DataCollection plugin post-run hook (dead until v1.1, like on_frame).
    for plugin in ctx.get('data_plugins', []):
        plugin.on_run_complete(
            total_frames=total_frames, total_hits=total_hits,
            look_counts=look_counts, source=source,
            all_trackers=all_trackers, pid_map=pid_map,
            video_name=ctx.get('video_name'),
            conditions=ctx.get('conditions', ''),
            fps=ctx.get('video_fps') or 0.0)

    resolved_summary = resolve_summary_path(summary_path, source)
    if resolved_summary:
        fps = ctx.get('video_fps') or 0.0
        write_summary_tables(
            resolved_summary, total_frames, fps, look_counts,
            all_trackers=all_trackers, pid_map=pid_map,
            video_name=ctx.get('video_name'),
            conditions=ctx.get('conditions', ''),
            gaze_stream=ctx.get('gaze_stream_rows'),
            detections_stream=ctx.get('detections_stream_rows'))

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
                          pid_map=pid_map,
                          stem=ctx.get('video_name') or None)

    # Post-run chart generation
    charts_path = ctx.get('charts_path')
    resolved_charts = resolve_chart_path(charts_path, source)
    if resolved_charts:
        fps = ctx.get('video_fps') or ctx.get('fps', 30.0)
        data_plugins = ctx.get('data_plugins', [])
        generate_run_charts(
            resolved_charts, all_trackers, total_frames, fps,
            pid_map=pid_map, data_plugins=data_plugins,
        )
