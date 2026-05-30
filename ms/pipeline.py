"""
ms.pipeline -- Extracted MindSight run loop (frame loop + per-frame stages).

This module holds the code that was historically embedded in ``ms.cli``: the
per-frame ``process_frame`` orchestrator, the auxiliary-stream helpers, the
static-image path, and the video/webcam ``run`` loop.  ``ms.cli`` now imports
``run`` from here and remains the thin CLI/model-wiring layer.

The extraction is pure code motion (SP1.2 step 1): behavior is byte-identical
to the previous in-cli implementation.  A GUI-consumable ``Pipeline`` class and
``FrameResult`` API are layered on top in a later step.

All pipeline stages communicate through a shared ``FrameContext`` object; each
stage reads the keys it needs and writes its results back.
"""

import time
from collections import deque
from pathlib import Path

import cv2

from ms.constants import IMAGE_EXTS
from ms.DataCollection.dashboard_output import (
    AnonSmoother,
    apply_face_anonymization,
    compose_dashboard,
    draw_overlay,
)
from ms.DataCollection.data_pipeline import collect_frame_data, finalize_run
from ms.io.sources import (
    enrich_aux_with_face_detection,
    open_aux_streams,
    open_video_source,
    read_aux_frames,
    read_image_source,
)
from ms.io.writers import finalize_video, open_event_log, open_video_writer
from ms.GazeTracking.gaze_pipeline import run_gaze_step
from ms.GazeTracking.gaze_processing import (
    GazeLockTracker,
    GazeSmootherReID,
    SmoothSnapTracker,
    SnapTemporalState,
)
from ms.ObjectDetection.detection_pipeline import run_detection_step
from ms.ObjectDetection.object_detection import ObjectPersistenceCache
from ms.Phenomena.phenomena_pipeline import (
    gaze_convergence,
    init_phenomena_trackers,
    joint_attention,
    post_run_summary,
    update_phenomena_step,
)
from ms.pipeline_config import (
    DetectionConfig,
    FrameContext,
    GazeConfig,
    OutputConfig,
    TrackerConfig,
    resolve_display_pid,
)
from ms.PostProcessing.RayForming import GazeLLEBlender, ObjectSnap


# ==============================================================================
# Core per-frame processing  (thin orchestrator -> logic lives in pipeline files)
# ==============================================================================

def process_frame(ctx, *, yolo, face_det, gaze_eng,
                  gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
                  obj_cache=None, phenomena_cfg=None,
                  detection_plugins=None,
                  depth_cfg=None, depth_backend=None):
    """
    Process one frame through detection, gaze, JA, and overlay stages.

    Reads from ctx: frame (required), plus optional cached_all_dets, cached_faces.
    Writes all stage outputs back to ctx.
    """
    # 1. Object detection
    run_detection_step(ctx, yolo=yolo, det_cfg=det_cfg, obj_cache=obj_cache,
                       detection_plugins=detection_plugins)

    if ctx.get('do_cache'):
        ctx['cached_all_dets_out'] = ctx['all_dets']

    for p in ctx['persons']:
        cv2.rectangle(ctx['frame'], (p['x1'], p['y1']), (p['x2'], p['y2']), (255, 120, 30), 1)

    # 1.5. Depth estimation (between detection and gaze)
    if depth_cfg and depth_cfg.enabled and depth_backend and 'depth_map' not in ctx:
        from ms.DepthEstimation.depth_pipeline import run_depth_step
        run_depth_step(ctx, depth_cfg=depth_cfg, depth_backend=depth_backend)
    if depth_cfg and depth_cfg.enabled:
        ctx['depth_cfg'] = depth_cfg

    # 2. Gaze estimation + ray-bbox intersection
    run_gaze_step(ctx, face_det=face_det, gaze_eng=gaze_eng, gaze_cfg=gaze_cfg)

    # 3. Joint attention + gaze convergence
    ja_enabled = phenomena_cfg.joint_attention if phenomena_cfg is not None else True
    ctx['joint_objs'] = (joint_attention(
        ctx['persons_gaze'], ctx['hits'], quorum=gaze_cfg.ja_quorum)
        if ja_enabled else set())
    ctx['tip_convergences'] = (
        gaze_convergence(ctx['persons_gaze'], gaze_cfg.tip_radius)
        if gaze_cfg.gaze_tips else [])
    ctx['tip_radius'] = gaze_cfg.tip_radius
    ctx['detect_extend'] = gaze_cfg.detect_extend
    ctx['detect_extend_scope'] = gaze_cfg.detect_extend_scope

    # 3.5 Face anonymization (before overlay so annotations render on top)
    mode = ctx.get('anonymize')
    if mode:
        apply_face_anonymization(ctx['frame'], ctx.get('face_bboxes', []),
                                 mode, ctx.get('anonymize_padding', 0.3),
                                 face_track_ids=ctx.get('face_track_ids'),
                                 smoother=ctx.get('anon_smoother'))

    # 4. Annotate frame
    draw_overlay(ctx, gaze_cfg=gaze_cfg)


# ==============================================================================
# Run helpers
# ==============================================================================

def _ja_mode_string(phenomena_cfg, gaze_cfg):
    """Build a human-readable summary of active JA accuracy features for the HUD."""
    ja_flags = []
    if phenomena_cfg.joint_attention and phenomena_cfg.ja_window > 0:
        ja_flags.append(f"win={phenomena_cfg.ja_window}/{phenomena_cfg.ja_window_thresh:.0%}")
    if gaze_cfg.hit_conf_gate > 0:
        ja_flags.append(f"hit-gate={gaze_cfg.hit_conf_gate:.2f}")
    if phenomena_cfg.ja_quorum < 1.0:
        ja_flags.append(f"quorum={phenomena_cfg.ja_quorum:.0%}")
    if gaze_cfg.gaze_cone_angle > 0:
        ja_flags.append(f"cone={gaze_cfg.gaze_cone_angle:.1f}\u00b0")
    return ("JA+ [" + " ".join(ja_flags) + "]") if ja_flags else None


def _run_image(source, *, yolo, face_det, gaze_eng, gaze_cfg, det_cfg,
               output_cfg, phenomena_cfg, detection_plugins, ja_mode_str,
               depth_cfg=None, depth_backend=None,
               gazelle_provider=None, ray_cfg=None):
    """Single-frame pipeline: detect -> gaze -> JA -> overlay -> display/save."""
    frame = read_image_source(source)

    ctx = FrameContext(frame=frame, frame_no=0, pid_map=output_cfg.pid_map,
                       aux_frames={},
                       anonymize=output_cfg.anonymize,
                       anonymize_padding=output_cfg.anonymize_padding,
                       gazelle_provider=gazelle_provider,
                       ray_cfg=ray_cfg)
    process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                  gaze_cfg=gaze_cfg, det_cfg=det_cfg,
                  phenomena_cfg=phenomena_cfg,
                  detection_plugins=detection_plugins,
                  depth_cfg=depth_cfg, depth_backend=depth_backend)

    hit_events = ctx['hit_events']
    joint_objs = ctx['joint_objs']
    tip_convs = ctx['tip_convergences']
    dets = ctx['objects']

    ctx['fps'] = 0.0
    ctx['n_dets'] = len(hit_events)
    ctx['joint_pct'] = 100.0 if joint_objs else 0.0
    ctx['confirmed_objs'] = joint_objs
    ctx['extra_hud'] = ja_mode_str
    display = compose_dashboard(ctx)

    pid_map = output_cfg.pid_map
    for ev in hit_events:
        plbl = resolve_display_pid(ev['face_idx'], pid_map)
        print(f"{plbl} \u2192 {ev['object']} ({ev['object_conf']:.2f})")
    if joint_objs:
        print("Joint attention:", ", ".join(dets[oi]['class_name']
              for oi in sorted(joint_objs) if oi < len(dets)))
    for faces_set, centroid in tip_convs:
        tag = "+".join(resolve_display_pid(fi, pid_map) for fi in sorted(faces_set))
        print(f"Gaze convergence ({tag}) at ({int(centroid[0])}, {int(centroid[1])})")
    if not hit_events and not tip_convs:
        print("No gaze intersections detected.")
    if output_cfg.save:
        out = Path(source).stem + "_gaze.jpg"
        cv2.imwrite(out, display)
        print(f"Saved \u2192 {out}")
    cv2.imshow("MindSight", display)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================================================
# Run loop
# ==============================================================================

def run(source, yolo, face_det, gaze_eng,
        gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
        tracker_cfg: TrackerConfig, output_cfg: OutputConfig,
        plugin_instances=None,
        detection_plugins=None,
        phenomena_cfg=None,
        fast_mode=False, skip_phenomena=0, lite_overlay=False,
        no_dashboard=False, profile=False,
        depth_cfg=None, depth_backend=None,
        gazelle_provider=None, ray_cfg=None):
    """Execute the full MindSight pipeline on a single source (image, video, or webcam).

    For images: runs one frame through detection/gaze/JA, displays results, and exits.
    For video/webcam: enters the real-time loop with phenomena tracking, CSV logging,
    heatmap accumulation, and dashboard display until the user presses Q or the
    video ends.
    """
    from ms.Phenomena.phenomena_config import PhenomenaConfig
    if phenomena_cfg is None:
        phenomena_cfg = PhenomenaConfig()

    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS
    ja_mode_str = _ja_mode_string(phenomena_cfg, gaze_cfg)

    # -- Static image mode -----------------------------------------------------
    if is_image:
        return _run_image(source, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                          gaze_cfg=gaze_cfg, det_cfg=det_cfg, output_cfg=output_cfg,
                          phenomena_cfg=phenomena_cfg,
                          detection_plugins=detection_plugins,
                          ja_mode_str=ja_mode_str,
                          depth_cfg=depth_cfg, depth_backend=depth_backend,
                          gazelle_provider=gazelle_provider, ray_cfg=ray_cfg)

    # -- Video / webcam loop ---------------------------------------------------
    cap, _fps = open_video_source(source)
    aux_captures, _aux_ended = open_aux_streams(output_cfg, _fps)
    grace_frames = max(0, int(round(tracker_cfg.reid_grace_seconds * _fps)))

    smoother  = GazeSmootherReID(grace_frames=grace_frames,
                                   max_dist=tracker_cfg.reid_max_dist)
    locker    = (GazeLockTracker(dwell_frames=tracker_cfg.dwell_frames,
                                 lock_dist=tracker_cfg.lock_dist)
                 if tracker_cfg.gaze_lock else None)
    obj_cache = (ObjectPersistenceCache(max_age=tracker_cfg.obj_persistence)
                 if tracker_cfg.obj_persistence > 0 else None)
    snap_temporal = (SnapTemporalState(
                         release_frames=tracker_cfg.snap_release_frames,
                         engage_frames=tracker_cfg.snap_engage_frames)
                     if gaze_cfg.adaptive_ray != "off" else None)
    smooth_snap = (SmoothSnapTracker(alpha=gaze_cfg.smooth_snap_alpha)
                   if gaze_cfg.smooth_snap != "off" else None)

    # Phenomena trackers (built-in + plugins unified as PhenomenaPlugin list)
    builtin_trackers = init_phenomena_trackers(phenomena_cfg)
    all_trackers = builtin_trackers + (plugin_instances or [])

    # Set JA mode string on the JA tracker if present
    from ms.Phenomena.Default import JointAttentionTracker
    for t in all_trackers:
        if isinstance(t, JointAttentionTracker):
            t.ja_mode_str = ja_mode_str
            break

    writer, video_path = open_video_writer(output_cfg.save, source, cap,
                                              no_dashboard=no_dashboard)

    log_fh, log_csv = open_event_log(output_cfg)

    if ja_mode_str:
        print(f"JA accuracy mode: {ja_mode_str}")

    skip                             = max(1, tracker_cfg.skip_frames)
    frame_no = total_hits            = 0
    total_frames                     = 0
    frame_times                      = deque(maxlen=30)
    _prev_fps                        = 0.0
    look_counts: dict                = {}
    heatmap_gaze: dict               = {}

    # Persistent run-level state carried across frames via FrameContext.
    # Each frame gets a fresh FrameContext seeded with these base values.
    run_ctx_base = dict(
        source=source,
        smoother=smoother, locker=locker, snap_temporal=snap_temporal,
        smooth_snap_tracker=smooth_snap,
        all_trackers=all_trackers,
        look_counts=look_counts,
        heatmap_path=output_cfg.heatmap_path,
        heatmap_gaze=heatmap_gaze,
        charts_path=output_cfg.charts_path,
        summary_path=output_cfg.summary_path,
        pid_map=output_cfg.pid_map,
        anonymize=output_cfg.anonymize,
        anonymize_padding=output_cfg.anonymize_padding,
        anon_smoother=(AnonSmoother() if output_cfg.anonymize else None),
        face_det=face_det,
        video_name=output_cfg.video_name,
        conditions=output_cfg.conditions,
        gazelle_provider=gazelle_provider,
        ray_cfg=ray_cfg,
        video_fps=_fps,
        gazelle_blender=(GazeLLEBlender(ray_cfg)
                         if ray_cfg is not None and gazelle_provider is not None
                         else None),
        ray_object_snap=(ObjectSnap(ray_cfg) if ray_cfg is not None else None),
    )

    # Performance mode: resolve effective phenomena-skip interval
    _phen_skip = skip_phenomena if skip_phenomena > 0 else 0
    if fast_mode and _phen_skip == 0:
        # Under --fast, default to skipping phenomena on non-detection frames
        _phen_skip = -1                       # sentinel: tied to do_det

    # Pass lite_overlay flag into gaze_cfg so draw_overlay can read it
    if lite_overlay:
        gaze_cfg = gaze_cfg  # same object -- we set a dynamic attr below
        gaze_cfg._lite_overlay = True
    else:
        gaze_cfg._lite_overlay = False

    # Profiling accumulators
    if profile:
        _prof = {'detect': 0.0, 'gaze': 0.0, 'phenomena': 0.0,
                 'draw': 0.0, 'dashboard': 0.0, 'n': 0}

    print("MindSight running -> press Q to quit.")
    cache: dict = {}
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Skip-frame optimisation: only run expensive YOLO detection every
            # N frames; intermediate frames reuse cached detections.
            do_det = (frame_no % skip == 0)

            aux_frames = read_aux_frames(aux_captures, _aux_ended, frame_no)
            enrich_aux_with_face_detection(
                aux_frames, aux_captures, face_det,
                output_cfg.pid_map)

            # Build per-frame context from the run-level base
            ctx = FrameContext(frame=frame, frame_no=frame_no,
                               aux_frames=aux_frames, **run_ctx_base)
            # Depth skip-frame: only run depth every N detection frames
            _depth_skip = (depth_cfg.skip_frames
                           if depth_cfg and depth_cfg.enabled else 1)
            do_depth = do_det and (frame_no % (skip * _depth_skip) == 0)

            if not do_det:
                ctx['cached_all_dets'] = cache.get('all_dets')
                ctx['cached_faces'] = cache.get('faces')
            if not do_depth and 'depth_map' in cache:
                ctx['depth_map'] = cache['depth_map']
            ctx['do_cache'] = do_det

            # Inject previous frame's gaze data for detection plugins
            ctx['prev_persons_gaze'] = cache.get('prev_persons_gaze', [])
            ctx['prev_face_track_ids'] = cache.get('prev_face_track_ids', [])

            if profile: _t1 = time.perf_counter()

            process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                          gaze_cfg=gaze_cfg, det_cfg=det_cfg, obj_cache=obj_cache,
                          phenomena_cfg=phenomena_cfg,
                          detection_plugins=detection_plugins,
                          depth_cfg=depth_cfg, depth_backend=depth_backend)

            if profile:
                _t2 = time.perf_counter()
                _prof['detect'] += _t2 - _t1

            if do_det:
                cache['all_dets'] = ctx['all_dets']
                if 'faces' in ctx:
                    cache['faces'] = ctx['faces']
            if 'depth_map' in ctx and do_depth:
                cache['depth_map'] = ctx['depth_map']

            # Cache gaze data for next frame's detection plugins
            cache['prev_persons_gaze'] = ctx.get('persons_gaze', [])
            cache['prev_face_track_ids'] = ctx.get('face_track_ids', [])

            # Inject previous frame's FPS so LiveDashboardBridge can read it
            ctx['fps'] = _prev_fps

            # Phenomena tracker updates (built-in + plugins, unified loop)
            # Determine whether to run phenomena this frame
            if _phen_skip == -1:
                do_phenomena = do_det           # --fast: tied to detection frames
            elif _phen_skip > 0:
                do_phenomena = (frame_no % _phen_skip == 0)
            else:
                do_phenomena = True

            if profile: _t3 = time.perf_counter()

            if do_phenomena:
                update_phenomena_step(ctx)
            else:
                # Seed defaults so downstream code doesn't KeyError
                if 'confirmed_objs' not in ctx:
                    ctx['confirmed_objs'] = ctx.get('joint_objs', set())
                if 'extra_hud' not in ctx:
                    ctx['extra_hud'] = None
                if 'joint_pct' not in ctx:
                    ctx['joint_pct'] = 0.0

            if profile:
                _t4 = time.perf_counter()
                _prof['phenomena'] += _t4 - _t3

            confirmed_objs = ctx['confirmed_objs']
            hit_events     = ctx['hit_events']
            joint_objs     = ctx['joint_objs']
            persons_gaze   = ctx['persons_gaze']
            face_track_ids = ctx.get('face_track_ids', [])

            # Frame statistics -- JA counters are now managed by JointAttentionTracker
            total_frames += 1
            joint_pct    = ctx.get('joint_pct', 0.0)
            is_joint     = bool(joint_objs)
            is_confirmed = bool(confirmed_objs)

            # Store derived values for data collection
            ctx['is_joint'] = is_joint
            ctx['is_confirmed'] = is_confirmed

            if profile: _t5 = time.perf_counter()

            # Tracker frame overlays (drawn after built-in gaze overlay)
            for t in all_trackers:
                t.draw_frame(ctx['frame'])

            if profile:
                _t6 = time.perf_counter()
                _prof['draw'] += _t6 - _t5

            # Console hit output
            pid_map = output_cfg.pid_map
            for ev in hit_events:
                total_hits += 1
                tag = ""
                if is_confirmed: tag = "  [JOINT+CONFIRMED]"
                elif is_joint:   tag = "  [JOINT]"
                plbl = resolve_display_pid(ev['face_idx'], pid_map)
                print(f"[{frame_no:05d}] {plbl} \u2192 {ev['object']}"
                      f" ({ev['object_conf']:.2f}){tag}")

            # Data collection (CSV logging, look counts, heatmap accumulation)
            collect_frame_data(ctx, log_csv=log_csv, frame_no=frame_no,
                               hit_events=hit_events, face_track_ids=face_track_ids,
                               persons_gaze=persons_gaze)

            # Rolling FPS estimate over the last 30 frames
            frame_times.append(time.perf_counter() - t0)
            cur_fps = 1.0 / (sum(frame_times) / len(frame_times))
            _prev_fps = cur_fps

            ctx['fps'] = cur_fps
            ctx['n_dets'] = len(hit_events)
            ctx['joint_pct'] = joint_pct

            if profile: _t7 = time.perf_counter()

            if no_dashboard:
                display = ctx['frame']
            else:
                display = compose_dashboard(ctx)

            if profile:
                _t8 = time.perf_counter()
                _prof['dashboard'] += _t8 - _t7
                _prof['n'] += 1
                if _prof['n'] % 100 == 0:
                    n = _prof['n']
                    print(f"[PROFILE] frame {n} avg: "
                          f"detect={_prof['detect']/n*1000:.1f}ms "
                          f"phenomena={_prof['phenomena']/n*1000:.1f}ms "
                          f"draw={_prof['draw']/n*1000:.1f}ms "
                          f"dashboard={_prof['dashboard']/n*1000:.1f}ms "
                          f"total={(time.perf_counter()-t0)*1000:.1f}ms")

            if writer:
                writer.write(display)
            cv2.imshow("MindSight", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_no += 1
    finally:
        cap.release()
        for ac in aux_captures.values():
            ac.release()
        if writer:
            writer.release()
            finalize_video(video_path)
        if log_fh:  log_fh.close()
        cv2.destroyAllWindows()

    # Post-run phenomena summaries
    post_run_summary(all_trackers, total_frames, pid_map=output_cfg.pid_map)

    # Post-run data output (stats print + CSV summary + heatmaps + charts)
    run_ctx = FrameContext(frame_no=frame_no, **run_ctx_base)
    run_ctx['total_frames'] = total_frames
    run_ctx['total_hits'] = total_hits
    run_ctx['fps'] = run_ctx_base.get('fps', 30.0)
    finalize_run(run_ctx)
