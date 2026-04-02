"""
MindSight.py -> Main program; orchestrates the full gaze-tracking pipeline.

Architecture
------------
The four logical stages of the run loop are each implemented in their own
module so that user plugins can swap out individual steps:

  ObjectDetection/detection_pipeline.py  -> YOLO detection step
  GazeTracking/gaze_pipeline.py          -> face detection + gaze + intersection
  Phenomena/phenomena_pipeline.py        -> phenomena tracker init / update / summary
  DataCollection/data_pipeline.py        -> CSV logging, look counts, heatmaps

This file is the thin orchestrator: it wires the stages together and owns
the CLI, model loading, and display loop.

All pipeline stages communicate through a shared ``FrameContext`` object.
Each stage reads the keys it needs and writes its results, so adding new
data never requires changing function signatures.

CLI flags
---------
Each submodule registers its own arguments via ``add_arguments(parser)``:

  ObjectDetection/object_detection.py  -> --model, --conf, --classes, etc.
  GazeTracking/gaze_processing.py      -> --gaze-model, --ray-length, --gaze-cone, etc.
  Phenomena/phenomena_tracking.py       -> --mutual-gaze, --social-ref, --ja-window, etc.

Run ``python MindSight.py --help`` for the full list.

Base pipeline: YOLO objects -> RetinaFace faces -> gaze estimation ->
               eye-landmark origin (auto-fallback to face bbox) ->
               ray\u2013bbox (or cone) intersection -> joint attention + phenomena
"""

import argparse, csv, time
from pathlib import Path

import cv2


# ── Pipeline stage imports ────────────────────────────────────────────────────
from ObjectDetection.detection_pipeline import run_detection_step
from GazeTracking.gaze_pipeline import run_gaze_step
from Phenomena.phenomena_pipeline import (
    init_phenomena_trackers, update_phenomena_step, post_run_summary,
    joint_attention, gaze_convergence,
)
from DataCollection.data_pipeline import collect_frame_data, finalize_run

# ── Sub-module imports ────────────────────────────────────────────────────────
from GazeTracking.gaze_processing import (
    GazeSmootherReID, GazeLockTracker, SnapHysteresisTracker,
)
from GazeTracking.gaze_factory import create_gaze_engine
from Plugins import (
    gaze_registry             as _gaze_registry,
    object_detection_registry as _od_registry,
    phenomena_registry        as _phenomena_registry,
)
from Phenomena.phenomena_tracking import add_arguments as _add_phenomena_arguments
from ObjectDetection.object_detection import ObjectPersistenceCache
from ObjectDetection.model_factory import create_yolo_detector, create_face_detector
from DataCollection.dashboard_output import draw_overlay, compose_dashboard, open_video_writer

# ── Constants & config ────────────────────────────────────────────────────────
from constants import IMAGE_EXTS
from pipeline_config import GazeConfig, DetectionConfig, TrackerConfig, OutputConfig, FrameContext


# ══════════════════════════════════════════════════════════════════════════════
# Core per-frame processing  (thin orchestrator -> logic lives in pipeline files)
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(ctx, *, yolo, face_det, gaze_eng,
                  gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
                  obj_cache=None, phenomena_cfg=None):
    """
    Process one frame through detection, gaze, JA, and overlay stages.

    Reads from ctx: frame (required), plus optional cached_all_dets, cached_faces.
    Writes all stage outputs back to ctx.
    """
    # 1. Object detection
    run_detection_step(ctx, yolo=yolo, det_cfg=det_cfg, obj_cache=obj_cache)

    if ctx.get('do_cache'):
        ctx['cached_all_dets_out'] = ctx['all_dets']

    for p in ctx['persons']:
        cv2.rectangle(ctx['frame'], (p['x1'], p['y1']), (p['x2'], p['y2']), (255, 120, 30), 1)

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

    # 4. Annotate frame
    draw_overlay(ctx, gaze_cfg=gaze_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Run loop
# ══════════════════════════════════════════════════════════════════════════════

def run(source, yolo, face_det, gaze_eng,
        gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
        tracker_cfg: TrackerConfig, output_cfg: OutputConfig,
        plugin_instances=None,
        phenomena_cfg=None):
    """Execute the full MindSight pipeline on a single source (image, video, or webcam).

    For images: runs one frame through detection/gaze/JA, displays results, and exits.
    For video/webcam: enters the real-time loop with phenomena tracking, CSV logging,
    heatmap accumulation, and dashboard display until the user presses Q or the
    video ends.
    """
    from Phenomena.phenomena_config import PhenomenaConfig
    if phenomena_cfg is None:
        phenomena_cfg = PhenomenaConfig()

    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS

    # Build a human-readable summary of active JA accuracy features for the HUD
    ja_flags = []
    if phenomena_cfg.joint_attention and phenomena_cfg.ja_window > 0:
        ja_flags.append(f"win={phenomena_cfg.ja_window}/{phenomena_cfg.ja_window_thresh:.0%}")
    if phenomena_cfg.ja_conf_gate > 0:
        ja_flags.append(f"gate={phenomena_cfg.ja_conf_gate:.2f}")
    if phenomena_cfg.ja_quorum < 1.0:
        ja_flags.append(f"quorum={phenomena_cfg.ja_quorum:.0%}")
    if gaze_cfg.gaze_cone_angle > 0:
        ja_flags.append(f"cone={gaze_cfg.gaze_cone_angle:.1f}\u00b0")
    ja_mode_str = ("JA+ [" + " ".join(ja_flags) + "]") if ja_flags else None

    # ── Static image mode ─────────────────────────────────────────────────────
    # Single-frame pipeline: detect → gaze → JA → overlay → display/save.
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {source}")

        ctx = FrameContext(frame=frame, frame_no=0)
        process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                      gaze_cfg=gaze_cfg, det_cfg=det_cfg,
                      phenomena_cfg=phenomena_cfg)

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

        for ev in hit_events:
            print(f"Face {ev['face_idx']} \u2192 {ev['object']} ({ev['object_conf']:.2f})")
        if joint_objs:
            print("Joint attention:", ", ".join(dets[oi]['class_name']
                  for oi in sorted(joint_objs) if oi < len(dets)))
        for faces_set, centroid in tip_convs:
            tag = "+".join(f"P{fi}" for fi in sorted(faces_set))
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
        return

    # ── Video / webcam loop ──────────────────────────────────────────────────
    # Initialize capture, per-run trackers, and output sinks before entering
    # the frame loop.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Convert the user-facing grace period (seconds) to frames using the
    # source FPS so Re-ID behaves consistently across different frame rates.
    _fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grace_frames = max(0, int(round(tracker_cfg.reid_grace_seconds * _fps)))

    smoother  = GazeSmootherReID(grace_frames=grace_frames)
    locker    = (GazeLockTracker(dwell_frames=tracker_cfg.dwell_frames,
                                 lock_dist=tracker_cfg.lock_dist)
                 if tracker_cfg.gaze_lock else None)
    obj_cache = (ObjectPersistenceCache(max_age=tracker_cfg.obj_persistence)
                 if tracker_cfg.obj_persistence > 0 else None)
    snap_hyst = (SnapHysteresisTracker(switch_frames=tracker_cfg.snap_switch_frames)
                 if gaze_cfg.adaptive_ray else None)

    # Phenomena trackers (built-in + plugins unified as PhenomenaPlugin list)
    ja_tracker, builtin_trackers = init_phenomena_trackers(phenomena_cfg)
    all_trackers = builtin_trackers + (plugin_instances or [])

    log_fh = log_csv = None

    writer = open_video_writer(output_cfg.save, source, cap)

    if output_cfg.log_path:
        log_fh  = open(output_cfg.log_path, "w", newline="")
        log_csv = csv.writer(log_fh)
        log_csv.writerow(["frame","face_idx","object","object_conf",
                          "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                          "joint_attention","joint_attention_confirmed"])
        print(f"Logging \u2192 {output_cfg.log_path}")

    if ja_mode_str:
        print(f"JA accuracy mode: {ja_mode_str}")

    skip                             = max(1, tracker_cfg.skip_frames)
    frame_no = total_hits            = 0
    joint_frames = confirmed_frames  = total_frames = 0
    frame_times                      = []
    look_counts: dict                = {}
    heatmap_gaze: dict               = {}

    # Persistent run-level state carried across frames via FrameContext.
    # Each frame gets a fresh FrameContext seeded with these base values.
    run_ctx_base = dict(
        source=source,
        smoother=smoother, locker=locker, snap_hysteresis=snap_hyst,
        ja_tracker=ja_tracker, ja_mode_str=ja_mode_str,
        all_trackers=all_trackers,
        look_counts=look_counts,
        heatmap_path=output_cfg.heatmap_path,
        heatmap_gaze=heatmap_gaze,
        summary_path=output_cfg.summary_path,
    )

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

            # Build per-frame context from the run-level base
            ctx = FrameContext(frame=frame, frame_no=frame_no, **run_ctx_base)
            if not do_det:
                ctx['cached_all_dets'] = cache.get('all_dets')
                ctx['cached_faces'] = cache.get('faces')
            ctx['do_cache'] = do_det

            process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                          gaze_cfg=gaze_cfg, det_cfg=det_cfg, obj_cache=obj_cache,
                          phenomena_cfg=phenomena_cfg)

            if do_det:
                cache['all_dets'] = ctx['all_dets']
                if 'faces' in ctx:
                    cache['faces'] = ctx['faces']

            # Phenomena tracker updates (built-in + plugins, unified loop)
            update_phenomena_step(ctx)

            confirmed_objs = ctx['confirmed_objs']
            hit_events     = ctx['hit_events']
            joint_objs     = ctx['joint_objs']
            tip_convs      = ctx['tip_convergences']
            persons_gaze   = ctx['persons_gaze']
            face_track_ids = ctx.get('face_track_ids', [])
            dets           = ctx['objects']

            # Frame statistics
            total_frames += 1
            if joint_objs or tip_convs:
                joint_frames += 1
            if confirmed_objs or tip_convs:
                confirmed_frames += 1

            joint_pct    = confirmed_frames / total_frames * 100
            is_joint     = bool(joint_objs)
            is_confirmed = bool(confirmed_objs)

            # Store derived values for data collection
            ctx['is_joint'] = is_joint
            ctx['is_confirmed'] = is_confirmed

            # Tracker frame overlays (drawn after built-in gaze overlay)
            for t in all_trackers:
                t.draw_frame(ctx['frame'])

            # Console hit output
            for ev in hit_events:
                total_hits += 1
                tag = ""
                if is_confirmed: tag = "  [JOINT+CONFIRMED]"
                elif is_joint:   tag = "  [JOINT]"
                print(f"[{frame_no:05d}] P{ev['face_idx']} \u2192 {ev['object']}"
                      f" ({ev['object_conf']:.2f}){tag}")

            # Data collection (CSV logging, look counts, heatmap accumulation)
            collect_frame_data(ctx, log_csv=log_csv, frame_no=frame_no,
                               hit_events=hit_events, face_track_ids=face_track_ids,
                               persons_gaze=persons_gaze)

            # Rolling FPS estimate over the last 30 frames
            frame_times.append(time.perf_counter() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            cur_fps = 1.0 / (sum(frame_times) / len(frame_times))

            ctx['fps'] = cur_fps
            ctx['n_dets'] = len(hit_events)
            ctx['joint_pct'] = joint_pct
            display = compose_dashboard(ctx)

            if writer:
                writer.write(display)
            cv2.imshow("MindSight", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_no += 1
    finally:
        cap.release()
        if writer:  writer.release()
        if log_fh:  log_fh.close()
        cv2.destroyAllWindows()

    # Post-run phenomena summaries
    post_run_summary(all_trackers, total_frames)

    # Post-run data output (stats print + CSV summary + heatmaps)
    run_ctx = FrameContext(frame_no=frame_no, **run_ctx_base)
    run_ctx['total_frames'] = total_frames
    run_ctx['joint_frames'] = joint_frames
    run_ctx['confirmed_frames'] = confirmed_frames
    run_ctx['total_hits'] = total_hits
    finalize_run(run_ctx)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _args():
    from GazeTracking.gaze_processing import add_arguments as _add_gaze_args
    from ObjectDetection.object_detection import add_arguments as _add_det_args

    p = argparse.ArgumentParser("MindSight -- Eye-Gaze Intersection Tracker")

    # ── Orchestration-level flags ────────────────────────────────────────────
    p.add_argument("--source", default="0",
                   help="Video input source, defaults to webcam")
    p.add_argument("--save", nargs="?", const=True, default=None, metavar="PATH",
                   help="Save annotated video. Omit a value to use "
                        "Outputs/Video/[stem]_Video_Output.mp4, or supply a custom path.")
    p.add_argument("--log", default=None)
    p.add_argument("--summary", nargs="?", const=True, default=None, metavar="PATH",
                   help="Save post-run summary CSV. Omit a value to use "
                        "Outputs/CSV Files/[stem]_Summary_Output.csv, or supply a custom path.")
    p.add_argument("--heatmap", nargs="?", const=True, default=None, metavar="PATH",
                   help="Save per-participant scene gaze heatmaps. Omit a value to use "
                        "Outputs/heatmaps/[stem]_Heatmap_Output (one PNG per participant), "
                        "or supply a custom directory/prefix path.")
    p.add_argument("--pipeline", default=None, metavar="YAML",
                   help="Load pipeline configuration from a YAML file. "
                        "CLI flags override YAML values.")
    p.add_argument("--project", default=None, metavar="DIR",
                   help="Run in project mode: process all videos in DIR/Inputs/Videos/ "
                        "using DIR/Pipeline/pipeline.yaml as config.")

    # ── Delegate to submodules ────────────────────────────────────────────────
    _add_det_args(p)
    _add_gaze_args(p)
    _add_phenomena_arguments(p)

    # ---- Plugin-contributed arguments ──────────────────────────────────────
    # Plugins receive the root parser so they can create their own argument
    # groups internally (argparse forbids nested groups).
    for _pname in _gaze_registry.names():
        _gaze_registry.get(_pname).add_arguments(p)
    for _pname in _od_registry.names():
        _od_registry.get(_pname).add_arguments(p)
    for _pname in _phenomena_registry.names():
        _phenomena_registry.get(_pname).add_arguments(p)

    return p.parse_args()


def _build_from_args(args):
    """Build all models and config objects from a parsed argparse namespace.

    Returns (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
             output_cfg, active_plugins, phenomena_cfg).
    """
    from Phenomena.phenomena_config import PhenomenaConfig

    yolo, class_ids, blacklist = create_yolo_detector(
        model_path=args.model,
        classes=args.classes or None,
        blacklist_names=args.blacklist,
        vp_file=args.vp_file,
        vp_model=args.vp_model,
    )
    face_det = create_face_detector()

    # Plugin discovery summary
    for _reg_label, _reg in (("Gaze",             _gaze_registry),
                              ("Object detection", _od_registry),
                              ("Phenomena",        _phenomena_registry)):
        if _reg.names():
            print(f"{_reg_label} plugins discovered: {', '.join(_reg.names())}")

    gaze_eng = create_gaze_engine(plugin_args=args)

    # Phenomena plugins: instantiate whichever flags were activated
    active_plugins: list = []
    for _pname in _phenomena_registry.names():
        _pcls = _phenomena_registry.get(_pname)
        try:
            _inst = _pcls.from_args(args)
        except Exception as _exc:
            raise RuntimeError(
                f"Phenomena plugin '{_pname}' failed to initialize: {_exc}"
            ) from _exc
        if _inst is not None:
            active_plugins.append(_inst)
            print(f"Phenomena plugin active: {_pname}")

    phenomena_cfg = PhenomenaConfig.from_namespace(args)
    gaze_cfg      = GazeConfig.from_namespace(args)
    det_cfg       = DetectionConfig.from_namespace(args, class_ids, blacklist)
    tracker_cfg   = TrackerConfig.from_namespace(args)
    output_cfg    = OutputConfig.from_namespace(args)

    return (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
            output_cfg, active_plugins or None, phenomena_cfg)


def main():
    args = _args()

    # Apply pipeline YAML if specified (CLI flags take precedence)
    if args.pipeline:
        from pipeline_loader import load_pipeline
        load_pipeline(args.pipeline, args)
        print(f"Loaded pipeline config: {args.pipeline}")

    # Project mode: batch-process all videos in a project directory
    if args.project:
        from project_runner import run_project
        run_project(args.project, run, _build_from_args, args)
        return

    # Single-source mode
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
     output_cfg, active_plugins, phenomena_cfg) = _build_from_args(args)

    run(source, yolo, face_det, gaze_eng,
        gaze_cfg, det_cfg, tracker_cfg, output_cfg,
        plugin_instances=active_plugins,
        phenomena_cfg=phenomena_cfg)


if __name__ == "__main__":
    main()
