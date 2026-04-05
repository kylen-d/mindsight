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
  GazeTracking/gaze_processing.py      -> --mgaze-model, --ray-length, --gaze-cone, etc.
  Phenomena/phenomena_tracking.py       -> --mutual-gaze, --social-ref, --ja-window, etc.

Run ``python MindSight.py --help`` for the full list.

Base pipeline: YOLO objects -> RetinaFace faces -> gaze estimation ->
               eye-landmark origin (auto-fallback to face bbox) ->
               ray-bbox (or cone) intersection -> joint attention + phenomena
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2

# ── Constants & config ────────────────────────────────────────────────────────
from constants import IMAGE_EXTS
from DataCollection.dashboard_output import (
    AnonSmoother,
    apply_face_anonymization,
    compose_dashboard,
    draw_overlay,
    finalize_video,
    open_video_writer,
)
from DataCollection.data_pipeline import collect_frame_data, finalize_run
from GazeTracking.gaze_factory import create_gaze_engine
from GazeTracking.gaze_pipeline import run_gaze_step

# ── Sub-module imports ────────────────────────────────────────────────────────
from GazeTracking.gaze_processing import (
    GazeLockTracker,
    GazeSmootherReID,
    SnapHysteresisTracker,
)

# ── Pipeline stage imports ────────────────────────────────────────────────────
from ObjectDetection.detection_pipeline import run_detection_step
from ObjectDetection.model_factory import create_face_detector, create_yolo_detector
from ObjectDetection.object_detection import ObjectPersistenceCache
from Phenomena.phenomena_pipeline import (
    gaze_convergence,
    init_phenomena_trackers,
    joint_attention,
    post_run_summary,
    update_phenomena_step,
)
from Phenomena.phenomena_tracking import add_arguments as _add_phenomena_arguments
from pipeline_config import (
    DetectionConfig,
    FrameContext,
    GazeConfig,
    OutputConfig,
    TrackerConfig,
    resolve_display_pid,
)
from Plugins import (
    gaze_registry as _gaze_registry,
)
from Plugins import (
    object_detection_registry as _od_registry,
)
from Plugins import (
    phenomena_registry as _phenomena_registry,
)

# ══════════════════════════════════════════════════════════════════════════════
# Core per-frame processing  (thin orchestrator -> logic lives in pipeline files)
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(ctx, *, yolo, face_det, gaze_eng,
                  gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
                  obj_cache=None, phenomena_cfg=None,
                  detection_plugins=None):
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


# ══════════════════════════════════════════════════════════════════════════════
# Run helpers
# ══════════════════════════════════════════════════════════════════════════════

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
               output_cfg, phenomena_cfg, detection_plugins, ja_mode_str):
    """Single-frame pipeline: detect -> gaze -> JA -> overlay -> display/save."""
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {source}")

    ctx = FrameContext(frame=frame, frame_no=0, pid_map=output_cfg.pid_map,
                       aux_frames={},
                       anonymize=output_cfg.anonymize,
                       anonymize_padding=output_cfg.anonymize_padding)
    process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                  gaze_cfg=gaze_cfg, det_cfg=det_cfg,
                  phenomena_cfg=phenomena_cfg,
                  detection_plugins=detection_plugins)

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


def _open_aux_streams(output_cfg, main_fps):
    """Open auxiliary video captures and return (captures_dict, ended_set)."""
    aux_captures: dict[tuple[str, str], cv2.VideoCapture] = {}
    _aux_ended: set[tuple[str, str]] = set()
    if output_cfg.aux_streams:
        for aux in output_cfg.aux_streams:
            ac = cv2.VideoCapture(aux.source)
            if not ac.isOpened():
                print(f"Warning: cannot open aux stream "
                      f"{aux.pid}:{aux.stream_type} ({aux.source}) -- skipping")
                continue
            aux_fps = ac.get(cv2.CAP_PROP_FPS) or 30.0
            if abs(aux_fps - main_fps) > 1.0:
                print(f"Warning: aux stream {aux.pid}:{aux.stream_type} "
                      f"FPS ({aux_fps:.1f}) differs from main ({main_fps:.1f}) "
                      f"-- frames may drift")
            aux_captures[(aux.pid, aux.stream_type)] = ac
        if aux_captures:
            print(f"Opened {len(aux_captures)} auxiliary stream(s)")
    return aux_captures, _aux_ended


def _read_aux_frames(aux_captures, _aux_ended, frame_no):
    """Read one frame from each auxiliary stream, returning a dict of frames."""
    aux_frames: dict[tuple[str, str], object] = {}
    for key, ac in aux_captures.items():
        ret_a, frame_a = ac.read()
        if ret_a:
            aux_frames[key] = frame_a
        else:
            aux_frames[key] = None
            if key not in _aux_ended:
                _aux_ended.add(key)
                print(f"Warning: aux stream {key[0]}:{key[1]} "
                      f"ended at frame {frame_no}")
    return aux_frames


# ══════════════════════════════════════════════════════════════════════════════
# Run loop
# ══════════════════════════════════════════════════════════════════════════════

def run(source, yolo, face_det, gaze_eng,
        gaze_cfg: GazeConfig, det_cfg: DetectionConfig,
        tracker_cfg: TrackerConfig, output_cfg: OutputConfig,
        plugin_instances=None,
        detection_plugins=None,
        phenomena_cfg=None,
        fast_mode=False, skip_phenomena=0, lite_overlay=False,
        no_dashboard=False, profile=False):
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
    ja_mode_str = _ja_mode_string(phenomena_cfg, gaze_cfg)

    # ── Static image mode ─────────────────────────────────────────────────────
    if is_image:
        return _run_image(source, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                          gaze_cfg=gaze_cfg, det_cfg=det_cfg, output_cfg=output_cfg,
                          phenomena_cfg=phenomena_cfg,
                          detection_plugins=detection_plugins,
                          ja_mode_str=ja_mode_str)

    # ── Video / webcam loop ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    _fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    aux_captures, _aux_ended = _open_aux_streams(output_cfg, _fps)
    grace_frames = max(0, int(round(tracker_cfg.reid_grace_seconds * _fps)))

    smoother  = GazeSmootherReID(grace_frames=grace_frames,
                                   max_dist=tracker_cfg.reid_max_dist)
    locker    = (GazeLockTracker(dwell_frames=tracker_cfg.dwell_frames,
                                 lock_dist=tracker_cfg.lock_dist)
                 if tracker_cfg.gaze_lock else None)
    obj_cache = (ObjectPersistenceCache(max_age=tracker_cfg.obj_persistence)
                 if tracker_cfg.obj_persistence > 0 else None)
    snap_hyst = (SnapHysteresisTracker(switch_frames=tracker_cfg.snap_switch_frames)
                 if gaze_cfg.adaptive_ray != "off" else None)

    # Phenomena trackers (built-in + plugins unified as PhenomenaPlugin list)
    builtin_trackers = init_phenomena_trackers(phenomena_cfg)
    all_trackers = builtin_trackers + (plugin_instances or [])

    # Set JA mode string on the JA tracker if present
    from Phenomena.Default import JointAttentionTracker
    for t in all_trackers:
        if isinstance(t, JointAttentionTracker):
            t.ja_mode_str = ja_mode_str
            break

    log_fh = log_csv = None

    writer, video_path = open_video_writer(output_cfg.save, source, cap,
                                              no_dashboard=no_dashboard)

    if output_cfg.log_path:
        log_fh  = open(output_cfg.log_path, "w", newline="")
        log_csv = csv.writer(log_fh)
        header = ["frame","face_idx","object","object_conf",
                  "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                  "joint_attention","joint_attention_confirmed",
                  "participant_label"]
        if output_cfg.video_name is not None:
            header = ["video_name", "conditions"] + header
        log_csv.writerow(header)
        print(f"Logging \u2192 {output_cfg.log_path}")

    if ja_mode_str:
        print(f"JA accuracy mode: {ja_mode_str}")

    skip                             = max(1, tracker_cfg.skip_frames)
    frame_no = total_hits            = 0
    total_frames                     = 0
    frame_times                      = deque(maxlen=30)
    look_counts: dict                = {}
    heatmap_gaze: dict               = {}

    # Persistent run-level state carried across frames via FrameContext.
    # Each frame gets a fresh FrameContext seeded with these base values.
    run_ctx_base = dict(
        source=source,
        smoother=smoother, locker=locker, snap_hysteresis=snap_hyst,
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
    )

    # Performance mode: resolve effective phenomena-skip interval
    _phen_skip = skip_phenomena if skip_phenomena > 0 else 0
    if fast_mode and _phen_skip == 0:
        # Under --fast, default to skipping phenomena on non-detection frames
        _phen_skip = -1                       # sentinel: tied to do_det

    # Pass lite_overlay flag into gaze_cfg so draw_overlay can read it
    if lite_overlay:
        gaze_cfg = gaze_cfg  # same object — we set a dynamic attr below
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

            aux_frames = _read_aux_frames(aux_captures, _aux_ended, frame_no)

            # Build per-frame context from the run-level base
            ctx = FrameContext(frame=frame, frame_no=frame_no,
                               aux_frames=aux_frames, **run_ctx_base)
            if not do_det:
                ctx['cached_all_dets'] = cache.get('all_dets')
                ctx['cached_faces'] = cache.get('faces')
            ctx['do_cache'] = do_det

            # Inject previous frame's gaze data for detection plugins
            ctx['prev_persons_gaze'] = cache.get('prev_persons_gaze', [])
            ctx['prev_face_track_ids'] = cache.get('prev_face_track_ids', [])

            if profile: _t1 = time.perf_counter()

            process_frame(ctx, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
                          gaze_cfg=gaze_cfg, det_cfg=det_cfg, obj_cache=obj_cache,
                          phenomena_cfg=phenomena_cfg,
                          detection_plugins=detection_plugins)

            if profile:
                _t2 = time.perf_counter()
                _prof['detect'] += _t2 - _t1

            if do_det:
                cache['all_dets'] = ctx['all_dets']
                if 'faces' in ctx:
                    cache['faces'] = ctx['faces']

            # Cache gaze data for next frame's detection plugins
            cache['prev_persons_gaze'] = ctx.get('persons_gaze', [])
            cache['prev_face_track_ids'] = ctx.get('face_track_ids', [])

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
            tip_convs      = ctx['tip_convergences']
            persons_gaze   = ctx['persons_gaze']
            face_track_ids = ctx.get('face_track_ids', [])
            dets           = ctx['objects']

            # Frame statistics — JA counters are now managed by JointAttentionTracker
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
    p.add_argument("--charts", nargs="?", const=True, default=None, metavar="PATH",
                   help="Generate post-run time-series charts for each phenomena tracker. "
                        "Omit a value to use Outputs/Charts/[stem]_Charts.png, "
                        "or supply a custom path.")
    p.add_argument("--pipeline", default=None, metavar="YAML",
                   help="Load pipeline configuration from a YAML file. "
                        "CLI flags override YAML values.")
    p.add_argument("--project", default=None, metavar="DIR",
                   help="Run in project mode: process all videos in DIR/Inputs/Videos/ "
                        "using DIR/Pipeline/pipeline.yaml as config.")
    p.add_argument("--participant-ids", default=None, metavar="IDS",
                   help="Comma-separated participant labels for single-video mode. "
                        "Positional: first label maps to track 0, second to track 1, "
                        "etc. E.g. --participant-ids S70,S71,S72")
    p.add_argument("--participant-csv", default=None, metavar="CSV",
                   help="Path to a participant_ids.csv mapping video filenames "
                        "to custom participant labels (see docs for format).")
    p.add_argument("--aux-stream", action="append", default=None,
                   dest="aux_streams_raw", metavar="PID:TYPE:SOURCE",
                   help="Auxiliary video stream mapped to a participant. "
                        "Format: PID:TYPE:SOURCE where PID is the participant "
                        "label, TYPE is the stream purpose (e.g. eye_camera, "
                        "first_person_view), and SOURCE is the file path. "
                        "Repeatable for multiple streams.")

    p.add_argument("--device", default="auto",
                   help="Compute device for all backends: auto, cpu, cuda, "
                        "or mps.  'auto' selects CUDA > MPS > CPU  (default: auto).")
    p.add_argument("--anonymize", choices=["blur", "black"], default=None,
                   help="Anonymize faces in the output video: 'blur' applies "
                        "heavy Gaussian blur, 'black' fills with a solid rectangle.")
    p.add_argument("--anonymize-padding", type=float, default=0.3, metavar="FRAC",
                   help="Fraction of face bbox size added as padding for "
                        "anonymization (default: 0.3).")

    # ── Performance flags ────────────────────────────────────────────────────
    perf = p.add_argument_group("Performance")
    perf.add_argument(
        "--fast", action="store_true", default=False,
        help="Enable bundled performance optimizations: skip phenomena on "
             "non-detection frames, throttle dashboard bridge, reduce GUI "
             "poll rate.")
    perf.add_argument(
        "--skip-phenomena", type=int, default=0, metavar="N",
        help="Run phenomena trackers only every N frames (0 = every frame). "
             "Independent of --skip-frames.  (default: 0)")
    perf.add_argument(
        "--lite-overlay", action="store_true", default=False,
        help="Minimal overlay: disable cone rendering, convergence markers, "
             "dwell arcs, and debug text.  Keeps gaze arrows, boxes, badges.")
    perf.add_argument(
        "--no-dashboard", action="store_true", default=False,
        help="Skip dashboard composition for maximum throughput. "
             "Displays the raw annotated frame only.")
    perf.add_argument(
        "--profile", action="store_true", default=False,
        help="Print per-stage timing breakdown every 100 frames.")

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
             output_cfg, active_plugins, phenomena_cfg, detection_plugins).
    """
    from Phenomena.phenomena_config import PhenomenaConfig

    yolo, class_ids, blacklist = create_yolo_detector(
        model_path=args.model,
        classes=args.classes or None,
        blacklist_names=args.blacklist,
        vp_file=args.vp_file,
        vp_model=args.vp_model,
        device=getattr(args, "device", "auto"),
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

    # Object detection plugins: instantiate whichever flags were activated
    detection_plugins: list = []
    for _pname in _od_registry.names():
        _dcls = _od_registry.get(_pname)
        try:
            _inst = _dcls.from_args(args)
        except Exception as _exc:
            raise RuntimeError(
                f"Object detection plugin '{_pname}' failed to initialize: {_exc}"
            ) from _exc
        if _inst is not None:
            detection_plugins.append(_inst)
            print(f"Object detection plugin active: {_pname}")

    # Build pid_map from inline IDs or CSV (single-video; project mode handles its own)
    pid_map = getattr(args, 'pid_map', None)
    if pid_map is None:
        from participant_ids import load_participant_csv, parse_inline_ids
        inline = getattr(args, 'participant_ids', None)
        csv_path = getattr(args, 'participant_csv', None)
        if inline:
            pid_map = parse_inline_ids(inline)
        elif csv_path:
            all_maps = load_participant_csv(csv_path)
            # For single-video mode, try to find the source's filename in the CSV
            src = getattr(args, 'source', None)
            if src and not isinstance(src, int):
                fname = Path(src).name
                pid_map = all_maps.get(fname)
                if pid_map is None and all_maps:
                    print(f"Warning: '{fname}' not found in participant CSV; "
                          f"available: {list(all_maps.keys())}")
    args.pid_map = pid_map

    # Parse --aux-stream PID:TYPE:SOURCE entries into AuxStreamConfig list
    from pipeline_config import AuxStreamConfig
    raw_aux = getattr(args, 'aux_streams_raw', None)
    if raw_aux and not getattr(args, 'aux_streams', None):
        aux_list = []
        for entry in raw_aux:
            parts = entry.split(":", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"--aux-stream requires PID:TYPE:SOURCE format, got '{entry}'"
                )
            aux_list.append(AuxStreamConfig(pid=parts[0], stream_type=parts[1],
                                            source=parts[2]))
        args.aux_streams = aux_list if aux_list else None

    phenomena_cfg = PhenomenaConfig.from_namespace(args)
    gaze_cfg      = GazeConfig.from_namespace(args)
    det_cfg       = DetectionConfig.from_namespace(args, class_ids, blacklist)
    tracker_cfg   = TrackerConfig.from_namespace(args)
    output_cfg    = OutputConfig.from_namespace(args)

    return (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
            output_cfg, active_plugins or None, phenomena_cfg,
            detection_plugins or None)


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
     output_cfg, active_plugins, phenomena_cfg,
     detection_plugins) = _build_from_args(args)

    run(source, yolo, face_det, gaze_eng,
        gaze_cfg, det_cfg, tracker_cfg, output_cfg,
        plugin_instances=active_plugins,
        detection_plugins=detection_plugins,
        phenomena_cfg=phenomena_cfg,
        fast_mode=args.fast,
        skip_phenomena=args.skip_phenomena,
        lite_overlay=args.lite_overlay,
        no_dashboard=args.no_dashboard,
        profile=args.profile)


if __name__ == "__main__":
    main()
