"""
ms.cli -- Main program; orchestrates the full gaze-tracking pipeline.

Architecture
------------
The four logical stages of the run loop are each implemented in their own
module so that user plugins can swap out individual steps:

  ms/ObjectDetection/detection_pipeline.py  -> YOLO detection step
  ms/GazeTracking/gaze_pipeline.py          -> face detection + gaze + intersection
  ms/Phenomena/phenomena_pipeline.py        -> phenomena tracker init / update / summary
  ms/DataCollection/data_pipeline.py        -> CSV logging, look counts, heatmaps

This file is the thin orchestrator: it wires the stages together and owns
the CLI, model loading, and display loop.

All pipeline stages communicate through a shared ``FrameContext`` object.
Each stage reads the keys it needs and writes its results, so adding new
data never requires changing function signatures.

CLI flags
---------
Each submodule registers its own arguments via ``add_arguments(parser)``:

  ms/ObjectDetection/object_detection.py  -> --model, --conf, --classes, etc.
  ms/GazeTracking/gaze_processing.py      -> --mgaze-model, --ray-length, --gaze-cone, etc.
  ms/Phenomena/phenomena_tracking.py       -> --mutual-gaze, --social-ref, --ja-window, etc.

Run ``python MindSight.py --help`` for the full list.

Base pipeline: YOLO objects -> RetinaFace faces -> gaze estimation ->
               eye-landmark origin (auto-fallback to face bbox) ->
               ray-bbox (or cone) intersection -> joint attention + phenomena
"""


from pathlib import Path

from ms.cli_flags import parse_cli
from ms.GazeTracking.gaze_factory import create_gaze_engine
from ms.ObjectDetection.model_factory import create_face_detector, create_yolo_detector

# -- Extracted run loop --------------------------------------------------------
# The per-frame orchestrator and video/webcam loop live in ms.pipeline now.
# ``run`` stays importable from ms.cli for backward compatibility (GUI workers,
# project_runner, tests) with an identical signature.
from ms.pipeline import run
from ms.pipeline_config import (
    DetectionConfig,
    GazeConfig,
    OutputConfig,
    TrackerConfig,
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


# ==============================================================================
# CLI
# ==============================================================================

def _args(argv=None):
    """Parse argv into a namespace with ``_explicit_cli`` attached.

    The parser is generated from the pydantic schema + the FlagSpec table in
    ``ms.cli_flags``; this remains the public entry point (GUI and tests
    import it), with an identical returned-namespace shape.
    """
    return parse_cli(argv)


def _build_from_args(args):
    """Build all models and config objects from a parsed argparse namespace.

    Returns (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
             output_cfg, active_plugins, phenomena_cfg, detection_plugins).
    """
    from ms.Phenomena.phenomena_config import PhenomenaConfig

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
        from ms.participant_ids import load_participant_csv, parse_inline_ids
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

    # Parse --aux-stream SOURCE:VIDEO_TYPE:LABEL:PIDS entries
    from ms.pipeline_config import AuxStreamConfig, VideoType
    raw_aux = getattr(args, 'aux_streams_raw', None)
    if raw_aux and not getattr(args, 'aux_streams', None):
        aux_list = []
        for entry in raw_aux:
            parts = entry.split(":", 3)
            if len(parts) != 4:
                raise ValueError(
                    f"--aux-stream requires SOURCE:VIDEO_TYPE:LABEL:PIDS "
                    f"format, got '{entry}'"
                )
            source, vtype_str, label, pids_str = parts
            try:
                vtype = VideoType(vtype_str)
            except ValueError:
                raise ValueError(
                    f"Unknown video_type '{vtype_str}'. "
                    f"Options: {[v.value for v in VideoType]}"
                )
            participants = [p.strip() for p in pids_str.split(",")
                           if p.strip()]
            auto_detect = getattr(args, 'aux_auto_detect', True)
            aux_list.append(AuxStreamConfig(
                source=source,
                video_type=vtype,
                stream_label=label,
                participants=participants,
                auto_detect_faces=auto_detect,
            ))
        args.aux_streams = aux_list if aux_list else None

    # Handle --no-depth overriding --depth
    if getattr(args, 'no_depth', False):
        args.depth = False

    phenomena_cfg = PhenomenaConfig.from_namespace(args)
    gaze_cfg      = GazeConfig.from_namespace(args)
    det_cfg       = DetectionConfig.from_namespace(args, class_ids, blacklist)
    tracker_cfg   = TrackerConfig.from_namespace(args)
    output_cfg    = OutputConfig.from_namespace(args)

    from ms.pipeline_config import DepthConfig
    depth_cfg = DepthConfig.from_namespace(args)
    depth_backend = None
    if depth_cfg.enabled:
        from ms.DepthEstimation.depth_backend import create_depth_backend
        depth_backend = create_depth_backend(
            depth_cfg, device=getattr(args, 'device', 'auto'))
        if depth_backend is not None:
            print(f"Depth estimation: {depth_cfg.backend} "
                  f"(input {depth_cfg.input_size}px)")
            depth_backend.warmup()

    # ── Gazelle blend (core ray forming) ─────────────────────────────────
    from ms.PostProcessing.RayForming import GazelleProvider, RayFormingConfig
    gazelle_provider = GazelleProvider.from_namespace(
        args, device=getattr(args, 'device', 'auto'))
    # Build RayFormingConfig from the full namespace so all GUI/CLI params
    # (belief blend, smooth snap, depth, etc.) are captured directly.
    ray_cfg = RayFormingConfig.from_namespace(args)

    return (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
            output_cfg, active_plugins or None, phenomena_cfg,
            detection_plugins or None, depth_cfg, depth_backend,
            gazelle_provider, ray_cfg)


def main():
    args = _args()

    # Apply pipeline YAML if specified (CLI flags take precedence)
    if args.pipeline:
        from ms.pipeline_loader import load_pipeline
        load_pipeline(args.pipeline, args)
        print(f"Loaded pipeline config: {args.pipeline}")

    # Project mode: batch-process all videos in a project directory
    if args.project:
        from ms.project_runner import run_project
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
     detection_plugins, depth_cfg, depth_backend,
     gazelle_provider, ray_cfg) = _build_from_args(args)

    run(source, yolo, face_det, gaze_eng,
        gaze_cfg, det_cfg, tracker_cfg, output_cfg,
        plugin_instances=active_plugins,
        detection_plugins=detection_plugins,
        phenomena_cfg=phenomena_cfg,
        fast_mode=args.fast,
        skip_phenomena=args.skip_phenomena,
        lite_overlay=args.lite_overlay,
        no_dashboard=args.no_dashboard,
        profile=args.profile,
        depth_cfg=depth_cfg, depth_backend=depth_backend,
        gazelle_provider=gazelle_provider, ray_cfg=ray_cfg)


if __name__ == "__main__":
    main()
