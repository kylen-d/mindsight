"""
mindsight.factory -- Model wiring.

Build live models + config dataclasses from a parsed argparse-style namespace.
Extracted from the ms/cli.py ``_build_from_args`` factory in SP1.5.
"""


from pathlib import Path

from mindsight.GazeTracking.gaze_factory import create_gaze_engine
from mindsight.ObjectDetection.model_factory import (
    create_face_detector,
    create_yolo_detector,
)
from mindsight.pipeline_config import (
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


def build_from_namespace(ns):
    """Build all models and config objects from a parsed argparse namespace.

    Returns (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
             output_cfg, active_plugins, phenomena_cfg, detection_plugins).
    """
    args = ns
    from mindsight.Phenomena.phenomena_config import PhenomenaConfig

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
        from mindsight.participant_ids import load_participant_csv, parse_inline_ids
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
    from mindsight.pipeline_config import AuxStreamConfig, VideoType
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

    from mindsight.pipeline_config import DepthConfig
    depth_cfg = DepthConfig.from_namespace(args)
    depth_backend = None
    if depth_cfg.enabled:
        from mindsight.DepthEstimation.depth_backend import create_depth_backend
        depth_backend = create_depth_backend(
            depth_cfg, device=getattr(args, 'device', 'auto'))
        if depth_backend is not None:
            print(f"Depth estimation: {depth_cfg.backend} "
                  f"(input {depth_cfg.input_size}px)")
            depth_backend.warmup()

    # ── Gazelle blend (core ray forming) ─────────────────────────────────
    from mindsight.PostProcessing.RayForming import GazelleProvider, RayFormingConfig
    gazelle_provider = GazelleProvider.from_namespace(
        args, device=getattr(args, 'device', 'auto'))
    # Build RayFormingConfig from the full namespace so all GUI/CLI params
    # (belief blend, smooth snap, depth, etc.) are captured directly.
    ray_cfg = RayFormingConfig.from_namespace(args)

    return (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
            output_cfg, active_plugins or None, phenomena_cfg,
            detection_plugins or None, depth_cfg, depth_backend,
            gazelle_provider, ray_cfg)
