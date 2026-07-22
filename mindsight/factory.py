"""
mindsight.factory -- Model wiring.

Build live models + config dataclasses from a parsed argparse-style namespace.
Extracted from the legacy cli.py ``_build_from_args`` factory in SP1.5.
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
    data_collection_registry as _dc_registry,
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


def _instantiate_plugins(registry, args, label, *, verbose):
    """Instantiate every registered plugin whose ``from_args`` activates.

    Shared by :func:`build_from_namespace` (verbose, once per batch) and
    :func:`rebuild_plugin_instances` (silent, once per video for the SP3.1
    Q4/D9 per-video state reset).  Returns a list of live instances (empty
    when nothing activates).
    """
    instances: list = []
    for _pname in registry.names():
        _pcls = registry.get(_pname)
        try:
            _inst = _pcls.from_args(args)
        except Exception as _exc:
            raise RuntimeError(
                f"{label} plugin '{_pname}' failed to initialize: {_exc}"
            ) from _exc
        if _inst is not None:
            instances.append(_inst)
            if verbose:
                print(f"{label} plugin active: {_pname}")
    return instances


def rebuild_plugin_instances(ns):
    """Re-instantiate phenomena + object-detection plugin instances from *ns*.

    Uses the SAME ``from_args`` path as :func:`build_from_namespace` but
    silently (no discovery / "active" prints), so a project batch can hand
    every video FRESH stateful plugin instances (SP3.1 Q4/D9) without
    rebuilding models.  Returns ``(active_plugins, detection_plugins)``, each
    ``None`` when empty -- matching the 14-tuple convention.
    """
    active = _instantiate_plugins(_phenomena_registry, ns, "Phenomena",
                                  verbose=False)
    detection = _instantiate_plugins(_od_registry, ns, "Object detection",
                                     verbose=False)
    return (active or None, detection or None)


def build_data_plugins(ns):
    """Instantiate active :class:`DataCollectionPlugin` instances from *ns*.

    SP3.1 Q5 (Option A) minimal wiring: DataCollection plugins are built via the
    SAME ``from_args`` path as the other three registries, so the orchestration
    layer can seed ``ctx['data_plugins']`` (which ``outputs.data_pipeline.
    finalize_run`` already consumes for post-run chart generation).  Returns a
    list of live instances -- empty when nothing activates (there are no in-repo
    DataCollection plugins, so every current run gets an empty list and outputs
    are byte-unchanged).
    """
    return _instantiate_plugins(_dc_registry, ns, "Data collection",
                                verbose=False)


def build_from_namespace(ns):
    """Build all models and config objects from a parsed argparse namespace.

    Returns (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
             output_cfg, active_plugins, phenomena_cfg, detection_plugins).
    """
    args = ns
    from mindsight.Phenomena.phenomena_config import PhenomenaConfig

    if getattr(args, "no_detector", False):
        # LP2: lightweight attention studies -- faces + gaze rays + tip-based
        # phenomena only.  The stub keeps the detector contract; nulling
        # args.model lets downstream weight collection omit the YOLO family.
        if args.vp_file:
            raise ValueError(
                "--no-detector cannot be combined with --vp-file: a visual "
                "prompt needs the YOLOE detector. Drop one of the two.")
        from mindsight.ObjectDetection.model_factory import NullDetector
        print("Object detection: OFF (--no-detector) -- faces, gaze rays, "
              "and tip-based phenomena only")
        yolo, class_ids, blacklist = NullDetector(), None, set()
        args.model = None
    else:
        if getattr(args, "vp_condition", None):
            if getattr(args, "vp_ignore_conditions", False):
                raise ValueError(
                    "--vp-condition and --vp-ignore-conditions are mutually "
                    "exclusive: pick one condition's subset OR the full "
                    "prompt.")
            if not args.vp_file:
                raise ValueError(
                    "--vp-condition needs --vp-file (a condition-tagged "
                    "visual prompt to select from).")
        yolo, class_ids, blacklist = create_yolo_detector(
            model_path=args.model,
            classes=args.classes or None,
            blacklist_names=args.blacklist,
            vp_file=args.vp_file,
            vp_model=args.vp_model,
            device=getattr(args, "device", "auto"),
            vp_condition=getattr(args, "vp_condition", None),
        )
    face_det = create_face_detector(
        conf_thresh=getattr(args, 'face_conf', 0.5),
        input_size=getattr(args, 'face_input_size', 640),
        model_name=getattr(args, 'face_model', None))

    # Plugin discovery summary
    for _reg_label, _reg in (("Gaze",             _gaze_registry),
                              ("Object detection", _od_registry),
                              ("Phenomena",        _phenomena_registry)):
        if _reg.names():
            print(f"{_reg_label} plugins discovered: {', '.join(_reg.names())}")

    gaze_eng = create_gaze_engine(plugin_args=args)

    # Phenomena + object-detection plugins: instantiate whichever flags
    # activated (verbose -- one discovery summary per batch).
    active_plugins = _instantiate_plugins(
        _phenomena_registry, args, "Phenomena", verbose=True)
    detection_plugins = _instantiate_plugins(
        _od_registry, args, "Object detection", verbose=True)

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
