"""
mindsight.cli -- Main program; orchestrates the full gaze-tracking pipeline.

Architecture
------------
The four logical stages of the run loop are each implemented in their own
module so that user plugins can swap out individual steps:

  mindsight/ObjectDetection/detection_pipeline.py  -> YOLO detection step
  mindsight/GazeTracking/gaze_pipeline.py          -> face detection + gaze + intersection
  mindsight/Phenomena/phenomena_pipeline.py        -> phenomena tracker init / update / summary
  mindsight/outputs/data_pipeline.py        -> CSV logging, look counts, heatmaps

This file is the thin orchestrator: it wires the stages together and owns
the CLI, model loading, and display loop.

All pipeline stages communicate through a shared ``FrameContext`` object.
Each stage reads the keys it needs and writes its results, so adding new
data never requires changing function signatures.

CLI flags
---------
Core flags are generated from the pydantic schema (``mindsight/config.py``) via the
``FlagSpec`` table in ``mindsight/cli_flags.py`` -- ``_args`` just delegates to
``cli_flags.parse_cli``.  Plugins still contribute their own flags at runtime
through the ``add_arguments(parser)`` method on each registered plugin class.

Run ``python MindSight.py --help`` for the full list.

Base pipeline: YOLO objects -> RetinaFace faces -> gaze estimation ->
               eye-landmark origin (auto-fallback to face bbox) ->
               ray-bbox (or cone) intersection -> joint attention + phenomena
"""


from mindsight.cli_flags import parse_cli
from mindsight.factory import build_from_namespace

# -- Extracted run loop --------------------------------------------------------
# The per-frame orchestrator and video/webcam loop live in mindsight.pipeline now.
# ``run`` stays importable from mindsight.cli for backward compatibility (GUI workers,
# project_runner, tests) with an identical signature.
from mindsight.pipeline import run


# ==============================================================================
# CLI
# ==============================================================================

def _args(argv=None):
    """Parse argv into a namespace with ``_explicit_cli`` attached.

    The parser is generated from the pydantic schema + the FlagSpec table in
    ``mindsight.cli_flags``; this remains the public entry point (GUI and tests
    import it), with an identical returned-namespace shape.
    """
    return parse_cli(argv)


def main():
    args = _args()

    # Apply pipeline YAML if specified (CLI flags take precedence)
    if args.pipeline:
        from mindsight.config_compat import load_pipeline
        load_pipeline(args.pipeline, args)
        print(f"Loaded pipeline config: {args.pipeline}")

    # Project mode: batch-process all videos in a project directory
    if args.project:
        from mindsight.project_runner import run_project
        run_project(args.project, run, build_from_namespace, args)
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
     gazelle_provider, ray_cfg) = build_from_namespace(args)

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
