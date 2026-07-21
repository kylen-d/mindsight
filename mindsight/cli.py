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
from mindsight.factory import build_data_plugins, build_from_namespace

# -- Extracted run loop --------------------------------------------------------
# The per-frame orchestrator and video/webcam loop live in mindsight.pipeline now.
# ``run`` stays importable from mindsight.cli for backward compatibility (GUI workers,
# project.runner, tests) with an identical signature.
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

    # Preflight report and exit (SP3.1 D16/Q8): requires --project.  Read-only --
    # builds no models, runs no videos.  Exit 0 when no check FAILED, else 1.
    if args.preflight:
        import sys

        if not args.project:
            print("--preflight requires --project DIR")
            sys.exit(2)
        from mindsight.project.preflight import format_report, run_preflight
        from mindsight.project.runner import load_project_config
        from pathlib import Path
        project = Path(args.project)
        project_cfg = load_project_config(project) if project.is_dir() else None
        report = run_preflight(project, project_cfg, ns=args)
        print(format_report(report, title=project.name))
        sys.exit(0 if report.ok else 1)

    # Project mode: batch-process all videos in a project directory.  The CLI is
    # a thin consumer of the event stream (D1): the iterator prints the batch
    # narration + Done-lines and owns ledger/manifest/global-CSV machinery; the
    # CLI only drives the cv2 display loop ('q' cancels the batch, T9).
    if args.project:
        import cv2

        from mindsight.pipeline import CancelToken
        from mindsight.project.events import (
            VideoDone,
            VideoError,
            VideoFrame,
            VideoSkipped,
        )
        from mindsight.project.runner import iter_project_runs

        cancel = CancelToken()
        try:
            for event in iter_project_runs(args.project, args, cancel=cancel,
                                           resume=not args.no_resume):
                if isinstance(event, VideoFrame):
                    cv2.imshow("MindSight", event.result.annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cancel.cancel()
                elif isinstance(event, (VideoDone, VideoError, VideoSkipped)):
                    cv2.destroyAllWindows()
        finally:
            cv2.destroyAllWindows()
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

    # SP3.1 Q5 (Option A): seed active DataCollection plugins into the run so
    # finalize_run's chart hook can reach them (empty -- no in-repo plugins).
    data_plugins = build_data_plugins(args)

    from mindsight.outputs import provenance
    started = provenance.utcnow_iso()
    run(source, yolo, face_det, gaze_eng,
        gaze_cfg, det_cfg, tracker_cfg, output_cfg,
        plugin_instances=active_plugins,
        detection_plugins=detection_plugins,
        phenomena_cfg=phenomena_cfg,
        fast_mode=args.fast,
        skip_phenomena=args.skip_phenomena,
        lite_overlay=args.lite_overlay,
        overlay_theme=getattr(args, 'overlay_theme', 'classic'),
        no_dashboard=args.no_dashboard,
        profile=args.profile,
        save_detections=getattr(args, 'save_detections', False),
        depth_cfg=depth_cfg, depth_backend=depth_backend,
        gazelle_provider=gazelle_provider, ray_cfg=ray_cfg,
        data_plugins=data_plugins)

    # Per-run provenance manifest (D8): written only when a file output is
    # configured; located next to the summary/log/saved-video (Q4).
    outputs = provenance.resolve_single_source_outputs(args, source)
    manifest_path = provenance.manifest_path_for(outputs)
    if manifest_path:
        from mindsight.config import PipelineConfig
        provenance.write_run_manifest(
            manifest_path, ns=args,
            config=PipelineConfig.from_namespace(args),
            source=source, output_paths=outputs,
            started=started, finished=provenance.utcnow_iso(),
            status="completed")


if __name__ == "__main__":
    main()
