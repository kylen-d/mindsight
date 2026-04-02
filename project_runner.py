"""
project_runner.py — Project-based batch processing for MindSight.

A Project is a directory with a standard layout::

    MyProject/
    ├── Inputs/
    │   ├── Videos/         # Source videos/images
    │   └── Prompts/        # VP files for this project
    ├── Outputs/
    │   ├── CSV Files/      # Per-video CSV outputs
    │   └── Videos/         # Per-video annotated outputs
    └── Pipeline/
        └── pipeline.yaml   # Project-specific pipeline config

Usage
-----
    python MindSight.py --project Projects/MyProject/

This:
1. Loads ``Pipeline/pipeline.yaml`` as the base pipeline config.
2. Iterates over all videos in ``Inputs/Videos/``.
3. Writes outputs to ``Outputs/`` organized per-video.
4. Applies any VP files from ``Inputs/Prompts/``.
"""

from __future__ import annotations

from pathlib import Path

from constants import IMAGE_EXTS


# Supported video extensions (beyond image extensions in constants.py)
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
_ALL_MEDIA = _VIDEO_EXTS | IMAGE_EXTS


def validate_project(project_dir: str | Path) -> Path:
    """
    Validate that a project directory has the required structure.

    Returns the resolved project path.
    Raises FileNotFoundError or ValueError on problems.
    """
    project = Path(project_dir).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project}")

    inputs_videos = project / "Inputs" / "Videos"
    if not inputs_videos.is_dir():
        raise ValueError(f"Project missing Inputs/Videos/ directory: {project}")

    # Ensure output directories exist
    (project / "Outputs" / "CSV Files").mkdir(parents=True, exist_ok=True)
    (project / "Outputs" / "Videos").mkdir(parents=True, exist_ok=True)

    return project


def discover_sources(project: Path) -> list[Path]:
    """
    Find all video/image files in the project's Inputs/Videos/ directory.

    Returns a sorted list of paths.
    """
    inputs_dir = project / "Inputs" / "Videos"
    sources = sorted(
        p for p in inputs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _ALL_MEDIA
    )
    return sources


def discover_vp_file(project: Path) -> str | None:
    """
    Find the first .vp.json file in the project's Inputs/Prompts/ directory.

    Returns the path as a string, or None if no VP file is found.
    """
    prompts_dir = project / "Inputs" / "Prompts"
    if not prompts_dir.is_dir():
        return None
    for p in sorted(prompts_dir.iterdir()):
        if p.name.endswith('.vp.json'):
            return str(p)
    return None


def project_output_paths(project: Path, source: Path) -> dict:
    """
    Compute output file paths for a given source within a project.

    Returns a dict with keys: save, log, summary, heatmap.
    """
    stem = source.stem
    return {
        'save':    str(project / "Outputs" / "Videos" / f"{stem}_Video_Output.mp4"),
        'log':     str(project / "Outputs" / "CSV Files" / f"{stem}_Events.csv"),
        'summary': str(project / "Outputs" / "CSV Files" / f"{stem}_Summary.csv"),
        'heatmap': str(project / "Outputs" / f"{stem}_Heatmap"),
    }


def run_project(project_dir: str | Path, run_fn, build_fn, args_ns) -> None:
    """
    Run MindSight on all sources in a project directory.

    Parameters
    ----------
    project_dir : str or Path
        Path to the project directory.
    run_fn : callable
        The ``run()`` function from MindSight.py.
    build_fn : callable
        A function that takes an argparse namespace and returns
        (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
         active_plugins, phenomena_cfg).
    args_ns : Namespace
        The parsed argparse namespace (with pipeline config already merged).
    """
    from pipeline_loader import load_pipeline
    from pipeline_config import OutputConfig

    project = validate_project(project_dir)
    sources = discover_sources(project)

    if not sources:
        print(f"No media files found in {project / 'Inputs' / 'Videos'}")
        return

    # Load project pipeline config if it exists
    pipeline_yaml = project / "Pipeline" / "pipeline.yaml"
    if pipeline_yaml.exists():
        load_pipeline(pipeline_yaml, args_ns)
        print(f"Loaded project pipeline: {pipeline_yaml}")

    # Apply project VP file if no CLI override
    vp = discover_vp_file(project)
    if vp and not getattr(args_ns, 'vp_file', None):
        args_ns.vp_file = vp
        print(f"Using project VP file: {vp}")

    # Build models once for the whole project
    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
     active_plugins, phenomena_cfg) = build_fn(args_ns)

    print(f"\nProject: {project.name}")
    print(f"Sources: {len(sources)} file(s)")
    print("=" * 60)

    for i, source in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] Processing: {source.name}")
        print("-" * 40)

        # Override output paths for this source
        paths = project_output_paths(project, source)
        run_output = OutputConfig(
            save=paths['save'],
            log_path=paths['log'],
            summary_path=paths['summary'],
            heatmap_path=paths['heatmap'],
        )

        try:
            run_fn(str(source), yolo, face_det, gaze_eng,
                   gaze_cfg, det_cfg, tracker_cfg, run_output,
                   plugin_instances=active_plugins,
                   phenomena_cfg=phenomena_cfg)
        except Exception as exc:
            print(f"Error processing {source.name}: {exc}")
            continue

    print(f"\nProject complete. Outputs in: {project / 'Outputs'}")
