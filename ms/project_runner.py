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

import yaml

from ms.constants import IMAGE_EXTS
from ms.participant_ids import load_participant_csv
from ms.pipeline_config import ProjectConfig, ProjectOutputConfig

# Supported video extensions (beyond image extensions in constants.py)
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
_ALL_MEDIA = _VIDEO_EXTS | IMAGE_EXTS


def load_project_config(project: Path) -> ProjectConfig | None:
    """Load ``project.yaml`` from the project root, if it exists.

    Returns a ``ProjectConfig`` or ``None`` when the file is absent.
    """
    yaml_path = project / "project.yaml"
    if not yaml_path.is_file():
        return None

    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # Parse conditions: each value should be a list of tag strings
    conditions: dict[str, list[str]] = {}
    for video, tags in (raw.get("conditions") or {}).items():
        if isinstance(tags, str):
            tags = [tags]
        conditions[video] = list(tags)

    # Parse participants: {video: {track_id(int): label(str)}}
    participants: dict[str, dict[int, str]] = {}
    for video, mapping in (raw.get("participants") or {}).items():
        participants[video] = {int(k): str(v) for k, v in mapping.items()}

    # Parse output section
    out_raw = raw.get("output") or {}
    output_cfg = ProjectOutputConfig(
        directory=out_raw.get("directory"),
    )

    return ProjectConfig(
        pipeline_path=raw.get("pipeline"),
        conditions=conditions,
        participants=participants,
        output=output_cfg,
    )


def save_project_config(project: Path, cfg: ProjectConfig) -> Path:
    """Write ``project.yaml`` to the project root.

    Returns the path to the written file.
    """
    data: dict = {"version": 1}

    if cfg.pipeline_path:
        data["pipeline"] = cfg.pipeline_path

    if cfg.conditions:
        data["conditions"] = cfg.conditions

    if cfg.participants:
        # Convert int keys to plain ints for clean YAML
        data["participants"] = {
            video: {int(tid): label for tid, label in mapping.items()}
            for video, mapping in cfg.participants.items()
        }

    if cfg.output.directory:
        data["output"] = {"directory": cfg.output.directory}

    yaml_path = project / "project.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return yaml_path


def validate_project(project_dir: str | Path,
                     project_cfg: ProjectConfig | None = None) -> Path:
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

    # Resolve output root from project config or default
    if project_cfg and project_cfg.output:
        out_root = project_cfg.output.resolve_root(project)
    else:
        out_root = project / "Outputs"

    # Ensure output directories exist
    (out_root / "CSV Files").mkdir(parents=True, exist_ok=True)
    (out_root / "Videos").mkdir(parents=True, exist_ok=True)

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


def discover_participant_ids(project: Path) -> dict[str, dict[int, str]] | None:
    """
    Look for ``participant_ids.csv`` at the project root.

    Returns the parsed mapping ``{video_filename: {track_id: label}}``
    or ``None`` if no file is found.
    """
    csv_path = project / "participant_ids.csv"
    if not csv_path.is_file():
        return None
    return load_participant_csv(csv_path)


def discover_aux_streams(project: Path) -> list:
    """Discover auxiliary video streams from ``Inputs/AuxStreams/``.

    Expected layout::

        Inputs/AuxStreams/
        ├── eye_camera/
        │   ├── S70.mp4
        │   └── S71.mp4
        └── first_person_view/
            └── S70.mp4

    Subdirectory name → ``stream_type``, file stem → ``pid``.

    Returns a list of ``AuxStreamConfig`` objects with resolved absolute
    paths, or an empty list if the directory does not exist.
    """
    from ms.pipeline_config import AuxStreamConfig

    aux_dir = project / "Inputs" / "AuxStreams"
    if not aux_dir.is_dir():
        return []

    configs: list[AuxStreamConfig] = []
    for type_dir in sorted(aux_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        stream_type = type_dir.name
        for media in sorted(type_dir.iterdir()):
            if media.is_file() and media.suffix.lower() in _ALL_MEDIA:
                configs.append(AuxStreamConfig(
                    pid=media.stem,
                    stream_type=stream_type,
                    source=str(media.resolve()),
                ))
    return configs


def project_output_paths(project: Path, source: Path,
                         project_cfg: ProjectConfig | None = None) -> dict:
    """
    Compute output file paths for a given source within a project.

    Returns a dict with keys: save, log, summary, heatmap.
    """
    if project_cfg and project_cfg.output:
        out_root = project_cfg.output.resolve_root(project)
    else:
        out_root = project / "Outputs"

    stem = source.stem
    return {
        'save':    str(out_root / "Videos" / f"{stem}_Video_Output.mp4"),
        'log':     str(out_root / "CSV Files" / f"{stem}_Events.csv"),
        'summary': str(out_root / "CSV Files" / f"{stem}_Summary.csv"),
        'heatmap': str(out_root / f"{stem}_Heatmap"),
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
         active_plugins, phenomena_cfg, detection_plugins).
    args_ns : Namespace
        The parsed argparse namespace (with pipeline config already merged).
    """
    from ms.pipeline_config import OutputConfig
    from ms.pipeline_loader import load_pipeline

    project = Path(project_dir).resolve()

    # Load project.yaml if it exists (study metadata: conditions, participants, output)
    project_cfg = load_project_config(project)

    project = validate_project(project_dir, project_cfg)
    sources = discover_sources(project)

    if not sources:
        print(f"No media files found in {project / 'Inputs' / 'Videos'}")
        return

    # Load pipeline YAML — project.yaml can override the default path
    if project_cfg and project_cfg.pipeline_path:
        pipeline_yaml = project / project_cfg.pipeline_path
    else:
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
     active_plugins, phenomena_cfg, detection_plugins) = build_fn(args_ns)

    # Discover per-video participant ID mappings
    # project.yaml participants take precedence over participant_ids.csv
    if project_cfg and project_cfg.participants:
        pid_maps = project_cfg.participants
        print(f"Loaded participant IDs from project.yaml for {len(pid_maps)} video(s)")
    else:
        pid_maps = discover_participant_ids(project)
        if pid_maps is not None:
            print(f"Loaded participant IDs for {len(pid_maps)} video(s)")

    # Discover auxiliary streams (directory convention)
    aux_streams = discover_aux_streams(project)

    # Also check for CSV-defined aux streams
    from ms.participant_ids import load_aux_streams_from_csv
    csv_path = project / "participant_ids.csv"
    if csv_path.is_file():
        csv_aux = load_aux_streams_from_csv(csv_path)
        if csv_aux:
            aux_streams = aux_streams + csv_aux

    if aux_streams:
        types = {a.stream_type for a in aux_streams}
        pids = {a.pid for a in aux_streams}
        print(f"Auxiliary streams: {len(aux_streams)} "
              f"({len(pids)} participant(s), {len(types)} type(s))")

    # Resolve output root
    if project_cfg and project_cfg.output:
        out_root = project_cfg.output.resolve_root(project)
    else:
        out_root = project / "Outputs"

    print(f"\nProject: {project.name}")
    print(f"Sources: {len(sources)} file(s)")
    if project_cfg and project_cfg.conditions:
        tags = set()
        for t in project_cfg.conditions.values():
            tags.update(t)
        print(f"Conditions: {len(tags)} unique tag(s)")
    print(f"Output root: {out_root}")
    print("=" * 60)

    for i, source in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] Processing: {source.name}")
        print("-" * 40)

        # Override output paths for this source
        paths = project_output_paths(project, source, project_cfg)
        video_pid_map = pid_maps.get(source.name) if pid_maps else None

        # Build condition string for this video
        video_tags = (project_cfg.conditions.get(source.name, [])
                      if project_cfg else [])
        conditions_str = "|".join(video_tags) if video_tags else ""

        run_output = OutputConfig(
            save=paths['save'],
            log_path=paths['log'],
            summary_path=paths['summary'],
            heatmap_path=paths['heatmap'],
            pid_map=video_pid_map,
            aux_streams=aux_streams if aux_streams else None,
            video_name=source.stem,
            conditions=conditions_str,
        )

        try:
            run_fn(str(source), yolo, face_det, gaze_eng,
                   gaze_cfg, det_cfg, tracker_cfg, run_output,
                   plugin_instances=active_plugins,
                   detection_plugins=detection_plugins,
                   phenomena_cfg=phenomena_cfg,
                   fast_mode=getattr(args_ns, 'fast', False),
                   skip_phenomena=getattr(args_ns, 'skip_phenomena', 0),
                   lite_overlay=getattr(args_ns, 'lite_overlay', False),
                   no_dashboard=getattr(args_ns, 'no_dashboard', False),
                   profile=getattr(args_ns, 'profile', False))
        except Exception as exc:
            print(f"Error processing {source.name}: {exc}")
            continue

    # ── Post-processing: generate global and per-condition CSVs ──────────
    csv_dir = out_root / "CSV Files"
    print("\nGenerating global CSVs...")
    from ms.DataCollection.global_csv import generate_condition_csvs, generate_global_csv

    global_summary = generate_global_csv(csv_dir, "summary")
    global_events = generate_global_csv(csv_dir, "events")

    if project_cfg and project_cfg.conditions:
        condition_dir = out_root / "By Condition"
        condition_dir.mkdir(parents=True, exist_ok=True)
        if global_summary:
            generate_condition_csvs(global_summary, condition_dir, "summary")
        if global_events:
            generate_condition_csvs(global_events, condition_dir, "events")

    print(f"\nProject complete. Outputs in: {out_root}")
