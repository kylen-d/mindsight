"""
project/runner.py — Project-based batch processing for MindSight.

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

from mindsight.constants import IMAGE_EXTS
from mindsight.participant_ids import load_participant_csv
from mindsight.pipeline_config import ProjectConfig, ProjectOutputConfig

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
        ├── eye_only/
        │   ├── S70.mp4
        │   └── S71.mp4
        ├── face_closeup/
        │   └── S70.mp4
        └── wide_closeup/
            └── S70+S71.mp4

    Subdirectory name maps to ``video_type``.  File stem is used for
    ``participants``: single name (``S70``) maps to one participant,
    ``+``-separated names (``S70+S71``) map to multiple.  The
    ``stream_label`` is auto-generated from directory and filename.

    Returns a list of ``AuxStreamConfig`` objects with resolved absolute
    paths, or an empty list if the directory does not exist.
    """
    from mindsight.pipeline_config import AuxStreamConfig, VideoType

    _VTYPE_MAP = {
        'eye_only': VideoType.EYE_ONLY,
        'face_closeup': VideoType.FACE_CLOSEUP,
        'wide_closeup': VideoType.WIDE_CLOSEUP,
    }

    aux_dir = project / "Inputs" / "AuxStreams"
    if not aux_dir.is_dir():
        return []

    configs: list[AuxStreamConfig] = []
    for type_dir in sorted(aux_dir.iterdir()):
        if not type_dir.is_dir():
            continue
        dir_name = type_dir.name
        vtype = _VTYPE_MAP.get(dir_name, VideoType.CUSTOM)

        for media in sorted(type_dir.iterdir()):
            if media.is_file() and media.suffix.lower() in _ALL_MEDIA:
                stem = media.stem
                # Support multi-participant files: S70+S71.mp4
                if '+' in stem:
                    participants = [p.strip() for p in stem.split('+')]
                else:
                    participants = [stem]

                stream_label = f"{dir_name}_{stem}"
                configs.append(AuxStreamConfig(
                    source=str(media.resolve()),
                    video_type=vtype,
                    stream_label=stream_label,
                    participants=participants,
                    auto_detect_faces=(vtype in (VideoType.WIDE_CLOSEUP,
                                                 VideoType.FACE_CLOSEUP)),
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
        'summary': str(out_root / "CSV Files" / f"{stem}_summary.csv"),
        'heatmap': str(out_root / f"{stem}_Heatmap"),
    }


def iter_project_runs(project_dir: str | Path, ns, *, project_cfg=None,
                      cancel=None, resume=True):
    """Run every source in a project, yielding a :mod:`ProjectEvent <mindsight.project.events>` stream.

    This is the SINGLE project-batch implementation (SP3.1 D1) consumed by both
    ``cli.main --project`` and the GUI ``ProjectWorker``.  It performs the exact
    sequence the pre-SP3 loop did -- load project config (only if not supplied),
    validate, discover sources, load the pipeline YAML (passing *ns* through
    untouched so the CLI/GUI YAML-precedence fork is preserved, T7), discover VP
    / participant maps / aux streams, build the models ONCE, then per source
    consult the resume ledger, run a per-video :class:`~mindsight.pipeline.Pipeline`
    over the shared models, write the per-video manifest, and record the terminal
    ledger state -- finally generating the global + per-condition CSVs.

    The generator is display-free: it yields ``VideoFrame`` per processed frame
    and never calls ``cv2.imshow`` or touches a queue.  The batch narration lines
    are printed here (so the CLI transcript is unchanged); consumers layer their
    own presentation on the events.

    Parameters
    ----------
    project_dir : str or Path
        Path to the project directory.
    ns : Namespace
        The parsed namespace (pipeline YAML is merged into it here).  Passed to
        ``load_pipeline`` and ``build_from_namespace`` UNCHANGED (T7/T4).
    project_cfg : ProjectConfig or None
        In-memory project config (the GUI's possibly-unsaved edits).  When
        ``None`` the ``project.yaml`` on disk is loaded (the CLI path).
    cancel : CancelToken or None
        Batch-level cancel.  When cancelled, the current video finalizes cleanly
        (T8) and the batch stops before the next source; global CSVs still run.
    resume : bool
        When ``False`` (``--no-resume`` / "Re-run all") the ledger ``decide`` is
        bypassed entirely -- everything reprocesses, nothing is archived.
    """
    from mindsight.config import PipelineConfig
    from mindsight.config_compat import load_pipeline
    from mindsight.factory import build_data_plugins, rebuild_plugin_instances
    from mindsight.outputs import provenance
    from mindsight.pipeline import (
        CancelToken,
        Pipeline,
        RunOptions,
        build_from_namespace,
    )
    from mindsight.pipeline_config import OutputConfig
    from mindsight.project.events import (
        BatchDone,
        BatchStarted,
        VideoArchived,
        VideoDone,
        VideoError,
        VideoFrame,
        VideoSkipped,
        VideoStarted,
    )
    from mindsight.project.ledger import Ledger, compute_video_hash

    project = Path(project_dir).resolve()

    # Load project.yaml ONLY if the caller did not supply one (the GUI passes an
    # in-memory ProjectConfig with possibly-unsaved edits; the CLI passes None).
    if project_cfg is None:
        project_cfg = load_project_config(project)

    project = validate_project(project_dir, project_cfg)
    sources = discover_sources(project)

    if not sources:
        print(f"No media files found in {project / 'Inputs' / 'Videos'}")
        return

    # Load pipeline YAML — project.yaml can override the default path.  ns is
    # passed through untouched: the CLI/GUI _explicit_cli precedence fork lives
    # inside load_pipeline and must stay forked (T7).
    if project_cfg and project_cfg.pipeline_path:
        pipeline_yaml = project / project_cfg.pipeline_path
    else:
        pipeline_yaml = project / "Pipeline" / "pipeline.yaml"
    if pipeline_yaml.exists():
        load_pipeline(pipeline_yaml, ns)
        print(f"Loaded project pipeline: {pipeline_yaml}")

    # Apply project VP file if no CLI override
    vp = discover_vp_file(project)
    if vp and not getattr(ns, 'vp_file', None):
        ns.vp_file = vp
        print(f"Using project VP file: {vp}")

    # Build models once for the whole project (14-tuple contract, T4)
    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
     active_plugins, phenomena_cfg, detection_plugins,
     depth_cfg, depth_backend,
     gazelle_provider, ray_cfg) = build_from_namespace(ns)

    # SP3.1 Q5 (Option A): active DataCollection plugins seeded into every
    # video's Pipeline (finalize_run consumes ctx['data_plugins']).  Empty for
    # every current project -- no in-repo DataCollection plugins, so outputs are
    # byte-unchanged.  Built once per batch, alongside the models.
    data_plugins = build_data_plugins(ns)

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
    from mindsight.participant_ids import load_aux_streams_from_csv
    csv_path = project / "participant_ids.csv"
    if csv_path.is_file():
        csv_aux = load_aux_streams_from_csv(csv_path)
        if csv_aux:
            aux_streams = aux_streams + csv_aux

    if aux_streams:
        vtypes = {a.video_type.value for a in aux_streams}
        all_pids = set()
        for a in aux_streams:
            all_pids.update(a.participants)
        print(f"Auxiliary streams: {len(aux_streams)} "
              f"({len(all_pids)} participant(s), {len(vtypes)} type(s))")

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

    # Provenance config is batch-level (models built once); per-video source /
    # output_paths / status are recorded per manifest.
    manifest_config = PipelineConfig.from_namespace(ns)

    # Resume ledger (D9): compute the batch config_hash ONCE (models are built
    # once per batch); per-video hashes + status are consulted/recorded below.
    # resume=False bypasses `decide` entirely (still rewrites the ledger, never
    # archives).
    no_resume = not resume
    batch_weights = provenance.collect_weights(ns)
    config_hash = provenance.run_identity(
        ns, config=manifest_config, weights=batch_weights)
    ledger = Ledger.load(out_root)

    options = RunOptions(
        fast_mode=getattr(ns, 'fast', False),
        skip_phenomena=getattr(ns, 'skip_phenomena', 0),
        lite_overlay=getattr(ns, 'lite_overlay', False),
        no_dashboard=getattr(ns, 'no_dashboard', False),
        profile=getattr(ns, 'profile', False),
    )

    yield BatchStarted(total=len(sources), out_root=out_root)

    for i, source in enumerate(sources, 1):
        # Batch-level cancel: stop before the next source (the previous video
        # finalized cleanly through its own cancel token); global CSVs still run.
        if cancel is not None and cancel.cancelled:
            break

        # Override output paths for this source
        paths = project_output_paths(project, source, project_cfg)
        video_pid_map = pid_maps.get(source.name) if pid_maps else None

        # Build condition string for this video
        video_tags = (project_cfg.conditions.get(source.name, [])
                      if project_cfg else [])
        conditions_str = "|".join(video_tags) if video_tags else ""

        # Consult the ledger (unless resume disabled): skip unchanged done
        # videos, archive superseded outputs on a config/source change.
        video_hash = compute_video_hash(
            source, pid_map=video_pid_map, conditions=conditions_str,
            aux_streams=aux_streams if aux_streams else None)
        hashes = (config_hash, video_hash)
        if not no_resume:
            decision = ledger.decide(source.name, hashes)
            if decision == "skip":
                print(f"[{i}/{len(sources)}] Skipping {source.name} "
                      f"(done, config unchanged)")
                yield VideoSkipped(run_id=source.name,
                                   reason="done, config unchanged")
                continue
            if decision == "redo_archive":
                dest = ledger.archive(source.name)
                yield VideoArchived(run_id=source.name, dest=dest)

        print(f"\n[{i}/{len(sources)}] Processing: {source.name}")
        print("-" * 40)
        yield VideoStarted(index=i, total=len(sources),
                           run_id=source.name, source=source)

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

        # Per-video state reset (SP3.1 Q4/D9): a video's numbers must not depend
        # on its batch position.  Rebuild the cheap stateful objects that are
        # SHARED across videos while keeping the loaded model weights -- a fresh
        # Gaze-LLE scheduler/heatmap-cache and fresh plugin instances.  The
        # pipeline's own trackers/smoothers are already fresh per-video locals of
        # _run_video, so track-IDs and belief anchors restart on their own.  Done
        # uniformly (first video included): resetting a pristine provider is a
        # no-op, so video a stays byte-identical to a standalone run.
        if gazelle_provider is not None:
            gazelle_provider.reset()
        active_plugins, detection_plugins = rebuild_plugin_instances(ns)

        # Per-video Pipeline sharing the once-built models; fresh cancel token
        # per video (batch cancel is mirrored into it so the current video
        # finalizes through the normal post-run paths, T8).
        video_pipeline = Pipeline(
            yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
            gaze_cfg=gaze_cfg, det_cfg=det_cfg, tracker_cfg=tracker_cfg,
            output_cfg=run_output, plugin_instances=active_plugins,
            detection_plugins=detection_plugins, phenomena_cfg=phenomena_cfg,
            depth_cfg=depth_cfg, depth_backend=depth_backend,
            gazelle_provider=gazelle_provider, ray_cfg=ray_cfg,
            data_plugins=data_plugins,
        )
        video_cancel = CancelToken()

        # Mark in_progress BEFORE the run so a kill -9 mid-run is recoverable
        # (T8: never marked from inside the Pipeline generator).
        ledger.mark_started(source.name, hashes, paths)
        started = provenance.utcnow_iso()
        status, error = "completed", None
        try:
            for result in video_pipeline.run(str(source), options=options,
                                             cancel=video_cancel):
                yield VideoFrame(run_id=source.name, result=result)
                if cancel is not None and cancel.cancelled:
                    video_cancel.cancel()
        except Exception as exc:
            print(f"Error processing {source.name}: {exc}")
            status, error = "error", str(exc)

        # Per-video provenance manifest (D8): Outputs/CSV Files/{stem}_manifest.json
        # (Q4). Written on success AND on error (status recorded either way).
        manifest_path = Path(paths['summary']).parent / f"{source.stem}_manifest.json"
        provenance.write_run_manifest(
            str(manifest_path), ns=ns, config=manifest_config,
            source=source, output_paths=paths,
            started=started, finished=provenance.utcnow_iso(),
            status=status, error=error)
        # Record terminal ledger state AFTER the orchestration layer returns
        # (T8: done/error is the layer's decision, not the generator's).
        if status == "error":
            ledger.mark_error(source.name, error)
            yield VideoError(run_id=source.name, error=error)
            continue
        ledger.mark_done(source.name, str(manifest_path))
        yield VideoDone(run_id=source.name, manifest_path=str(manifest_path))

    # ── Post-processing: generate global and per-condition CSVs ──────────
    csv_dir = out_root / "CSV Files"
    print("\nGenerating global CSVs...")
    from mindsight.outputs.global_csv import (
        GLOBAL_TABLES,
        generate_condition_csvs,
        generate_global_csv,
    )

    split = bool(project_cfg and project_cfg.conditions)
    condition_dir = out_root / "By Condition"
    for suffix, out_name in GLOBAL_TABLES:
        global_path = generate_global_csv(csv_dir, suffix, out_name)
        if split and global_path:
            generate_condition_csvs(global_path, condition_dir, suffix)

    print(f"\nProject complete. Outputs in: {out_root}")
    yield BatchDone(out_root=out_root)
