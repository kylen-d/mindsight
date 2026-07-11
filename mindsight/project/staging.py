"""
project/staging.py -- Run staging currency + discovery (SP3.1 D2 / Batch E).

A :class:`RunSpec` is ONE staged run: a source video plus the metadata that
drives its processing (participant map, condition tags) and its output paths.
It is the single currency the unified batch loop (:func:`iter_project_runs
<mindsight.project.runner.iter_project_runs>`) consumes -- the CLI, the GUI, and
the manual "Add single run" path all produce ``RunSpec`` lists.

Two input layouts are supported (SP3.1 Q1), auto-detected -- never a
``project.yaml`` switch:

- **legacy / flat** (``Inputs/Videos/*.mp4``, the paper contract, DEFAULT): one
  run per video; ``run_id`` is the video filename (``a.mp4``) so pre-SP3 resume
  ledgers keep resuming; outputs stay flat under ``Outputs/CSV Files`` +
  ``Outputs/Videos`` (booby trap T1 -- byte-untouched).
- **run-folder** (``Inputs/Runs/<run_id>/``): one run per subfolder, each holding
  EXACTLY ONE primary video, an optional ``run.yaml`` (SP3.1 Q2), and an optional
  ``aux/<type>/`` subdir; ``run_id`` is the folder name; outputs mirror per-run
  under ``Outputs/Runs/<run_id>/`` (SP3.1 Q3, wired in Step 10).

A project with BOTH ``Inputs/Runs/`` and ``Inputs/Videos/`` populated is
ambiguous -- a preflight FAIL, and a hard error on the run path.

``run.yaml`` (Q2) keys, all optional: ``participants`` (``track_id -> label``,
the pid map), ``conditions`` (tag or list of tags), and the manifest-only
``date`` / ``session`` / ``notes`` / ``extra``.  Unknown top-level keys are a
preflight WARN (typo guard).  Metadata precedence per run is
``run.yaml > project.yaml > participant_ids.csv``.  Only participants /
conditions influence processing or CSV columns; everything else travels into the
per-run manifest, never a CSV column.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mindsight.project.runner import (
    _ALL_MEDIA,
    aux_streams_from_dir,
    discover_sources,
    project_output_paths,
)

# ══════════════════════════════════════════════════════════════════════════════
# RunSpec -- the staging currency (D2)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunSpec:
    """One staged run: a source plus everything needed to process + place it.

    ``run_id`` is the ledger key + event ``run_id`` (legacy: the video filename;
    run-folder: the folder name).  ``conditions`` is the pipe-delimited tag
    string (a CSV column).  ``meta`` carries manifest-only provenance
    (date/session/notes/extra) -- NEVER a CSV column.
    """

    run_id: str
    source: Path
    pid_map: dict | None
    conditions: str
    aux_streams: list | None
    output_paths: dict
    meta: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# run.yaml parsing (Q2) -- plain-English validation, never raises
# ══════════════════════════════════════════════════════════════════════════════

_KNOWN_RUN_KEYS = {"participants", "conditions", "date", "session", "notes",
                   "extra"}
_MANIFEST_KEYS = ("date", "session", "notes", "extra")


@dataclass(frozen=True)
class RunMeta:
    """Parsed ``run.yaml`` for one run folder.

    ``error`` is a plain-English validation message (surfaced via preflight) or
    ``None``; ``unknown_keys`` are unrecognised top-level keys (preflight WARN).
    ``manifest_meta`` is the manifest-only subset (date/session/notes/extra).
    """

    pid_map: dict | None
    conditions: list
    manifest_meta: dict
    unknown_keys: list
    error: str | None


def parse_run_yaml(path) -> RunMeta:
    """Parse a per-run ``run.yaml`` (Q2), returning a :class:`RunMeta`.

    A missing file is an empty (valid) ``RunMeta``.  Malformed content yields a
    ``RunMeta`` with ``error`` set -- this function NEVER raises, so preflight can
    report the problem as a check rather than crashing.
    """
    path = Path(path)
    if not path.is_file():
        return RunMeta(None, [], {}, [], None)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        return RunMeta(None, [], {}, [], f"run.yaml is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        return RunMeta(None, [], {}, [],
                       "run.yaml must be a mapping of keys to values")
    return parse_run_mapping(raw)


def parse_run_mapping(raw: dict) -> RunMeta:
    """Validate an already-loaded run-metadata mapping (the Q2 key set).

    Shared by :func:`parse_run_yaml` (file contents) and the manual staging
    APIs (a dict from the GUI dialog).  Never raises; problems land in
    ``error``, unrecognised top-level keys in ``unknown_keys``.
    """
    unknown = sorted(k for k in raw if k not in _KNOWN_RUN_KEYS)

    pid_map = None
    parts = raw.get("participants")
    if parts is not None:
        if not isinstance(parts, dict):
            return RunMeta(None, [], {}, unknown,
                           "run.yaml 'participants' must be a mapping of "
                           "track_id -> label (e.g. {0: S70, 1: S71})")
        try:
            pid_map = {int(k): str(v) for k, v in parts.items()}
        except (ValueError, TypeError):
            return RunMeta(None, [], {}, unknown,
                           "run.yaml 'participants' keys must be integer track "
                           "ids (e.g. 0, 1)")

    conditions: list = []
    conds = raw.get("conditions")
    if conds is not None:
        if isinstance(conds, str):
            conditions = [conds]
        elif isinstance(conds, (list, tuple)):
            conditions = [str(c) for c in conds]
        else:
            return RunMeta(None, [], {}, unknown,
                           "run.yaml 'conditions' must be a tag or a list of tags")

    manifest_meta = {k: raw[k] for k in _MANIFEST_KEYS
                     if k in raw and raw[k] is not None}
    # YAML auto-parses a bare ``date: 2026-07-02`` into a ``datetime.date``;
    # stringify it so the value stays JSON-native for the manifest.
    import datetime as _dt
    if isinstance(manifest_meta.get("date"), (_dt.date, _dt.datetime)):
        manifest_meta["date"] = manifest_meta["date"].isoformat()
    return RunMeta(pid_map, conditions, manifest_meta, unknown, None)


# ══════════════════════════════════════════════════════════════════════════════
# Layout detection + run-folder inspection (Q1)
# ══════════════════════════════════════════════════════════════════════════════

LEGACY, RUN_FOLDER, AMBIGUOUS = "legacy", "run_folder", "ambiguous"


def _runs_dir(project: Path) -> Path:
    return project / "Inputs" / "Runs"


def _has_run_folders(project: Path) -> bool:
    d = _runs_dir(project)
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def _has_flat_videos(project: Path) -> bool:
    d = project / "Inputs" / "Videos"
    return d.is_dir() and any(
        p.is_file() and p.suffix.lower() in _ALL_MEDIA for p in d.iterdir())


def detect_layout(project) -> str:
    """Return ``"legacy"`` / ``"run_folder"`` / ``"ambiguous"`` (never raises).

    ``"ambiguous"`` means BOTH ``Inputs/Runs/`` and ``Inputs/Videos/`` are
    populated (a preflight FAIL, Q1).  An empty project reads as ``"legacy"``.
    """
    project = Path(project)
    has_runs = _has_run_folders(project)
    has_videos = _has_flat_videos(project)
    if has_runs and has_videos:
        return AMBIGUOUS
    if has_runs:
        return RUN_FOLDER
    return LEGACY


def _primary_videos(folder: Path) -> list:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in _ALL_MEDIA)


@dataclass(frozen=True)
class RunFolderInfo:
    """Non-raising inspection of one ``Inputs/Runs/<run_id>/`` folder.

    ``videos`` is every primary media file found (exactly one is valid);
    ``meta`` is the parsed ``run.yaml``.  Preflight drives its per-folder checks
    off these; the strict producer turns them into ``RunSpec``s (raising on any
    violation).
    """

    run_id: str
    folder: Path
    videos: list
    meta: RunMeta


def inspect_run_folders(project) -> list:
    """Inspect every ``Inputs/Runs/<run_id>/`` folder (sorted, never raises)."""
    project = Path(project)
    runs_dir = _runs_dir(project)
    if not runs_dir.is_dir():
        return []
    infos: list = []
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        infos.append(RunFolderInfo(
            run_id=folder.name,
            folder=folder,
            videos=_primary_videos(folder),
            meta=parse_run_yaml(folder / "run.yaml"),
        ))
    return infos


# ══════════════════════════════════════════════════════════════════════════════
# Planned sessions (UP5) -- run.yaml present, no video yet; fulfilled by a
# LIVE recording OR by attaching footage from a separate device (UP5r2)
# ══════════════════════════════════════════════════════════════════════════════


def is_planned(info: RunFolderInfo) -> bool:
    """UP5 rule: a run folder with a ``run.yaml`` but NO primary video is a
    PLANNED session -- awaiting footage, from a live recording or a file
    recorded on another device (UP5r2).  A videoless folder without run.yaml
    stays an error -- junk folders must not silently pass."""
    return not info.videos and (info.folder / "run.yaml").is_file()


def planned_runs(project) -> list:
    """Every planned session in *project* (list of RunFolderInfo)."""
    return [i for i in inspect_run_folders(project) if is_planned(i)]


def plan_run(project, run_id, meta=None) -> Path:
    """Create a PLANNED session folder: ``Inputs/Runs/<id>/run.yaml`` only,
    no video yet (UP5 ruling 3 + r2: fulfilled later by a live recording or
    attached footage).  Collision-safe like :func:`stage_run`; *meta* uses
    the same Q2 keys.  Returns the created folder."""
    project = Path(project)
    _validated_meta(meta)                       # plain-English errors first
    runs_dir = _runs_dir(project)
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = _sanitize_run_id(str(run_id))
    run_dir = runs_dir / base
    n = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{base}_{n}"
        n += 1
    run_dir.mkdir()
    _write_run_yaml(run_dir, dict(meta or {}))
    # An empty meta still needs the file -- run.yaml IS the planned marker.
    (run_dir / "run.yaml").touch(exist_ok=True)
    return run_dir


def attach_recording(project, run_id, recording, *, sidecar=None,
                     meta=None, mode="move") -> Path:
    """Stage footage as run *run_id*'s primary video (UP5/UP5r2).

    Fills a PLANNED folder, or creates the folder for an ad-hoc session.  The
    footage lands as ``<run_id><suffix>`` -- ``mode="move"`` (default; right
    for a live capture's temp file) or ``mode="copy"`` (right for footage
    recorded on a separate device: the original stays untouched, matching the
    wizard's copy default).  *meta* (Q2 keys) merges over any existing
    run.yaml values; the capture-timestamps *sidecar* moves alongside
    (non-media, so discovery never sees it).  Returns the staged video path.
    Raises plain-English ``ValueError`` on missing footage or a run that
    already holds a video.
    """
    project = Path(project)
    recording = Path(recording)
    if not recording.is_file():
        raise ValueError(f"recording not found: {recording}")
    rid = _sanitize_run_id(str(run_id))
    run_dir = _runs_dir(project) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _primary_videos(run_dir)
    if existing:
        raise ValueError(
            f"run '{rid}' already has a video ({existing[0].name}) -- "
            "record into a new or planned session instead")
    if meta:
        _validated_meta(meta)
        current = parse_run_yaml(run_dir / "run.yaml")
        data: dict = {}
        if current.pid_map:
            data["participants"] = dict(current.pid_map)
        if current.conditions:
            data["conditions"] = list(current.conditions)
        data.update({k: v for k, v in dict(current.manifest_meta).items()
                     if v is not None})
        data.update({k: v for k, v in dict(meta).items() if v is not None})
        _write_run_yaml(run_dir, data)
        (run_dir / "run.yaml").touch(exist_ok=True)
    dest = run_dir / (rid + recording.suffix.lower())
    if mode == "copy":
        shutil.copy2(str(recording), str(dest))
    else:
        shutil.move(str(recording), str(dest))
    if sidecar is not None and Path(sidecar).is_file():
        # Rename to the run's identity -- the temp capture name is noise.
        shutil.move(str(sidecar),
                    str(run_dir / f"{rid}_capture_timestamps.csv"))
    return dest


# ══════════════════════════════════════════════════════════════════════════════
# Output-path placement
# ══════════════════════════════════════════════════════════════════════════════


def _out_root(project: Path, project_cfg) -> Path:
    if project_cfg and getattr(project_cfg, "output", None):
        return project_cfg.output.resolve_root(project)
    return project / "Outputs"


def run_folder_output_paths(out_root: Path, run_id: str) -> dict:
    """Per-run mirrored output paths under ``Outputs/Runs/<run_id>/`` (Q3).

    Global aggregates (``Global_*`` / ``By Condition``) stay under
    ``Outputs/CSV Files`` -- that placement lives in the batch post-processing,
    not here.
    """
    run_dir = out_root / "Runs" / run_id
    return {
        'save':    str(run_dir / f"{run_id}_Video_Output.mp4"),
        'log':     str(run_dir / f"{run_id}_Events.csv"),
        'summary': str(run_dir / f"{run_id}_summary.csv"),
        'heatmap': str(run_dir / f"{run_id}_Heatmap"),
    }


def run_display_name(spec: RunSpec) -> str:
    """The stem used for this run's ``video_name`` column + manifest filename.

    Derived from the (single-sourced) summary output path so both producers
    agree: legacy ``a_summary.csv`` -> ``a`` (== the source stem, byte-compat);
    run-folder ``<run_id>_summary.csv`` -> ``run_id``.
    """
    name = Path(spec.output_paths["summary"]).name
    suffix = "_summary.csv"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return Path(spec.source).stem


# ══════════════════════════════════════════════════════════════════════════════
# Producers
# ══════════════════════════════════════════════════════════════════════════════


def _legacy_run_specs(project: Path, project_cfg, *, aux_streams, pid_maps):
    """Legacy flat-video producer -- byte-compatible with the pre-SP3 loop.

    ``run_id`` is the video filename (the pre-SP3 ledger key); pid maps /
    conditions are looked up by that filename; output paths + aux are exactly
    what the flat loop always computed.
    """
    specs: list = []
    for source in discover_sources(project):
        pid_map = pid_maps.get(source.name) if pid_maps else None
        tags = (project_cfg.conditions.get(source.name, [])
                if project_cfg and project_cfg.conditions else [])
        conditions = "|".join(tags) if tags else ""
        specs.append(RunSpec(
            run_id=source.name,
            source=source,
            pid_map=pid_map,
            conditions=conditions,
            aux_streams=aux_streams or None,
            output_paths=project_output_paths(project, source, project_cfg),
            meta={},
        ))
    return specs


def _run_folder_specs(project: Path, project_cfg, *, pid_maps):
    """Run-folder producer (Q1/Q2) -- raises ``ValueError`` on any folder that
    is not exactly-one-video with a valid ``run.yaml`` (preflight reports these
    granularly first; the run path fails loudly)."""
    out_root = _out_root(project, project_cfg)
    specs: list = []
    for info in inspect_run_folders(project):
        rid = info.run_id
        if info.meta.error:
            raise ValueError(f"Run folder '{rid}': {info.meta.error}")
        if not info.videos:
            if is_planned(info):
                continue    # UP5: planned live session -- awaiting recording
            raise ValueError(
                f"Run folder '{rid}' has no video -- expected exactly one "
                f"primary video in Inputs/Runs/{rid}/")
        if len(info.videos) > 1:
            names = ", ".join(v.name for v in info.videos)
            raise ValueError(
                f"Run folder '{rid}' has {len(info.videos)} videos ({names}); "
                f"expected exactly one primary video")

        # Metadata precedence: run.yaml > project.yaml > participant_ids.csv.
        pid_map = info.meta.pid_map
        if pid_map is None and project_cfg and project_cfg.participants:
            pid_map = project_cfg.participants.get(rid)
        if pid_map is None and pid_maps:
            pid_map = pid_maps.get(rid)

        tags = info.meta.conditions
        if not tags and project_cfg and project_cfg.conditions:
            tags = project_cfg.conditions.get(rid, [])
        conditions = "|".join(tags) if tags else ""

        aux = aux_streams_from_dir(info.folder / "aux")
        specs.append(RunSpec(
            run_id=rid,
            source=info.videos[0],
            pid_map=pid_map,
            conditions=conditions,
            aux_streams=aux or None,
            output_paths=run_folder_output_paths(out_root, rid),
            meta=dict(info.meta.manifest_meta),
        ))
    return specs


def discover_run_specs(project, project_cfg=None, *, layout=None,
                       aux_streams=None, pid_maps=None):
    """Produce the ``list[RunSpec]`` for *project* (dispatch by layout).

    ``layout`` is detected when not supplied; ``"ambiguous"`` raises a
    plain-English ``ValueError``.  ``aux_streams`` (batch-level, legacy) and
    ``pid_maps`` (the resolved study-wide map) are passed through from the
    caller so discovery is not repeated.
    """
    project = Path(project)
    layout = layout or detect_layout(project)
    if layout == AMBIGUOUS:
        raise ValueError(
            "Project has BOTH Inputs/Runs/ and Inputs/Videos/ populated -- the "
            "layout is ambiguous. Use one: run folders (Inputs/Runs/<run_id>/) "
            "OR flat videos (Inputs/Videos/).")
    if layout == RUN_FOLDER:
        return _run_folder_specs(project, project_cfg, pid_maps=pid_maps)
    return _legacy_run_specs(project, project_cfg, aux_streams=aux_streams,
                             pid_maps=pid_maps)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-run metadata edits (G-DEFER-1): the single WRITE path for participants /
# conditions, layout-aware (D11 -- the GUI is a thin caller).
# ══════════════════════════════════════════════════════════════════════════════

_UNSET = object()


def update_run_metadata(project, run_id, *, participants=_UNSET,
                        conditions=_UNSET) -> None:
    """Set a run's participants / conditions before running (SP3.1 G-DEFER-1).

    Layout-aware, reusing the existing parse/write paths and precedence rules:

    - **run-folder** projects: rewrites ``Inputs/Runs/<run_id>/run.yaml``, merging
      the new values over the existing keys (the manifest-only ``date`` /
      ``session`` / ``notes`` / ``extra`` are preserved; an emptied run.yaml is
      removed so a bare folder just works).
    - **legacy** projects: updates the ``participants`` / ``conditions`` entries in
      ``project.yaml`` keyed by ``run_id`` (== the video filename), via
      :func:`load_project_config` / :func:`save_project_config`.

    ``participants`` is a ``{track_id: label}`` mapping (or ``None`` to clear);
    ``conditions`` is a tag string / list of tags (or ``None`` to clear).  A field
    left at its default is untouched.  Validation problems raise a plain-English
    ``ValueError`` (the shared Q2 validator); an ambiguous layout is refused.
    """
    project = Path(project)
    set_p = participants is not _UNSET
    set_c = conditions is not _UNSET
    if not set_p and not set_c:
        return

    # Validate through the shared Q2 validator (plain-English errors).
    check: dict = {}
    if set_p and participants is not None:
        check["participants"] = participants
    if set_c and conditions is not None:
        check["conditions"] = conditions
    parsed = parse_run_mapping(check)
    if parsed.error:
        raise ValueError(parsed.error)
    pid_map = parsed.pid_map          # {int: str} or None
    cond_list = parsed.conditions     # list[str]

    layout = detect_layout(project)
    if layout == AMBIGUOUS:
        raise ValueError(
            "Project has BOTH Inputs/Runs/ and Inputs/Videos/ populated -- the "
            "layout is ambiguous; cannot edit run metadata.")

    if layout == RUN_FOLDER:
        folder = _runs_dir(project) / run_id
        if not folder.is_dir():
            raise ValueError(f"run folder not found: Inputs/Runs/{run_id}/")
        rp = folder / "run.yaml"
        raw: dict = {}
        if rp.is_file():
            loaded = yaml.safe_load(rp.read_text())
            if isinstance(loaded, dict):
                raw = loaded
        if set_p:
            if pid_map:
                raw["participants"] = {int(k): str(v) for k, v in pid_map.items()}
            else:
                raw.pop("participants", None)
        if set_c:
            if cond_list:
                raw["conditions"] = list(cond_list)
            else:
                raw.pop("conditions", None)
        if raw:
            with open(rp, "w") as fh:
                yaml.dump(raw, fh, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)
        elif rp.is_file():
            rp.unlink()
        return

    # Legacy flat layout: project.yaml entries keyed by the video filename.
    from mindsight.pipeline_config import ProjectConfig
    from mindsight.project.runner import (
        load_project_config,
        save_project_config,
    )
    cfg = load_project_config(project) or ProjectConfig()
    conditions_map = dict(cfg.conditions or {})
    participants_map = dict(cfg.participants or {})
    if set_p:
        if pid_map:
            participants_map[run_id] = {int(k): str(v)
                                        for k, v in pid_map.items()}
        else:
            participants_map.pop(run_id, None)
    if set_c:
        if cond_list:
            conditions_map[run_id] = list(cond_list)
        else:
            conditions_map.pop(run_id, None)
    save_project_config(project, ProjectConfig(
        pipeline_path=cfg.pipeline_path,
        conditions=conditions_map,
        participants=participants_map,
        output=cfg.output,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Manual staging (Q7): stage a single video into a project / run it right now
# ══════════════════════════════════════════════════════════════════════════════

# Filesystem-unsafe characters in a run_id -> underscore (Q1: folder names are
# sanitized).  Same class the global-CSV condition filenames use.
_UNSAFE_RUN_ID = re.compile(r'[/\\:*?"<>|]')


def _sanitize_run_id(name: str) -> str:
    return _UNSAFE_RUN_ID.sub("_", name).strip() or "run"


def _validated_meta(meta: dict | None) -> RunMeta:
    """Validate a manual-staging metadata dict (strict: the APIs raise)."""
    parsed = parse_run_mapping(dict(meta or {}))
    if parsed.error:
        raise ValueError(parsed.error)
    if parsed.unknown_keys:
        raise ValueError(
            f"unknown run metadata key(s): {', '.join(parsed.unknown_keys)} "
            f"-- use only: {', '.join(sorted(_KNOWN_RUN_KEYS))}")
    return parsed


def _write_run_yaml(folder: Path, meta: dict) -> None:
    data = {k: meta[k] for k in
            ("participants", "conditions", *_MANIFEST_KEYS)
            if k in meta and meta[k] is not None}
    if not data:
        return                        # a bare folder just works (Q1)
    with open(folder / "run.yaml", "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _unique_run_dir(runs_root: Path, run_id: str) -> Path:
    """Collision-safe run-folder path: ``<run_id>``, else ``<run_id>_2``, ..."""
    candidate = runs_root / run_id
    n = 2
    while candidate.exists():
        candidate = runs_root / f"{run_id}_{n}"
        n += 1
    return candidate


def stage_run(project, video, meta=None, *, run_id=None, mode="copy") -> RunSpec:
    """Stage *video* into *project* as a new ``Inputs/Runs/<run_id>/`` folder (Q7).

    Creates the run folder (collision-safe: an existing ``<run_id>`` gets a
    ``_2`` / ``_3`` ... suffix), places the video by ``mode`` -- ``"copy"``
    (default: self-contained, portable project) or ``"move"`` (no duplicate
    storage; falls back to copy+delete across devices) -- and writes ``run.yaml``
    when *meta* carries any of the Q2 keys.  Returns the staged :class:`RunSpec`.

    Raises ``ValueError`` (plain-English) when the video is missing, *meta* is
    invalid, or the project uses the flat legacy layout (staging a run folder
    would make the layout ambiguous, Q1).
    """
    project = Path(project).resolve()
    video = Path(video)
    if not video.is_file():
        raise ValueError(f"video not found: {video}")
    if mode not in ("copy", "move"):
        raise ValueError(f"mode must be 'copy' or 'move', not {mode!r}")
    if _has_flat_videos(project):
        raise ValueError(
            f"project {project.name} uses the flat Inputs/Videos/ layout -- "
            "staging a run folder would make the layout ambiguous. Move the "
            "existing videos into run folders first, or stage into a "
            "run-folder project.")
    parsed = _validated_meta(meta)

    run_dir = _unique_run_dir(_runs_dir(project),
                              _sanitize_run_id(run_id or video.stem))
    run_dir.mkdir(parents=True)
    dest = run_dir / video.name
    if mode == "move":
        # shutil.move rename-or-copies: same-device is an os.rename; across
        # devices it falls back to copy2 + unlink automatically.
        shutil.move(str(video), str(dest))
    else:
        shutil.copy2(str(video), str(dest))
    _write_run_yaml(run_dir, dict(meta or {}))

    project_cfg = None
    try:
        from mindsight.project.runner import load_project_config
        project_cfg = load_project_config(project)
    except Exception:
        pass
    return RunSpec(
        run_id=run_dir.name,
        source=dest,
        pid_map=parsed.pid_map,
        conditions="|".join(parsed.conditions) if parsed.conditions else "",
        aux_streams=None,
        output_paths=run_folder_output_paths(_out_root(project, project_cfg),
                                             run_dir.name),
        meta=dict(parsed.manifest_meta),
    )


def _run_timestamp() -> str:
    """Local wall-clock stamp ``YYYYMMDD-HHMMSS`` for one-off run ids.

    Factored out so tests can pin it -- successive captures a real second apart
    never collide, but two calls in the same test tick would.
    """
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def camera_run_spec(index: int, output_dir, meta=None) -> RunSpec:
    """A one-off :class:`RunSpec` for a live-camera quick run (UP1 D2).

    No staging, no ledger: the capture writes the same project-shaped CSVs +
    annotated recording as :func:`single_run_spec`, flat into *output_dir*.  The
    ``run_id`` carries a wall-clock timestamp (``camera{index}_YYYYMMDD-HHMMSS``)
    so successive captures never overwrite each other's outputs.

    ``source`` is the camera index as a string; :func:`open_video_source
    <mindsight.io.sources.open_video_source>` normalizes it to the ``int`` cv2
    needs at open time.  *output_dir* is REQUIRED (the GUI always passes one --
    a camera has no source folder to default beside); a missing one raises a
    plain-English ``ValueError`` for the caller to surface.
    """
    if not output_dir:
        raise ValueError(
            "choose an output directory -- a live camera capture has no source "
            "folder to write beside, so one must be given")
    parsed = _validated_meta(meta)              # UP5: optional session details
    run_id = _sanitize_run_id(f"camera{index}_{_run_timestamp()}")
    out_dir = Path(output_dir)
    return RunSpec(
        run_id=run_id,
        source=str(index),
        pid_map=parsed.pid_map or {},
        conditions="|".join(parsed.conditions) if parsed.conditions else "",
        aux_streams=None,
        output_paths={
            'save':    str(out_dir / f"{run_id}_Video_Output.mp4"),
            'log':     str(out_dir / f"{run_id}_Events.csv"),
            'summary': str(out_dir / f"{run_id}_summary.csv"),
            'heatmap': str(out_dir / f"{run_id}_Heatmap"),
        },
        meta=dict(parsed.manifest_meta),
    )


def single_run_spec(video, meta=None, output_dir=None, project=None) -> RunSpec:
    """A one-off :class:`RunSpec` for the GUI "Run now" path (Q7).

    No staging, no ledger: the video runs where it is, with the entered
    participants/conditions producing the same project-shaped CSVs + manifest,
    written flat into *output_dir*.

    Output location (B1 F1): an explicit *output_dir* always wins.  When it is
    omitted, the outputs default to the open *project*'s Outputs root -- NOT a
    CWD-relative ``Outputs/`` (a GUI launched from Finder/installer has an
    arbitrary CWD, so those files vanish from the user's perspective).  With
    neither an *output_dir* nor a *project*, there is no sane default, so a
    plain-English ``ValueError`` is raised for the caller to surface.

    Raises ``ValueError`` (plain-English) on a missing video, invalid *meta*, or
    no output-directory context.
    """
    video = Path(video)
    if not video.is_file():
        raise ValueError(f"video not found: {video}")
    parsed = _validated_meta(meta)
    if output_dir:
        out_dir = Path(output_dir)
    elif project is not None:
        proj = Path(project)
        project_cfg = None
        try:
            from mindsight.project.runner import load_project_config
            project_cfg = load_project_config(proj)
        except Exception:
            project_cfg = None
        out_dir = _out_root(proj, project_cfg)
    else:
        raise ValueError(
            "choose an output directory -- no output folder was given and no "
            "project is open to default to")
    run_id = _sanitize_run_id(video.stem)
    return RunSpec(
        run_id=run_id,
        source=video,
        pid_map=parsed.pid_map,
        conditions="|".join(parsed.conditions) if parsed.conditions else "",
        aux_streams=None,
        output_paths={
            'save':    str(out_dir / f"{run_id}_Video_Output.mp4"),
            'log':     str(out_dir / f"{run_id}_Events.csv"),
            'summary': str(out_dir / f"{run_id}_summary.csv"),
            'heatmap': str(out_dir / f"{run_id}_Heatmap"),
        },
        meta=dict(parsed.manifest_meta),
    )
