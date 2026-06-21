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
