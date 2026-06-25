"""
project/project.py -- The ``Project`` facade (SP3.1 D3).

A single object the GUI and CLI open a project through, so nothing frontend-side
reaches into ``runner`` / ``ledger`` internals:

    Project.open(path)                    -> Project   (validate + load config)
    project.runs()                        -> list[Path]  (discovered sources)
    project.run(ns, *, resume, cancel,
                project_cfg)              -> Iterator[ProjectEvent]
    project.status()                      -> list[VideoStatus]  (read-only ledger)
    project.preflight(ns=None)            -> PreflightReport     (Batch D)

``open`` validates the project structure and loads ``project.yaml`` but builds NO
models (that happens lazily inside ``run``).  ``run`` is a thin wrapper over
``iter_project_runs`` -- the one project-run implementation.  ``preflight``
delegates to ``mindsight.project.preflight.run_preflight`` (SP3.1 D4).

``runs()`` returns discovered source paths for now; SP3.1 Batch E replaces this
with a ``RunSpec`` list once ``staging.py`` exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mindsight.project.ledger import Ledger
from mindsight.project.runner import (
    discover_participant_ids,
    iter_project_runs,
    load_project_config,
    validate_project,
)


@dataclass(frozen=True)
class VideoStatus:
    """Read-only ledger view of one run's state (D3).

    ``status`` is ``done`` / ``in_progress`` / ``error`` (or ``None`` for a bare
    record); ``config_hash`` / ``video_hash`` are the resume identity; ``finished``
    is the terminal ISO timestamp; ``error`` the exception text on failure.
    """
    run_id: str
    status: str | None
    config_hash: str | None
    video_hash: str | None
    finished: str | None
    error: str | None


class Project:
    """Facade over a MindSight project directory (validate + config + runs)."""

    def __init__(self, path: Path, config):
        self._path = path
        self._config = config

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def open(cls, path) -> "Project":
        """Validate the project structure and load ``project.yaml`` (no models).

        Raises ``FileNotFoundError`` / ``ValueError`` (via ``validate_project``)
        when the directory is missing or lacks ``Inputs/Videos/``.
        """
        project = Path(path).resolve()
        config = load_project_config(project)
        # validate_project resolves the path + ensures the output dirs exist.
        project = validate_project(project, config)
        return cls(project, config)

    # ── read-only accessors ─────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self):
        return self._config

    def runs(self) -> list:
        """Discovered runs as ``list[RunSpec]`` (SP3.1 D2/D3).

        Layout-aware: legacy flat videos or per-run ``Inputs/Runs/`` folders.
        Resolves the study-wide participant map + batch aux streams the same way
        the runner does, so the returned specs match what a run would process.
        """
        from mindsight.project.runner import discover_aux_streams
        from mindsight.project.staging import (
            RUN_FOLDER,
            detect_layout,
            discover_run_specs,
        )
        layout = detect_layout(self._path)
        if layout == RUN_FOLDER:
            pid_maps = discover_participant_ids(self._path)
            aux = None
        elif self._config and self._config.participants:
            pid_maps, aux = self._config.participants, discover_aux_streams(self._path)
        else:
            pid_maps = discover_participant_ids(self._path)
            aux = discover_aux_streams(self._path)
        return discover_run_specs(self._path, self._config, layout=layout,
                                  aux_streams=aux, pid_maps=pid_maps)

    def _out_root(self) -> Path:
        if self._config and self._config.output:
            return self._config.output.resolve_root(self._path)
        return self._path / "Outputs"

    def status(self) -> list[VideoStatus]:
        """Read-only ledger view: one :class:`VideoStatus` per recorded run."""
        ledger = Ledger.load(self._out_root())
        out: list[VideoStatus] = []
        for run_id, rec in sorted(ledger.videos().items()):
            out.append(VideoStatus(
                run_id=run_id,
                status=rec.get("status"),
                config_hash=rec.get("config_hash"),
                video_hash=rec.get("video_hash"),
                finished=rec.get("finished"),
                error=rec.get("error"),
            ))
        return out

    # ── run / preflight ─────────────────────────────────────────────────────

    def run(self, ns, *, resume=True, cancel=None, project_cfg=None):
        """Run the batch, yielding a :mod:`ProjectEvent <mindsight.project.events>` stream.

        Thin wrapper over ``iter_project_runs``.  ``project_cfg`` overrides the
        loaded config (the GUI's possibly-unsaved edits); when ``None`` the
        config loaded at ``open`` time is used.
        """
        cfg = project_cfg if project_cfg is not None else self._config
        return iter_project_runs(self._path, ns, project_cfg=cfg,
                                 cancel=cancel, resume=resume)

    def decisions(self, ns, *, resume=True, project_cfg=None) -> dict:
        """Preview the resume plan per run: ``run_id -> "skip"|"redo"|"redo_archive"``.

        Mirrors what :func:`iter_project_runs` will decide (Q6 runs-table preview,
        D11 -- computed in the project layer, never the GUI).  ``resume=False``
        ("Re-run all") short-circuits to ``redo`` for every run without touching
        the models or weights.  Best-effort: any failure resolving the batch
        identity leaves a run out of the mapping rather than raising.
        """
        cfg = project_cfg if project_cfg is not None else self._config
        specs = self.runs()
        if not resume:
            return {s.run_id: "redo" for s in specs}

        import copy

        from mindsight.config import PipelineConfig
        from mindsight.config_compat import load_pipeline
        from mindsight.outputs import provenance
        from mindsight.project.ledger import Ledger, compute_video_hash

        # Merge the project pipeline YAML into a COPY of ns exactly as the runner
        # does (T7 ns pass-through) so the config identity matches the real run.
        ns_copy = copy.deepcopy(ns)
        if cfg and cfg.pipeline_path:
            pipeline_yaml = self._path / cfg.pipeline_path
        else:
            pipeline_yaml = self._path / "Pipeline" / "pipeline.yaml"
        try:
            if pipeline_yaml.exists():
                load_pipeline(pipeline_yaml, ns_copy)
            config = PipelineConfig.from_namespace(ns_copy)
            weights = provenance.collect_weights(ns_copy)
            config_hash = provenance.run_identity(
                ns_copy, config=config, weights=weights)
        except Exception:
            return {}

        ledger = Ledger.load(self._out_root())
        out: dict = {}
        for spec in specs:
            try:
                video_hash = compute_video_hash(
                    spec.source, pid_map=spec.pid_map,
                    conditions=spec.conditions, aux_streams=spec.aux_streams)
                out[spec.run_id] = ledger.decide(
                    spec.run_id, (config_hash, video_hash))
            except Exception:
                continue
        return out

    def invalidate(self, run_id: str) -> bool:
        """Drop *run_id* from the ledger so its next run reprocesses (Q6 per-row
        "Re-run this run").  Returns ``True`` when a record was dropped."""
        ledger = Ledger.load(self._out_root())
        return ledger.invalidate(run_id)

    def preflight(self, ns=None):
        """Structured readiness checklist (SP3.1 D4).

        Delegates to :func:`mindsight.project.preflight.run_preflight` with the
        already-loaded config.  Read-only: builds no models, runs no videos.
        """
        from mindsight.project.preflight import run_preflight
        return run_preflight(self._path, self._config, ns=ns)
