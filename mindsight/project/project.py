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
    discover_sources,
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

    def runs(self) -> list[Path]:
        """Discovered source media (Batch E upgrades this to ``list[RunSpec]``)."""
        return discover_sources(self._path)

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

    def preflight(self, ns=None):
        """Structured readiness checklist (SP3.1 D4).

        Delegates to :func:`mindsight.project.preflight.run_preflight` with the
        already-loaded config.  Read-only: builds no models, runs no videos.
        """
        from mindsight.project.preflight import run_preflight
        return run_preflight(self._path, self._config, ns=ns)
