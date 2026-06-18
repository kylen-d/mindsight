"""
project/events.py -- Event stream for the unified project-run iterator (SP3.1 D1).

``iter_project_runs`` (project/runner.py) is a single event-streaming
implementation of the project batch loop, consumed by BOTH the CLI
(``cli.main --project``) and the GUI (``GUI/workers.ProjectWorker``).  It yields
one of the frozen dataclasses below per orchestration step; each consumer maps
the events onto its own surface (the CLI prints + drives the cv2 display loop,
the GUI translates them into its progress/frame/log queues).

The event union::

    BatchStarted(total, out_root)          -- setup done; N sources; ledger ready
    VideoStarted(index, total, run_id, source)
    VideoFrame(run_id, result)             -- one FrameResult (display-free here)
    VideoSkipped(run_id, reason)           -- ledger said skip (done, unchanged)
    VideoArchived(run_id, dest)            -- superseded outputs archived (redo)
    VideoDone(run_id, manifest_path)       -- video finished + manifest written
    VideoError(run_id, error)              -- video raised; manifest recorded error
    BatchDone(out_root)                    -- global CSVs written; batch complete

The dataclasses are pure data (no behavior); the batch narration lines and the
ledger/manifest/global-CSV machinery live in ``iter_project_runs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class BatchStarted:
    """Setup finished: ``total`` sources discovered, outputs under ``out_root``."""
    total: int
    out_root: Path


@dataclass(frozen=True)
class VideoStarted:
    """A source is about to be processed (1-based ``index`` of ``total``)."""
    index: int
    total: int
    run_id: str
    source: Path


@dataclass(frozen=True)
class VideoFrame:
    """One processed frame's :class:`~mindsight.pipeline.FrameResult` for *run_id*.

    The iterator is display-free (D1): consumers decide whether to imshow the
    frame (CLI) or push ``result.annotated.copy()`` to a queue (GUI).
    """
    run_id: str
    result: object


@dataclass(frozen=True)
class VideoSkipped:
    """The ledger decided to skip *run_id* (done, config + source unchanged)."""
    run_id: str
    reason: str


@dataclass(frozen=True)
class VideoArchived:
    """Superseded outputs for *run_id* were archived before reprocessing."""
    run_id: str
    dest: object


@dataclass(frozen=True)
class VideoDone:
    """*run_id* finished; its per-video manifest is at ``manifest_path``."""
    run_id: str
    manifest_path: str


@dataclass(frozen=True)
class VideoError:
    """*run_id* raised while processing; ``error`` is the exception text."""
    run_id: str
    error: str


@dataclass(frozen=True)
class BatchDone:
    """The batch completed: per-video runs done and global CSVs generated."""
    out_root: Path


ProjectEvent = Union[
    BatchStarted,
    VideoStarted,
    VideoFrame,
    VideoSkipped,
    VideoArchived,
    VideoDone,
    VideoError,
    BatchDone,
]
