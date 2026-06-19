"""
project/ledger.py -- Crash-resumable project batch ledger (SP2.1 D9).

A project batch writes ``Outputs/_run/ledger.json`` recording, per video, whether
it is ``in_progress`` / ``done`` / ``error`` together with the composite
``config_hash`` (``provenance.run_identity``) and a per-video ``video_hash``.  On
a later run the batch consults the ledger:

- ``done`` + both hashes match      -> SKIP (one console line);
- ``done`` + either hash mismatched -> archive the old outputs + manifest into
  ``Outputs/_run/superseded/<UTC>_<stem>/`` then reprocess (``redo_archive``);
- ``in_progress`` (a kill -9 mid-run) / ``error`` / absent -> reprocess in place,
  no archive (``redo``).

Every transition rewrites the ledger ATOMICALLY (tmp file + ``os.replace``) so a
kill -9 landing mid-write never leaves a torn ledger (SP2.1 lesson 12).  Resume is
ON by default; the CLI ``--no-resume`` bypasses ``decide`` entirely (still
rewrites the ledger, never archives -- the "I know what I'm doing" path).

Pure stdlib.  Timestamps are ISO-8601 with a UTC offset.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

LEDGER_VERSION = 1


def _utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with offset."""
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    """Compact UTC stamp for superseded archive dirs (yyyymmddTHHMMSSZ)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compute_video_hash(source, *, pid_map=None, conditions="",
                       aux_streams=None) -> str:
    """Per-video identity hash (SP2.1 D6).

    sha256 over the source file's ``(size, mtime_ns)``, the participant-id map,
    the condition string, and the sorted auxiliary-stream tuples
    ``(source, video_type, stream_label, participants)``.  Two runs skip only
    when this AND the batch ``config_hash`` both match, so re-timestamping /
    swapping the source video, or changing conditions / pid_map / aux streams,
    forces a reprocess.
    """
    p = Path(str(source))
    st = p.stat()
    aux = sorted(
        (
            str(a.source),
            a.video_type.value if hasattr(a.video_type, "value")
            else str(a.video_type),
            str(a.stream_label),
            [str(x) for x in a.participants],
        )
        for a in (aux_streams or [])
    )
    payload = {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "pid_map": {str(k): str(v) for k, v in (pid_map or {}).items()},
        "conditions": conditions or "",
        "aux": aux,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Ledger:
    """The per-project resume ledger backed by ``Outputs/_run/ledger.json``."""

    def __init__(self, path: Path, data: dict):
        self._path = Path(path)
        self._data = data

    # ── load / persist ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, out_root) -> "Ledger":
        """Load (or initialise) the ledger under ``<out_root>/_run/``.

        A missing or unreadable ledger.json yields a fresh empty ledger; the
        file is not written until the first transition.
        """
        path = Path(out_root) / "_run" / "ledger.json"
        data = {"ledger_version": LEDGER_VERSION, "videos": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict) and "videos" in loaded:
                    data = loaded
            except (json.JSONDecodeError, OSError):
                # Torn / corrupt ledger -> start clean (correctness beats a
                # confusing half-parse; a fresh ledger reprocesses everything).
                pass
        return cls(path, data)

    def _write(self) -> None:
        """Atomically rewrite ledger.json (tmp + os.replace)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.tmp.{os.getpid()}")
        with open(tmp, "w") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    # ── record helpers ──────────────────────────────────────────────────────

    def record(self, video) -> dict | None:
        """Return the stored record for *video*, or ``None`` if absent."""
        return self._data["videos"].get(video)

    def videos(self) -> dict:
        """Return a shallow copy of the per-video record mapping (read-only view).

        Powers ``Project.status()`` without exposing the mutable backing dict.
        """
        return dict(self._data.get("videos", {}))

    def _record(self, video) -> dict:
        return self._data["videos"].setdefault(video, {})

    # ── transitions (each rewrites atomically) ──────────────────────────────

    def mark_started(self, video, hashes, output_paths) -> None:
        """Record *video* as ``in_progress`` with its hashes + output paths."""
        config_hash, video_hash = hashes
        rec = self._record(video)
        rec.update({
            "status": "in_progress",
            "config_hash": config_hash,
            "video_hash": video_hash,
            "started": _utcnow_iso(),
            "finished": None,
            "error": None,
            "output_paths": {k: str(v) for k, v in dict(output_paths).items()},
            "manifest": None,
        })
        self._write()

    def mark_done(self, video, manifest_path=None) -> None:
        """Record *video* as ``done`` and stamp its manifest path."""
        rec = self._record(video)
        rec["status"] = "done"
        rec["finished"] = _utcnow_iso()
        rec["error"] = None
        if manifest_path is not None:
            rec["manifest"] = str(manifest_path)
        self._write()

    def mark_error(self, video, exc) -> None:
        """Record *video* as ``error`` with the exception text."""
        rec = self._record(video)
        rec["status"] = "error"
        rec["finished"] = _utcnow_iso()
        rec["error"] = str(exc)
        self._write()

    # ── resume decision ─────────────────────────────────────────────────────

    def decide(self, video, hashes) -> str:
        """Return ``"skip"`` | ``"redo"`` | ``"redo_archive"`` for *video*.

        - absent / ``in_progress`` (crash) / ``error`` -> ``"redo"`` (in place);
        - ``done`` + both hashes match                  -> ``"skip"``;
        - ``done`` + either hash differs                -> ``"redo_archive"``.
        """
        config_hash, video_hash = hashes
        rec = self.record(video)
        if rec is None:
            return "redo"
        if rec.get("status") != "done":
            return "redo"
        if (rec.get("config_hash") == config_hash
                and rec.get("video_hash") == video_hash):
            return "skip"
        return "redo_archive"

    def invalidate(self, video) -> bool:
        """Drop *video*'s record so a later ``decide`` returns ``redo`` (D10).

        Powers the per-video "Re-run this run" action (Q6): the next batch
        reprocesses the video IN PLACE (``decide`` sees no record -> ``redo``, no
        archive -- there is no stored done-state to supersede).  Returns ``True``
        when a record was dropped, ``False`` when *video* was already absent.
        """
        if video in self._data["videos"]:
            del self._data["videos"][video]
            self._write()
            return True
        return False

    # ── supersede archiving ─────────────────────────────────────────────────

    def archive(self, video) -> Path | None:
        """Move *video*'s existing outputs + manifest into a superseded dir.

        Files (and the heatmap directory) named in the stored ``output_paths``
        plus the stored ``manifest`` are moved into
        ``<out_root>/_run/superseded/<UTC>_<stem>/``.  Returns the archive dir,
        or ``None`` when there is nothing to move.
        """
        rec = self.record(video)
        if rec is None:
            return None
        stem = Path(video).stem
        dest = (self._path.parent / "superseded" / f"{_utc_stamp()}_{stem}")

        moved = False
        candidates = list((rec.get("output_paths") or {}).values())
        manifest = rec.get("manifest")
        if manifest:
            candidates.append(manifest)
        for raw in candidates:
            if not raw:
                continue
            src = Path(str(raw))
            if not src.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / src.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(src), str(target))
            moved = True
        return dest if moved else None
