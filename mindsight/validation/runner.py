"""
validation/runner.py — Prepare, place, and persist validation runs.

The workbench takes the Inference Tuning tab's CURRENT namespace
(`_build_namespace()` — the seam every GUI run flows through), rewrites
only the run-target fields here, and drives the ordinary GazeWorker
with it.  This module stays GUI-free: pure namespace surgery, run-dir
allocation, and score persistence, all unit-testable.

Runs land in ``<validation root>/.runs/<set-slug>/run-NNN/`` with the
run's streams, ``score.json``, and a ``settings.json`` namespace
snapshot (the phase-4 history/compare diff source).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .scoring import score_run
from .store import ValidationSet, ValidationSetError, _slug

_RUN_RE = re.compile(r"^run-(\d+)$")


def runs_root(validation_dir: Path) -> Path:
    return Path(validation_dir) / ".runs"


def allocate_run_dir(validation_dir: Path, set_name: str) -> Path:
    """Create and return the next sequential run dir for *set_name*."""
    base = runs_root(validation_dir) / _slug(set_name)
    base.mkdir(parents=True, exist_ok=True)
    seq = 0
    for p in base.iterdir():
        m = _RUN_RE.match(p.name)
        if m:
            seq = max(seq, int(m.group(1)))
    run_dir = base / f"run-{seq + 1:03d}"
    run_dir.mkdir()
    return run_dir


def list_run_dirs(validation_dir: Path, set_name: str) -> list[Path]:
    """Existing run dirs for *set_name*, oldest first."""
    base = runs_root(validation_dir) / _slug(set_name)
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if _RUN_RE.match(p.name)),
                  key=lambda p: p.name)


def latest_score(validation_dir: Path, set_name: str) -> dict | None:
    """Most recent persisted score for *set_name* (None when no run has
    scored yet).  Unreadable score files are skipped."""
    for run_dir in reversed(list_run_dirs(validation_dir, set_name)):
        path = run_dir / "score.json"
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def run_history(validation_dir: Path, set_name: str) -> list[dict]:
    """All scored runs for *set_name*, oldest first:
    ``[{run, score, settings, changed}]`` where ``changed`` is the
    settings diff vs the PREVIOUS scored run (empty for the first).
    Runs without a readable score are skipped; a missing settings
    snapshot yields an empty diff."""
    history: list[dict] = []
    prev_settings = None
    for run_dir in list_run_dirs(validation_dir, set_name):
        try:
            score = json.loads((run_dir / "score.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        try:
            settings = json.loads((run_dir / "settings.json").read_text())
        except (OSError, json.JSONDecodeError):
            settings = None
        changed = (settings_diff(prev_settings, settings)
                   if prev_settings is not None and settings is not None
                   else {})
        history.append({"run": run_dir.name, "score": score,
                        "settings": settings, "changed": changed})
        if settings is not None:
            prev_settings = settings
    return history


#: Namespace keys the runner itself rewrites every run -- meaningless in a
#: "which knobs did the user change" diff.
_DIFF_IGNORE = {"source", "summary", "log", "save", "heatmap", "charts",
                "no_dashboard", "save_detections"}


def settings_diff(old: dict, new: dict) -> dict:
    """{key: (old_value, new_value)} for every setting that differs,
    ignoring the run-target fields the runner rewrites itself."""
    keys = (set(old) | set(new)) - _DIFF_IGNORE
    return {k: (old.get(k), new.get(k))
            for k in sorted(keys) if old.get(k) != new.get(k)}


def prepare_validation_namespace(ns, vset: ValidationSet, run_dir: Path):
    """Deep-copied namespace targeting *vset*'s video and *run_dir*.

    Everything the user dialed into the tab is preserved; only the run
    targets move: the set's video as source, streams into the run dir,
    detections stream ON (the IoU metric's input), no saved video /
    heatmaps / charts / dashboard.
    """
    if not vset.video:
        raise ValidationSetError(f"Set {vset.name!r} has no video.")
    if not Path(vset.video).is_file():
        raise ValidationSetError(
            f"Set {vset.name!r} video not found: {vset.video}")
    run_dir = Path(run_dir)
    stem = Path(vset.video).stem
    ns2 = copy.deepcopy(ns)
    ns2.source = str(vset.video)
    ns2.summary = str(run_dir / f"{stem}_summary.csv")
    ns2.log = str(run_dir / f"{stem}_events.csv")
    ns2.save = None
    ns2.heatmap = None
    ns2.charts = None
    ns2.no_dashboard = True
    ns2.save_detections = True
    return ns2


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def score_and_persist(vset: ValidationSet, run_dir: Path, ns=None,
                      radius: float = 80.0) -> dict:
    """Score *run_dir* against *vset*; write score.json (+ settings.json
    when *ns* is given) into the run dir.  Returns the score dict."""
    run_dir = Path(run_dir)
    stem = Path(vset.video).stem
    result = score_run(vset, run_dir, stem, radius=radius)
    (run_dir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    if ns is not None:
        snapshot = {k: _jsonable(v) for k, v in sorted(vars(ns).items())
                    if not k.startswith("_")}
        (run_dir / "settings.json").write_text(
            json.dumps(snapshot, indent=2, default=str) + "\n")
    return result
