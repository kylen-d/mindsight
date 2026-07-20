"""
validation/sweep.py — Auto-tune sweep engine (W4C item 1).

A sweep is the eval-matrix pattern, in-app: take the tab's CURRENT
namespace as the base, vary one or two knobs over user-chosen value
lists, and run every combination sequentially through the ordinary
validation runner.  Each combination is a normal ``run-NNN`` with its
own ``score.json`` + ``settings.json``, so History and the settings
diff already understand sweep runs for free.  A sweep manifest
(``.runs/<set>/sweep-NNN.json``) records the combo list, run-dir names
and winner so the workbench can reopen the last sweep table.

GUI-free by design: combo expansion, namespace overrides, manifest
persistence and winner pick are all plain functions.  The Auto-tune
dialog owns the Qt worker loop.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

from .runner import _DIFF_IGNORE, prepare_validation_namespace, runs_root
from .store import ValidationSet, ValidationSetError, _slug

#: Hard ceiling on combinations per sweep (ruling R6, 2026-07-19).
COMBO_CAP = 12

#: Knobs offered by the Auto-tune dialog's picker (ruling R6): the
#: kickoff's five plus the two the user's own tuning history touches
#: (rf_len_slew, detect_scale).  ``(dest, label, cast)`` — the engine
#: itself accepts any namespace dest; this list only curates the UI.
CURATED_KNOBS = (
    ("rf_len_gain", "Ray length gain", float),
    ("min_call_gap", "Min call gap (frames)", int),
    ("rf_len_refresh_gap", "Length refresh gap (frames)", int),
    ("conf", "Detection confidence", float),
    ("snap_quality_thresh", "Snap quality threshold", float),
    ("rf_len_slew", "Length slew (updates)", int),
    ("detect_scale", "Detection scale", float),
)

_SWEEP_RE = re.compile(r"^sweep-(\d+)\.json$")


def expand_combos(knobs, cap: int = COMBO_CAP) -> list[dict]:
    """``[(dest, [values…]), …]`` (one or two knobs) → override dicts,
    one per combination, in cartesian order (first knob outermost).

    Raises ValidationSetError (plain-English, dialog-ready) on no
    knobs, more than two, duplicate dests, an empty value list, a
    runner-owned dest, or a combination count over *cap*.
    """
    if not knobs:
        raise ValidationSetError("Pick at least one knob to sweep.")
    if len(knobs) > 2:
        raise ValidationSetError(
            f"A sweep varies at most two knobs, got {len(knobs)}.")
    dests = [d for d, _ in knobs]
    if len(set(dests)) != len(dests):
        raise ValidationSetError(
            "The same knob is listed twice -- pick two different knobs.")
    for dest, values in knobs:
        if dest in _DIFF_IGNORE:
            raise ValidationSetError(
                f"{dest!r} is a run-target field the runner rewrites -- "
                "it cannot be swept.")
        if not values:
            raise ValidationSetError(f"Knob {dest!r} has no values.")
    n = math.prod(len(values) for _, values in knobs)
    if n > cap:
        raise ValidationSetError(
            f"{n} combinations exceeds the sweep cap of {cap} -- "
            "shorten a value list.")
    return [dict(zip(dests, combo))
            for combo in itertools.product(*(values for _, values in knobs))]


def estimate_seconds(n_combos: int, clip_frames: int,
                     avg_fps) -> float | None:
    """Wall-clock estimate for a sweep (None when no fps is on record —
    i.e. the set has never scored a run with a live counter)."""
    if not avg_fps or avg_fps <= 0 or not clip_frames:
        return None
    return n_combos * clip_frames / float(avg_fps)


def prepare_sweep_namespace(base_ns, vset: ValidationSet, run_dir: Path,
                            overrides: dict):
    """One combo's namespace: the ordinary validation preparation (deep
    copy, run targets rewritten) with *overrides* applied on top.  The
    base namespace is never mutated, so combos stay independent."""
    ns = prepare_validation_namespace(base_ns, vset, run_dir)
    for dest, value in overrides.items():
        if dest in _DIFF_IGNORE:
            raise ValidationSetError(
                f"{dest!r} is a run-target field the runner rewrites -- "
                "it cannot be swept.")
        setattr(ns, dest, value)
    return ns


# ── Manifest ─────────────────────────────────────────────────────────────────

def new_sweep_manifest(set_name: str, knobs) -> dict:
    """Fresh manifest for a sweep over *knobs* (same shape expand_combos
    takes).  ``results`` fills in combo order as runs finish:
    ``{"overrides", "run", "score", "error"}`` per entry."""
    return {
        "format": 1,
        "set": set_name,
        "knobs": [[dest, list(values)] for dest, values in knobs],
        "results": [],
        "winner": None,
    }


def allocate_sweep_path(validation_dir: Path, set_name: str) -> Path:
    """Next sequential ``sweep-NNN.json`` path for *set_name* (parent
    dirs created; the file itself is not)."""
    base = runs_root(validation_dir) / _slug(set_name)
    base.mkdir(parents=True, exist_ok=True)
    seq = 0
    for p in base.iterdir():
        m = _SWEEP_RE.match(p.name)
        if m:
            seq = max(seq, int(m.group(1)))
    return base / f"sweep-{seq + 1:03d}.json"


def save_sweep(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def latest_sweep(validation_dir: Path, set_name: str) -> dict | None:
    """Most recent readable sweep manifest for *set_name* (None when the
    set has never swept).  Unreadable files are skipped."""
    base = runs_root(validation_dir) / _slug(set_name)
    if not base.is_dir():
        return None
    paths = sorted((p for p in base.iterdir() if _SWEEP_RE.match(p.name)),
                   key=lambda p: p.name, reverse=True)
    for path in paths:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def pick_winner(results) -> int | None:
    """Index of the scored result with the lowest mean px error (the
    sweep's headline metric); None when nothing scored."""
    best = None
    for i, entry in enumerate(results):
        score = entry.get("score") or {}
        mean = score.get("endpoint_px_mean")
        if mean is None:
            continue
        if best is None or mean < results[best]["score"]["endpoint_px_mean"]:
            best = i
    return best
