"""
run_outputs.py -- Pure (no Qt) readers for a run's already-written outputs
(SP3.1 Batch G fix-forward, G-ENH-4).

Consume-don't-compute (D11): the Analyze Footage output panel renders what the
pipeline already wrote -- the per-run Events / summary / stream CSVs -- and
NOTHING here writes into a project's Outputs/ tree (that would break the P2/P3
identity baselines and booby trap T1).  Charts are rendered in-GUI from these
readers; they are display-only.

Works for both layouts: legacy runs place their CSVs in ``Outputs/CSV Files/``,
run-folder runs in ``Outputs/Runs/<run_id>/`` -- both are reachable through the
``RunSpec.output_paths`` each producer staged, so no layout branching happens
here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RunOutputs:
    """The on-disk output CSVs of one completed (or partially written) run."""

    run_id: str
    stem: str                    # output naming stem (video_name column value)
    csv_paths: tuple             # every CSV found for this run, sorted
    summary: Path | None         # {stem}_summary.csv when present
    events: Path | None          # {stem}_Events.csv when present


def discover_run_outputs(specs) -> list:
    """Return a :class:`RunOutputs` per run that has ANY CSV written on disk.

    *specs* is the project's ``list[RunSpec]``; each spec's staged
    ``output_paths`` names where its CSVs live (legacy flat vs per-run mirror).
    Stream CSVs (scanpath, novel-salience...) are picked up by the shared stem
    prefix inside the run's output directory.
    """
    out: list = []
    for spec in specs:
        summary = Path(spec.output_paths["summary"])
        events = Path(spec.output_paths["log"])
        stem = summary.name[:-len("_summary.csv")] \
            if summary.name.endswith("_summary.csv") else Path(spec.source).stem
        run_dir = summary.parent
        found = []
        if run_dir.is_dir():
            found = sorted(
                p for p in run_dir.glob(f"{stem}*.csv") if p.is_file())
        if not found:
            continue
        out.append(RunOutputs(
            run_id=spec.run_id,
            stem=stem,
            csv_paths=tuple(found),
            summary=summary if summary.is_file() else None,
            events=events if events.is_file() else None,
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CSV table loading (read-only viewer)
# ══════════════════════════════════════════════════════════════════════════════


def load_csv_rows(path, max_rows: int = 1000) -> tuple:
    """Read a CSV: ``(header, rows, truncated)``, capped at *max_rows* rows."""
    header: list = []
    rows: list = []
    truncated = False
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            if len(rows) >= max_rows:
                truncated = True
                break
            rows.append(row)
    return header, rows, truncated


# ══════════════════════════════════════════════════════════════════════════════
# Chart series (from the written CSVs -- display-only)
# ══════════════════════════════════════════════════════════════════════════════


def look_time_table(summary_csv, *, metric: str = "pct_of_video",
                    phenomenon: str = "object_look_time",
                    max_objects: int = 12) -> dict:
    """``participant -> {object: value}`` from a run's summary CSV.

    Reads the tidy summary rows (phenomenon/participant/object/metric/value)
    and keeps the *max_objects* objects with the highest total value so the
    bar chart stays readable.
    """
    table: dict = {}
    totals: dict = {}
    with open(summary_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("phenomenon") != phenomenon
                    or row.get("metric") != metric):
                continue
            who = row.get("participant") or "?"
            obj = row.get("object") or "?"
            try:
                val = float(row.get("value", ""))
            except ValueError:
                continue
            table.setdefault(who, {})[obj] = val
            totals[obj] = totals.get(obj, 0.0) + val
    if len(totals) > max_objects:
        keep = set(sorted(totals, key=totals.get, reverse=True)[:max_objects])
        table = {who: {o: v for o, v in objs.items() if o in keep}
                 for who, objs in table.items()}
    return table


def gaze_timeline(events_csv, *, max_rows: int = 20000) -> tuple:
    """Gaze-target timeline from a run's Events CSV.

    Returns ``(objects, per_participant)`` where *objects* is the ordered list
    of distinct gaze targets and *per_participant* maps each participant label
    to ``(t_seconds_list, object_index_list)`` ready to scatter-plot.
    """
    objects: list = []
    obj_idx: dict = {}
    per: dict = {}
    with open(events_csv, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= max_rows:
                break
            obj = row.get("object") or "?"
            try:
                t = float(row.get("t_seconds", ""))
            except ValueError:
                continue
            if obj not in obj_idx:
                obj_idx[obj] = len(objects)
                objects.append(obj)
            who = row.get("participant_label") or row.get("face_idx") or "?"
            xs, ys = per.setdefault(str(who), ([], []))
            xs.append(t)
            ys.append(obj_idx[obj])
    return objects, per
