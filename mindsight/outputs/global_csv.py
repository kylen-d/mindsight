"""
outputs/global_csv.py — Project-level CSV aggregation (tidy-aware).

After all per-video processing is complete, this module:
1. Concatenates each per-video tidy table type (summary + every event/
   timeseries stream) into a single ``Global_<type>.csv``.
2. Splits each global table by condition tags into per-condition CSVs.

Every tidy table carries ``video_name`` and ``conditions`` as its first two
columns and a single, uniform header across videos, so aggregation is a pure
append with the header written once. There are no ``#`` comment rows anymore.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Characters unsafe for filenames — replaced with underscore
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')

# Per-video table suffix -> global output filename. Covers the scalar summary,
# the frame-event log, and every tidy stream file a plugin can emit. The frame
# log keeps its capitalised ``_Events.csv`` suffix (distinct from the lowercase
# ``_events.csv`` streams).
GLOBAL_TABLES: list[tuple[str, str]] = [
    ("_summary.csv", "Global_summary.csv"),
    ("_Events.csv", "Global_Events.csv"),
    ("_gaze.csv", "Global_gaze.csv"),
    ("_phenomena_events.csv", "Global_phenomena_events.csv"),
    ("_scanpath.csv", "Global_scanpath.csv"),
    ("_novel_salience_events.csv", "Global_novel_salience_events.csv"),
    ("_eye_movement_events.csv", "Global_eye_movement_events.csv"),
    ("_pupillometry_timeseries.csv", "Global_pupillometry_timeseries.csv"),
    ("_pupillometry_blinks.csv", "Global_pupillometry_blinks.csv"),
]


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return _UNSAFE_CHARS.sub('_', name).strip()


def generate_global_csv(csv_dir: Path, suffix: str, out_name: str,
                        *, search_dirs=None) -> Path | None:
    """Concatenate every per-video file ending in *suffix* into *out_name*.

    Parameters
    ----------
    csv_dir  : directory the global CSV is written into (``Outputs/CSV Files/``).
    suffix   : per-video filename suffix (e.g. ``"_summary.csv"``).
    out_name : global output filename (e.g. ``"Global_summary.csv"``).
    search_dirs : optional directories to gather per-run source files from
        (SP3.1 Q3 -- the per-run ``Outputs/Runs/<run_id>/`` dirs for run-folder
        projects).  When ``None`` the flat ``csv_dir`` is scanned (legacy,
        byte-unchanged).  The global output always lands in ``csv_dir``.

    Returns
    -------
    Path to the written global CSV, or ``None`` if no source files were found.
    """
    dirs = [csv_dir] if search_dirs is None else list(search_dirs)
    source_files = sorted(
        (p for d in dirs if Path(d).is_dir() for p in Path(d).iterdir()
         if p.name.endswith(suffix) and not p.name.startswith("Global_")),
        key=lambda p: (p.parent.name, p.name),
    )
    if not source_files:
        return None

    out_path = csv_dir / out_name
    header: list[str] | None = None

    with open(out_path, "w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        for src in source_files:
            with open(src, "r", newline="") as in_fh:
                for row in csv.reader(in_fh):
                    if not row:
                        continue
                    if header is None:
                        header = row
                        writer.writerow(row)
                        continue
                    # Skip the repeated header row from subsequent files.
                    if row == header:
                        continue
                    writer.writerow(row)

    print(f"Global {out_name} → {out_path}")
    return out_path


def generate_condition_csvs(global_csv_path: Path, condition_dir: Path,
                            suffix: str) -> list[Path]:
    """Split a global tidy table by the ``conditions`` column.

    Each unique tag gets its own ``{tag}{suffix}`` file. A video with multiple
    pipe-delimited tags (e.g. ``"Emotional|Group A"``) appears in each match.

    Parameters
    ----------
    global_csv_path : path to a global tidy table.
    condition_dir   : directory to write per-condition files into.
    suffix          : per-video table suffix (e.g. ``"_summary.csv"``), used to
                      name output files ``{tag}{suffix}``.

    Returns
    -------
    List of paths to the written per-condition CSV files.
    """
    condition_dir.mkdir(parents=True, exist_ok=True)

    header = None
    tag_rows: dict[str, list[list[str]]] = {}

    with open(global_csv_path, "r", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            if header is None:
                header = row
                continue
            # Column index 1 is 'conditions' (pipe-delimited).
            conditions_str = row[1] if len(row) > 1 else ""
            tags = [t.strip() for t in conditions_str.split("|") if t.strip()]
            for tag in tags:
                tag_rows.setdefault(tag, []).append(row)

    if not header or not tag_rows:
        return []

    written: list[Path] = []
    for tag, rows in sorted(tag_rows.items()):
        safe_name = _sanitize_filename(tag)
        out_path = condition_dir / f"{safe_name}{suffix}"
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(out_path)
        print(f"  Condition '{tag}' → {out_path}")

    return written
