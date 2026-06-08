"""
DataCollection/global_csv.py — Project-level CSV aggregation.

After all per-video processing is complete, this module:
1. Combines per-video CSVs into a single Global CSV (summary and/or events).
2. Splits the Global CSV by condition tags into per-condition CSVs.

Per-video CSVs are expected to have ``video_name`` and ``conditions`` as
their first two columns (added by project mode in Phase 2).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Characters unsafe for filenames — replaced with underscore
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|]')


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return _UNSAFE_CHARS.sub('_', name).strip()


def generate_global_csv(csv_dir: Path, csv_type: str) -> Path | None:
    """Combine all per-video CSVs of the given type into a single global file.

    Parameters
    ----------
    csv_dir   : directory containing per-video CSV files.
    csv_type  : ``"summary"`` or ``"events"``.

    Returns
    -------
    Path to the written global CSV, or ``None`` if no source files were found.
    """
    if csv_type == "summary":
        suffix = "_Summary.csv"
        out_name = "Global_Summary.csv"
    elif csv_type == "events":
        suffix = "_Events.csv"
        out_name = "Global_Events.csv"
    else:
        raise ValueError(f"Unknown csv_type: {csv_type!r}")

    # Discover per-video CSV files (exclude existing Global files)
    source_files = sorted(
        p for p in csv_dir.iterdir()
        if p.name.endswith(suffix) and not p.name.startswith("Global_")
    )
    if not source_files:
        return None

    out_path = csv_dir / out_name
    header: list[str] | None = None

    with open(out_path, "w", newline="") as out_fh:
        writer = csv.writer(out_fh)

        for src in source_files:
            with open(src, "r", newline="") as in_fh:
                reader = csv.reader(in_fh)
                for row in reader:
                    if not row:
                        continue
                    # Skip comment lines (# SECTION:, # GROUP:, etc.)
                    if row[0].startswith('#'):
                        continue
                    # Write header only once (from first file)
                    if header is None:
                        header = row
                        writer.writerow(row)
                        continue
                    # Skip duplicate header rows from subsequent files
                    if row == header:
                        continue
                    writer.writerow(row)

    print(f"Global {csv_type} CSV \u2192 {out_path}")
    return out_path


def generate_condition_csvs(global_csv_path: Path, condition_dir: Path,
                            csv_type: str) -> list[Path]:
    """Split a global CSV by the ``conditions`` column.

    Each unique tag gets its own CSV. A video with multiple pipe-delimited
    tags (e.g. ``"Emotional|Group A"``) appears in each matching file.

    Parameters
    ----------
    global_csv_path : path to the global CSV (summary or events).
    condition_dir   : directory to write per-condition files into.
    csv_type        : ``"summary"`` or ``"events"`` (used for filename suffix).

    Returns
    -------
    List of paths to the written per-condition CSV files.
    """
    condition_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_Summary.csv" if csv_type == "summary" else "_Events.csv"

    # First pass: discover all unique tags and build row buckets
    header = None
    tag_rows: dict[str, list[list[str]]] = {}

    with open(global_csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if header is None:
                header = row
                continue
            # Column index 1 is 'conditions' (pipe-delimited)
            conditions_str = row[1] if len(row) > 1 else ""
            tags = [t.strip() for t in conditions_str.split("|") if t.strip()]
            for tag in tags:
                tag_rows.setdefault(tag, []).append(row)

    if not header or not tag_rows:
        return []

    # Write per-condition files
    written: list[Path] = []
    for tag, rows in sorted(tag_rows.items()):
        safe_name = _sanitize_filename(tag)
        out_path = condition_dir / f"{safe_name}{suffix}"
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        written.append(out_path)
        print(f"  Condition '{tag}' \u2192 {out_path}")

    return written
