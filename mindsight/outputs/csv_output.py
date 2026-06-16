"""
outputs/csv_output.py — Tidy long-format summary CSV output.

Responsibilities
----------------
- write_summary_tables: writes the post-run summary as tidy long-format
  tables. One scalar-metrics file ``{stem}_summary.csv`` holds every scalar
  phenomenon metric (columns
  ``video_name,conditions,phenomenon,participant,partner,object,metric,value``);
  each event/timeseries stream (scanpath, novel-salience events, eye-movement
  events, pupillometry timeseries, …) gets its own typed
  ``{stem}_<stream>.csv`` file, written only when it has data.

  Trackers contribute scalar rows via ``PhenomenaPlugin.summary_metrics`` and
  streams via ``PhenomenaPlugin.summary_tables``.  A plugin that still
  overrides only the legacy ``csv_rows`` hook has its rows dumped verbatim to
  ``{stem}_plugin_{name}.csv``.  Object look-time is emitted by this writer
  itself (it is not a tracker).
"""

import csv
import re
from pathlib import Path

from mindsight.constants import OUTPUTS_ROOT as _OUTPUTS_ROOT
from mindsight.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin

# Header for the scalar-metrics file. Identical in single and project mode
# (video_name/conditions are empty strings in single mode) so Global concat is
# a pure append.
_SUMMARY_HEADER = ["video_name", "conditions", "phenomenon", "participant",
                   "partner", "object", "metric", "value"]

_SUMMARY_SUFFIX_RE = re.compile(r"_summary$", re.IGNORECASE)


def resolve_summary_path(summary_arg, source) -> "str | None":
    """Resolve the --summary flag value to a concrete file path or None.

    summary_arg : True  → Outputs/CSV Files/[stem]_summary.csv
                  str   → that path
                  None/False → None (no summary written)
    source      : video file path (str/Path) or webcam index (int).
    """
    if not summary_arg:
        return None
    if summary_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        return str(_OUTPUTS_ROOT / "CSV Files" / f"{stem}_summary.csv")
    return summary_arg


def _seconds(frames, fps) -> str:
    """Format a frame count as seconds (3 dp), or "" when fps is unknown."""
    if not fps:
        return ""
    return f"{frames / fps:.3f}"


def _stream_base(summary_path: Path) -> "tuple[Path, str]":
    """Return (parent_dir, base_stem) for deriving stream filenames.

    ``.../trimmed_summary.csv`` → (.../, "trimmed"); a summary path that does
    not end in ``_summary`` falls back to its full stem.
    """
    stem = summary_path.stem
    base = _SUMMARY_SUFFIX_RE.sub("", stem)
    return summary_path.parent, base


def write_summary_tables(path, total_frames, fps, look_counts,
                         all_trackers=None, pid_map=None,
                         video_name=None, conditions=''):
    """Write the tidy summary file set rooted at *path*.

    Parameters
    ----------
    path          : scalar-metrics output file path (str or Path). Stream files
                    are written alongside it as ``{base}_<stream>.csv``.
    total_frames  : total number of processed frames.
    fps           : primary source frame rate (for seconds conversions).
    look_counts   : dict mapping (face_idx, obj_cls) -> frame count.
    all_trackers  : list of PhenomenaPlugin instances (built-in + external).
    video_name    : source video stem (str) or None for single-video mode.
    conditions    : pipe-delimited condition tags (str) or empty string.
    """
    vname = video_name if video_name is not None else ""
    conds = conditions or ""
    prefix = [vname, conds]

    summary_path = Path(path)
    parent, base = _stream_base(summary_path)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    scalar_rows: list[list] = []

    def _emit(phenomenon, participant, partner, obj, metric, value):
        scalar_rows.append([phenomenon, participant, partner, obj,
                            metric, value])

    # Object look-time (built-in, not a tracker).
    for (face_idx, obj_cls), count in sorted(look_counts.items()):
        pid = resolve_display_pid(face_idx, pid_map)
        pct = count / total_frames * 100 if total_frames else 0.0
        _emit("object_look_time", pid, "", obj_cls, "frames_active", count)
        _emit("object_look_time", pid, "", obj_cls, "seconds_active",
              _seconds(count, fps))
        _emit("object_look_time", pid, "", obj_cls, "pct_of_video",
              f"{pct:.4f}")

    # Tracker scalar metrics.
    for tracker in (all_trackers or []):
        default_name = (getattr(tracker, 'summary_label', None)
                        or getattr(tracker, 'name', ''))
        for m in tracker.summary_metrics(total_frames, fps, pid_map=pid_map):
            _emit(m.get('phenomenon', default_name),
                  m.get('participant', ''), m.get('partner', ''),
                  m.get('object', ''), m['metric'], m['value'])

    # Deterministic order: phenomenon, participant, partner, object, metric.
    scalar_rows.sort(key=lambda r: (str(r[0]), str(r[1]), str(r[2]),
                                    str(r[3]), str(r[4])))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_SUMMARY_HEADER)
        for r in scalar_rows:
            writer.writerow(prefix + r)
    print(f"Summary → {summary_path}")

    # ── Stream tables ─────────────────────────────────────────────────────────
    for tracker in (all_trackers or []):
        tables = tracker.summary_tables(total_frames, fps, pid_map=pid_map)
        for table_name, (header, rows) in tables.items():
            if not rows:
                continue
            out_path = parent / f"{base}_{table_name}.csv"
            with open(out_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["video_name", "conditions"] + list(header))
                for r in rows:
                    writer.writerow(prefix + list(r))
            print(f"  Stream → {out_path}")

    # ── Legacy csv_rows passthrough ───────────────────────────────────────────
    # A plugin overriding ONLY the legacy csv_rows hook (neither tidy hook) has
    # its rows dumped verbatim so third-party plugins keep producing output.
    for tracker in (all_trackers or []):
        t = type(tracker)
        overrides_legacy = t.csv_rows is not PhenomenaPlugin.csv_rows
        overrides_tidy = (
            t.summary_metrics is not PhenomenaPlugin.summary_metrics
            or t.summary_tables is not PhenomenaPlugin.summary_tables)
        if not overrides_legacy or overrides_tidy:
            continue
        rows = tracker.csv_rows(total_frames, pid_map=pid_map)
        if not rows:
            continue
        name = getattr(tracker, 'name', 'plugin') or 'plugin'
        out_path = parent / f"{base}_plugin_{name}.csv"
        with open(out_path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"  Plugin (legacy) → {out_path}")
