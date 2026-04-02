"""
DataCollection/csv_output.py — CSV output and data recording.

Responsibilities
----------------
- write_summary_csv: writes a post-run multi-section CSV summarising all
  phenomena metrics collected during a session (joint attention, look-time,
  mutual gaze, social referencing, gaze following, attentional synchrony,
  gaze aversion, scanpaths, and gaze leadership).
"""

import csv
from pathlib import Path

from constants import OUTPUTS_ROOT as _OUTPUTS_ROOT


def resolve_summary_path(summary_arg, source) -> "str | None":
    """Resolve the --summary flag value to a concrete file path or None.

    summary_arg : True  → Outputs/CSV Files/[stem]_Summary_Output.csv
                  str   → that path
                  None/False → None (no summary written)
    source      : video file path (str/Path) or webcam index (int).
    """
    if not summary_arg:
        return None
    if summary_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        return str(_OUTPUTS_ROOT / "CSV Files" / f"{stem}_Summary_Output.csv")
    return summary_arg


def write_summary_csv(path, total_frames, joint_frames, look_counts,
                      all_trackers=None):
    """Write a post-run summary CSV to *path*.

    Parameters
    ----------
    path          : output file path (str or Path).
    total_frames  : total number of processed frames.
    joint_frames  : frames in which confirmed joint attention occurred.
    look_counts   : dict mapping (face_idx, obj_cls) -> frame count.
    all_trackers  : list of PhenomenaPlugin instances (built-in + external).
                    Each tracker contributes its own CSV section via csv_rows().
    """
    rows = []

    # ── Joint attention (built-in, not a tracker) ─────────────────────────────
    rows.append(["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"])
    ja_pct = joint_frames / total_frames * 100 if total_frames else 0.0
    rows.append(["joint_attention", "all", "",
                 joint_frames, total_frames, f"{ja_pct:.4f}"])

    # ── Object look-time (built-in, not a tracker) ───────────────────────────
    rows.append([])
    rows.append(["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"])
    for (face_idx, obj_cls), count in sorted(look_counts.items()):
        pct = count / total_frames * 100 if total_frames else 0.0
        rows.append(["object_look_time", f"P{face_idx}", obj_cls,
                     count, total_frames, f"{pct:.4f}"])

    # ── All tracker sections (built-in + external plugins) ───────────────────
    for tracker in (all_trackers or []):
        tracker_rows = tracker.csv_rows(total_frames)
        if tracker_rows:
            rows.append([])
            rows.extend(tracker_rows)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"Summary \u2192 {path}")
