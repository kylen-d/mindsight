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

from ms.constants import OUTPUTS_ROOT as _OUTPUTS_ROOT
from ms.pipeline_config import resolve_display_pid

# Logical grouping of tracker sections in summary CSV.
# Tracker names map to PhenomenaPlugin.name attributes.
_TRACKER_GROUPS = [
    ("Dyadic Interactions",
     ["joint_attention", "mutual_gaze", "social_referencing", "gaze_follow"]),
    ("Individual Gaze Behaviour",
     ["attn_span", "gaze_aversion", "scanpath"]),
    ("Group Dynamics",
     ["gaze_leadership"]),
]

_TRACKER_DISPLAY_NAMES = {
    "joint_attention": "Joint Attention",
    "mutual_gaze": "Mutual Gaze",
    "social_referencing": "Social Referencing",
    "gaze_follow": "Gaze Following",
    "attn_span": "Attention Span",
    "gaze_aversion": "Gaze Aversion",
    "scanpath": "Scanpath",
    "gaze_leadership": "Gaze Leadership",
}


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


def write_summary_csv(path, total_frames, look_counts,
                      all_trackers=None, pid_map=None,
                      video_name=None, conditions=''):
    """Write a post-run summary CSV to *path*.

    Parameters
    ----------
    path          : output file path (str or Path).
    total_frames  : total number of processed frames.
    look_counts   : dict mapping (face_idx, obj_cls) -> frame count.
    all_trackers  : list of PhenomenaPlugin instances (built-in + external).
                    Each tracker contributes its own CSV section via csv_rows().
                    JointAttentionTracker contributes the JA section automatically.
    video_name    : source video stem (str) or None for single-video mode.
    conditions    : pipe-delimited condition tags (str) or empty string.
    """
    # When video_name is set, prepend video_name and conditions to every data row.
    _has_project_cols = video_name is not None

    # Header rows from tracker csv_rows() start with "category"
    _HEADER_MARKERS = {"category"}

    def _prefix(row):
        """Prepend project columns to a data row if in project mode."""
        if not _has_project_cols or not row or str(row[0]).startswith('#'):
            return row
        # Tracker sub-header rows: prepend column names, not values
        if row[0] in _HEADER_MARKERS:
            return ["video_name", "conditions"] + row
        return [video_name, conditions] + row

    rows = []

    # ── Object look-time (built-in, not a tracker) ────────────────────────────
    rows.append(["# SECTION: Object Look Time"])
    header = ["category", "participant", "object",
              "frames_active", "total_frames", "value_pct"]
    if _has_project_cols:
        header = ["video_name", "conditions"] + header
    rows.append(header)
    for (face_idx, obj_cls), count in sorted(look_counts.items()):
        pct = count / total_frames * 100 if total_frames else 0.0
        rows.append(_prefix(["object_look_time", resolve_display_pid(face_idx, pid_map),
                             obj_cls, count, total_frames, f"{pct:.4f}"]))

    # ── Grouped tracker sections (built-in + external plugins) ───────────────
    # Build name→tracker lookup for ordering
    tracker_map = {}
    for tracker in (all_trackers or []):
        tracker_map[getattr(tracker, 'name', '')] = tracker

    emitted = set()
    for group_name, tracker_names in _TRACKER_GROUPS:
        group_started = False
        for tname in tracker_names:
            tracker = tracker_map.get(tname)
            if tracker is None:
                continue
            tracker_rows = tracker.csv_rows(total_frames, pid_map=pid_map)
            if not tracker_rows:
                continue
            if not group_started:
                rows.append([])
                rows.append([f"# GROUP: {group_name}"])
                group_started = True
            display = _TRACKER_DISPLAY_NAMES.get(tname, tname.replace('_', ' ').title())
            rows.append([])
            rows.append([f"# SECTION: {display}"])
            rows.extend(_prefix(r) for r in tracker_rows)
            emitted.add(tname)

    # Any trackers not in the known groups (external plugins)
    other_started = False
    for tracker in (all_trackers or []):
        tname = getattr(tracker, 'name', '')
        if tname in emitted:
            continue
        tracker_rows = tracker.csv_rows(total_frames, pid_map=pid_map)
        if not tracker_rows:
            continue
        if not other_started:
            rows.append([])
            rows.append(["# GROUP: Other"])
            other_started = True
        display = tname.replace('_', ' ').title()
        rows.append([])
        rows.append([f"# SECTION: {display}"])
        rows.extend(_prefix(r) for r in tracker_rows)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"Summary \u2192 {path}")
