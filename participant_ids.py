"""
participant_ids.py — Parse and manage custom participant ID mappings.

Provides two input paths:

1. **Inline** (CLI ``--participant-ids S70,S71``):
   Positional — the first label maps to track 0, the second to track 1, etc.

2. **CSV file** (project mode ``participant_ids.csv``):
   Normalised format with columns ``video_filename, track_id, participant_label``.
   One row per participant per video.  Example::

       video_filename,track_id,participant_label
       video1.mp4,0,S70
       video1.mp4,1,S71
       video2.mp4,0,SubjectA
       video2.mp4,1,SubjectB
"""

from __future__ import annotations

import csv
from pathlib import Path


def parse_inline_ids(id_string: str) -> dict[int, str]:
    """Parse a comma-separated participant-ID string into a track→label map.

    ``"S70,S71,S72"`` → ``{0: "S70", 1: "S71", 2: "S72"}``

    Returns an empty dict if *id_string* is empty or whitespace-only.
    """
    labels = [s.strip() for s in id_string.split(",") if s.strip()]
    return {i: lbl for i, lbl in enumerate(labels)}


def load_participant_csv(csv_path: str | Path) -> dict[str, dict[int, str]]:
    """Load a ``participant_ids.csv`` file.

    Parameters
    ----------
    csv_path : path to the CSV file.

    Returns
    -------
    ``{video_filename: {track_id: participant_label}}``

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If a required column is missing or a ``track_id`` is not an integer.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Participant-ID CSV not found: {path}")

    result: dict[str, dict[int, str]] = {}

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for col in ("video_filename", "track_id", "participant_label"):
            if col not in (reader.fieldnames or []):
                raise ValueError(
                    f"Participant-ID CSV is missing required column '{col}'. "
                    f"Found columns: {reader.fieldnames}"
                )

        warned_dupes: set[tuple[str, int]] = set()
        for row_num, row in enumerate(reader, start=2):
            fname = row["video_filename"].strip()
            label = row["participant_label"].strip()
            try:
                tid = int(row["track_id"])
            except ValueError:
                raise ValueError(
                    f"Row {row_num}: track_id must be an integer, "
                    f"got '{row['track_id']}'"
                )

            per_video = result.setdefault(fname, {})
            if tid in per_video and (fname, tid) not in warned_dupes:
                print(f"Warning: participant_ids.csv row {row_num}: "
                      f"duplicate track_id {tid} for '{fname}' "
                      f"(overwriting '{per_video[tid]}' with '{label}')")
                warned_dupes.add((fname, tid))
            per_video[tid] = label

    return result


def load_aux_streams_from_csv(csv_path: str | Path) -> list:
    """Extract auxiliary stream definitions from a ``participant_ids.csv``.

    The CSV may contain optional columns ``aux_stream_type`` and
    ``aux_stream_source``.  When both are present *and non-empty* for a
    row, an ``AuxStreamConfig`` is created using the row's
    ``participant_label`` as the PID.

    Returns a (possibly empty) list of ``AuxStreamConfig`` objects.
    The function is backward-compatible: CSVs without the extra columns
    simply return an empty list.
    """
    from pipeline_config import AuxStreamConfig

    path = Path(csv_path)
    if not path.is_file():
        return []

    configs: list = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "aux_stream_type" not in fields or "aux_stream_source" not in fields:
            return []

        for row in reader:
            stype = row.get("aux_stream_type", "").strip()
            source = row.get("aux_stream_source", "").strip()
            label = row.get("participant_label", "").strip()
            if stype and source and label:
                configs.append(AuxStreamConfig(pid=label,
                                               stream_type=stype,
                                               source=source))
    return configs
