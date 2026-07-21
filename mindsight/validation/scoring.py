"""
validation/scoring.py — Score a pipeline run against a validation set.

The gaze metrics reproduce ``scripts/eval_gaze.py score`` (endpoint
distance mean/median/p95, hit@radius, off-screen rank AUC, per-
participant means) and add the two suite-only metrics from the plan:
Mean Angular Error (predicted ray vs origin->label — needs no new label
type) and object IoU (labeled boxes vs the opt-in
``{stem}_detections.csv`` side stream).

W4C multi-clip sets: every clip's streams are collected from the SAME
run dir (one stem per clip, ``ValidationSet.clip_stems()``), the raw
accumulators are pooled, and the metrics are computed once over the
pool — for a single-clip set this reduces exactly to the historical
computation.  ``per_video`` carries the per-clip breakdown; clips whose
gaze stream is missing (e.g. a run cancelled between clips) are listed
in ``skipped_videos`` instead of failing the whole score.

Pure functions over CSV/JSON files — no GUI, no cv2, no pipeline.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .store import ValidationClip, ValidationSet, ValidationSetError


def _rank_auc(pos, neg):
    """AUC via the rank-sum identity (eval_gaze formula); None when a
    class is empty.  ``pos`` should score LOWER than ``neg``."""
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in neg] + [(v, 0) for v in pos],
                    key=lambda t: t[0])
    n = len(ranked)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    pos_rank_sum = sum(r for r, (_v, is_neg) in zip(ranks, ranked) if is_neg)
    n_neg, n_pos = len(neg), len(pos)
    return (pos_rank_sum - n_neg * (n_neg + 1) / 2.0) / (n_neg * n_pos)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _load_gaze_rows(gaze_csv: Path):
    """({(frame, face_idx): row}, {(frame, participant_label): row})."""
    by_tid: dict = {}
    by_label: dict = {}
    with open(gaze_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            frame = int(row["frame"])
            by_tid[(frame, int(row["face_idx"]))] = row
            by_label[(frame, row["participant_label"])] = row
    return by_tid, by_label


def _load_detections(det_csv: Path):
    """{frame: [(x1, y1, x2, y2), ...]} — classes are ignored: labeled
    boxes carry user-chosen names that need not match detector classes."""
    dets = defaultdict(list)
    with open(det_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            dets[int(row["frame"])].append(
                (int(row["x1"]), int(row["y1"]),
                 int(row["x2"]), int(row["y2"])))
    return dets


class _Pool:
    """Raw scoring accumulators, mergeable across clips."""

    def __init__(self):
        self.errs: list[float] = []
        self.maes: list[float] = []
        self.hits = 0
        self.scored = 0
        self.on_inout: list[float] = []
        self.off_inout: list[float] = []
        self.per_pid: dict = defaultdict(list)
        self.ious: list[float] = []
        self.has_detections = False


def _collect_clip(clip: ValidationClip, run_dir: Path, stem: str,
                  radius: float, pool: _Pool) -> None:
    """Score one clip's streams into *pool* (the historical per-file
    computation, minus the final metric formulas)."""
    by_tid, by_label = _load_gaze_rows(Path(run_dir) / f"{stem}_gaze.csv")

    for frame in sorted(clip.labels):
        for pid, label in sorted(clip.labels[frame].items()):
            # Digit participant keys match the gaze stream's face_idx
            # (the eval-harness convention); anything else matches the
            # participant_label column (custom labels).
            row = (by_tid.get((frame, int(pid))) if pid.isdigit()
                   else by_label.get((frame, pid)))
            if row is None or label in ("uncertain", "skip"):
                continue
            inout = float(row["inout_score"]) if row["inout_score"] else None
            if label == "offscreen":
                if inout is not None:
                    pool.off_inout.append(inout)
                continue
            if inout is not None:
                pool.on_inout.append(inout)
            ex, ey = float(row["ray_end_x"]), float(row["ray_end_y"])
            err = math.hypot(ex - label["x"], ey - label["y"])
            pool.errs.append(err)
            pool.per_pid[pid].append(err)
            pool.scored += 1
            if err <= radius:
                pool.hits += 1
            # Mean Angular Error: predicted ray vs origin->label.
            ox, oy = float(row["origin_x"]), float(row["origin_y"])
            vp = (ex - ox, ey - oy)
            vt = (label["x"] - ox, label["y"] - oy)
            np_, nt = math.hypot(*vp), math.hypot(*vt)
            if np_ > 1e-6 and nt > 1e-6:
                cosang = (vp[0] * vt[0] + vp[1] * vt[1]) / (np_ * nt)
                pool.maes.append(math.degrees(
                    math.acos(max(-1.0, min(1.0, cosang)))))

    # Object IoU vs the opt-in detections stream: for every labeled box
    # on a labeled frame, the best IoU against ANY detection that frame
    # (0 when the run detected nothing there); mean over labeled boxes.
    det_csv = Path(run_dir) / f"{stem}_detections.csv"
    if clip.objects and det_csv.is_file():
        pool.has_detections = True
        dets = _load_detections(det_csv)
        pool.ious.extend(
            max((_iou((b["x1"], b["y1"], b["x2"], b["y2"]), d)
                 for d in dets.get(frame, [])), default=0.0)
            for frame, boxes in sorted(clip.objects.items())
            for b in boxes
        )


def _metrics(pool: _Pool, radius: float) -> dict:
    errs = sorted(pool.errs)

    def pct(p):
        return errs[min(len(errs) - 1, int(p / 100 * len(errs)))]

    result = {
        "scored_points": pool.scored,
        "endpoint_px_mean": sum(errs) / len(errs),
        "endpoint_px_median": pct(50),
        "endpoint_px_p95": pct(95),
        "hit_rate": pool.hits / pool.scored,
        "hit_radius_px": radius,
        "mae_deg_mean": (sum(pool.maes) / len(pool.maes)
                         if pool.maes else None),
        "offscreen_auc": _rank_auc(pool.off_inout, pool.on_inout),
        "offscreen_labels": len(pool.off_inout),
        "per_participant_mean_px": {
            str(pid): sum(v) / len(v)
            for pid, v in sorted(pool.per_pid.items())},
    }
    if pool.has_detections:
        result["object_iou_mean"] = (sum(pool.ious) / len(pool.ious)
                                     if pool.ious else None)
        result["object_boxes_scored"] = len(pool.ious)
    else:
        result["object_iou_mean"] = None
        result["object_boxes_scored"] = 0
    return result


def score_run(vset: ValidationSet, run_dir: Path, stem: str | None = None,
              radius: float = 80.0) -> dict:
    """Score one run's streams against *vset*; returns the metrics dict.

    Multi-clip sets score every clip out of the same *run_dir* (one stem
    per clip) and pool; single-clip sets reproduce the historical
    result exactly.  *stem* overrides the single-clip stem (legacy
    signature).  Raises ValidationSetError when nothing could be scored
    (plain-English message for the workbench).
    """
    run_dir = Path(run_dir)
    if not vset.clips:
        raise ValidationSetError(f"Set {vset.name!r} has no videos.")
    stems = vset.clip_stems()
    if stem is not None and len(vset.clips) == 1:
        stems = [stem]

    pool = _Pool()
    per_video: dict = {}
    skipped: list[str] = []
    for clip, clip_stem in zip(vset.clips, stems):
        if not (run_dir / f"{clip_stem}_gaze.csv").is_file():
            skipped.append(clip_stem)
            continue
        clip_pool = _Pool()
        _collect_clip(clip, run_dir, clip_stem, radius, clip_pool)
        if clip_pool.errs:
            per_video[clip_stem] = {
                "scored_points": clip_pool.scored,
                "endpoint_px_mean": sum(clip_pool.errs) / len(clip_pool.errs),
                "hit_rate": clip_pool.hits / clip_pool.scored,
            }
        # Merge into the pooled accumulators.
        pool.errs.extend(clip_pool.errs)
        pool.maes.extend(clip_pool.maes)
        pool.hits += clip_pool.hits
        pool.scored += clip_pool.scored
        pool.on_inout.extend(clip_pool.on_inout)
        pool.off_inout.extend(clip_pool.off_inout)
        for pid, v in clip_pool.per_pid.items():
            pool.per_pid[pid].extend(v)
        pool.ious.extend(clip_pool.ious)
        pool.has_detections |= clip_pool.has_detections

    if skipped and not per_video and not pool.errs:
        raise ValidationSetError(
            "Run has no gaze stream "
            f"({skipped[0]}_gaze.csv missing).")
    if not pool.errs:
        raise ValidationSetError(
            "No scorable (on-screen) label matched any gaze row -- check "
            "that the set's participants exist in this clip.")

    result = _metrics(pool, radius)
    if len(vset.clips) > 1:
        result["per_video"] = per_video
        result["videos_scored"] = len(per_video)
        if skipped:
            result["skipped_videos"] = skipped
    return result
