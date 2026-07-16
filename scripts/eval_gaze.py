#!/usr/bin/env python
"""eval_gaze.py -- gaze-accuracy eval runner + scorer (v1.1 W3.0).

Dev tool.  Gives accuracy work a NUMBER: runs the pipeline on a clip into an
isolated run dir, then scores its per-frame gaze stream ({stem}_gaze.csv,
v1.1 W1.4) against hand labels from scripts/eval_annotate.py.

Usage
-----
Run a config (everything after ``--`` passes through to MindSight)::

    python scripts/eval_gaze.py run test_data/trimmed.mp4 --tag baseline -- \
        --model Weights/YOLO/yolov8n.pt \
        --mgaze-model Weights/MGaze/resnet50_gaze.onnx \
        --rf-gazelle-model Weights/Gazelle/gazelle_dinov2_vitb14.pt \
        --rf-gazelle-interval 10

Score a run against the labels::

    python scripts/eval_gaze.py score test_data/trimmed.mp4 --tag baseline

Compare every scored run::

    python scripts/eval_gaze.py table test_data/trimmed.mp4

Metrics
-------
* endpoint_px      mean / median / p95 distance from the finalized ray end
                   to the labeled target point (on-screen labels only)
* endpoint_norm    the same, divided by the frame diagonal
* hit_rate         fraction of labeled frames where the ray end lands within
                   --radius px of the label (default 80 = tip radius default)
* offscreen_auc    how well inout_score separates off-screen from on-screen
                   labels (rank AUC; needs the inout head active, W3.1)

Runs live in eval_data/runs/<tag>/; eval_data/ is untracked by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval_data"
PYTHON = sys.executable


def _run(video: Path, tag: str, passthrough: list[str]) -> None:
    run_dir = EVAL_DIR / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    cmd = [PYTHON, str(REPO_ROOT / "MindSight.py"),
           "--source", str(video),
           "--log", str(run_dir / f"{stem}_events.csv"),
           "--summary", str(run_dir / f"{stem}_summary.csv"),
           "--no-dashboard", *passthrough]
    env = dict(os.environ)
    env["HOME"] = tempfile.mkdtemp(prefix="mindsight-eval-home-")
    env["PYTHONPATH"] = str(REPO_ROOT)
    # Fresh fake HOMEs keep app state isolated, but torch.hub would then
    # re-download the DINOv2 backbone (~330MB+) EVERY run -- the Gaze-LLE
    # checkpoints carry heads only.  Share one hub cache across eval runs.
    env.setdefault("TORCH_HOME", str(EVAL_DIR / "torch_cache"))
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if res.returncode != 0:
        sys.exit(f"pipeline run failed (exit {res.returncode})")
    print(f"run '{tag}' complete -> {run_dir}")


def _load_labels(video: Path) -> dict:
    path = EVAL_DIR / f"{video.stem}_labels.json"
    if not path.is_file():
        sys.exit(f"error: {path} not found -- label frames first with "
                 f"scripts/eval_annotate.py")
    return json.loads(path.read_text())["labels"]


def _load_gaze_rows(run_dir: Path, stem: str) -> dict:
    """{(frame, tid): row} from a run's gaze stream."""
    path = run_dir / f"{stem}_gaze.csv"
    if not path.is_file():
        sys.exit(f"error: {path} not found (was this run made before W1.4, "
                 f"or without --summary?)")
    rows: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[(int(row["frame"]), int(row["face_idx"]))] = row
    return rows


def _frame_diag(video: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return math.hypot(w, h) or 1.0


def _rank_auc(pos: list[float], neg: list[float]) -> float | None:
    """AUC via the rank-sum identity; None when a class is empty.

    ``pos`` should score LOWER than ``neg`` (offscreen inout < onscreen
    inout), so we compute AUC of (neg > pos)."""
    if not pos or not neg:
        return None
    ranked = sorted((v, 1) for v in neg) + sorted((v, 0) for v in pos)
    ranked.sort(key=lambda t: t[0])
    i = 0
    n = len(ranked)
    ranks = [0.0] * n
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    pos_rank_sum = sum(r for r, (_v, is_neg) in zip(ranks, ranked) if is_neg)
    n_neg, n_pos = len(neg), len(pos)
    return (pos_rank_sum - n_neg * (n_neg + 1) / 2.0) / (n_neg * n_pos)


def _score(video: Path, tag: str, radius: float) -> dict:
    stem = video.stem
    labels = _load_labels(video)
    rows = _load_gaze_rows(EVAL_DIR / "runs" / tag, stem)
    diag = _frame_diag(video)

    errs: list[float] = []
    hits = 0
    scored = 0
    on_inout: list[float] = []
    off_inout: list[float] = []
    per_pid: dict = defaultdict(list)

    for frame_str, per_face in labels.items():
        frame = int(frame_str)
        for tid_str, label in per_face.items():
            tid = int(tid_str)
            row = rows.get((frame, tid))
            if row is None or label in ("uncertain", "skip"):
                continue
            inout = float(row["inout_score"]) if row["inout_score"] else None
            if label == "offscreen":
                if inout is not None:
                    off_inout.append(inout)
                continue
            if inout is not None:
                on_inout.append(inout)
            dx = float(row["ray_end_x"]) - label["x"]
            dy = float(row["ray_end_y"]) - label["y"]
            err = math.hypot(dx, dy)
            errs.append(err)
            per_pid[tid].append(err)
            scored += 1
            if err <= radius:
                hits += 1

    if not errs:
        sys.exit("error: no scorable (on-screen) labels matched gaze rows")

    errs.sort()

    def pct(p):
        return errs[min(len(errs) - 1, int(p / 100 * len(errs)))]

    result = {
        "tag": tag,
        "scored_points": scored,
        "endpoint_px_mean": sum(errs) / len(errs),
        "endpoint_px_median": pct(50),
        "endpoint_px_p95": pct(95),
        "endpoint_norm_mean": sum(errs) / len(errs) / diag,
        "hit_rate": hits / scored,
        "hit_radius_px": radius,
        "offscreen_auc": _rank_auc(off_inout, on_inout),
        "offscreen_labels": len(off_inout),
        "per_participant_mean_px": {
            f"P{tid}": sum(v) / len(v) for tid, v in sorted(per_pid.items())},
    }
    out = EVAL_DIR / "runs" / tag / f"{stem}_score.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nscore -> {out}")
    return result


def _table(video: Path) -> None:
    stem = video.stem
    runs = sorted((EVAL_DIR / "runs").glob(f"*/{stem}_score.json"))
    if not runs:
        sys.exit("no scored runs found -- run 'score' first")
    cols = ["tag", "scored_points", "endpoint_px_mean", "endpoint_px_median",
            "endpoint_px_p95", "hit_rate", "offscreen_auc"]
    print("  ".join(f"{c:>20}" for c in cols))
    for path in runs:
        d = json.loads(path.read_text())
        cells = []
        for c in cols:
            v = d.get(c)
            cells.append(f"{v:>20.3f}" if isinstance(v, float)
                         else f"{str(v):>20}")
        print("  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run", "score", "table"))
    ap.add_argument("video")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--radius", type=float, default=80.0,
                    help="hit-rate radius in px (default 80)")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    video = Path(args.video)
    if args.mode == "run":
        _run(video, args.tag, passthrough)
    elif args.mode == "score":
        _score(video, args.tag, args.radius)
    else:
        _table(video)


if __name__ == "__main__":
    main()
