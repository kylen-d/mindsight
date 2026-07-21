#!/usr/bin/env python
"""eval_annotate.py -- click-through gaze-target labeler (v1.1 W3.0).

Dev tool, not a GUI feature.  Steps through a stratified sample of frames and
records, per participant, where they are ACTUALLY looking -- the ground truth
the eval harness (scripts/eval_gaze.py) scores pipeline output against.

Workflow
--------
1. Produce a reference run first so faces carry stable track IDs::

       python scripts/eval_gaze.py run VIDEO --tag baseline

2. Annotate (reads the reference run's {stem}_gaze.csv to overlay each
   participant's labeled origin)::

       python scripts/eval_annotate.py VIDEO --run eval_data/runs/baseline

3. Score any run against the labels::

       python scripts/eval_gaze.py score VIDEO --run eval_data/runs/<tag>

Controls
--------
    left-click   label the CURRENT participant's gaze target at the cursor
    o            current participant is looking OFF-SCREEN
    u            uncertain -- excluded from scoring
    s            skip this participant on this frame
    n / space    next sampled frame (auto-advances when all faces labeled)
    b            back one sampled frame
    q / ESC      save + quit

Labels land in eval_data/{stem}_labels.json (eval_data/ is untracked --
participant imagery and derived data stay local to this machine).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval_data"

_COLOURS = [(80, 200, 255), (100, 255, 130), (255, 150, 50), (200, 149, 255)]


def _load_gaze_csv(run_dir: Path, stem: str) -> dict:
    """{frame: [(tid, origin_x, origin_y), ...]} from a run's gaze stream."""
    path = run_dir / f"{stem}_gaze.csv"
    if not path.is_file():
        sys.exit(f"error: {path} not found -- produce it first with "
                 f"'python scripts/eval_gaze.py run VIDEO --tag <tag>'")
    per_frame: dict = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            per_frame[int(row["frame"])].append(
                (int(row["face_idx"]),
                 float(row["origin_x"]), float(row["origin_y"])))
    return per_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--run", required=True,
                    help="reference run dir holding {stem}_gaze.csv")
    ap.add_argument("--every", type=int, default=10,
                    help="sample every Nth frame (default 10)")
    ap.add_argument("--out", default=None,
                    help="labels JSON (default eval_data/{stem}_labels.json)")
    args = ap.parse_args()

    video = Path(args.video)
    stem = video.stem
    out_path = Path(args.out) if args.out else EVAL_DIR / f"{stem}_labels.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gaze = _load_gaze_csv(Path(args.run), stem)

    labels: dict = {}
    if out_path.is_file():
        labels = json.loads(out_path.read_text()).get("labels", {})
        print(f"resuming: {len(labels)} frames already labeled")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"error: cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = [f for f in range(0, total, max(1, args.every)) if f in gaze]
    if not sample:
        sys.exit("error: no sampled frames carry gaze rows; check --every")
    print(f"{len(sample)} sampled frames, {total} total")

    state = {"click": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "eval_annotate  (q saves + quits)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        out_path.write_text(json.dumps(
            {"video": video.name, "every": args.every, "labels": labels},
            indent=1, sort_keys=True))
        print(f"saved {len(labels)} labeled frames -> {out_path}")

    si = 0
    while 0 <= si < len(sample):
        frame_no = sample[si]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            si += 1
            continue
        faces = sorted(gaze[frame_no])
        frame_labels = labels.setdefault(str(frame_no), {})

        # first unlabeled participant on this frame
        pending = [tid for tid, _x, _y in faces
                   if str(tid) not in frame_labels]
        advance = not pending
        current = pending[0] if pending else None

        disp = frame.copy()
        for i, (tid, ox, oy) in enumerate(faces):
            col = _COLOURS[i % len(_COLOURS)]
            cv2.circle(disp, (int(ox), int(oy)), 6, col, -1)
            cv2.putText(disp, f"P{tid}", (int(ox) + 8, int(oy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            lab = frame_labels.get(str(tid))
            if isinstance(lab, dict):
                cv2.drawMarker(disp, (int(lab["x"]), int(lab["y"])), col,
                               cv2.MARKER_CROSS, 18, 2)
        hud = (f"[{si + 1}/{len(sample)}] frame {frame_no}   "
               + (f"label P{current}: click target | o=offscreen | "
                  f"u=uncertain | s=skip" if current is not None
                  else "all labeled -- n=next"))
        cv2.putText(disp, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)
        cv2.imshow(win, disp)

        if advance:
            key = cv2.waitKey(400) & 0xFF
            if key in (255, ord("n"), ord(" ")):
                si += 1
                continue
        else:
            key = cv2.waitKey(30) & 0xFF

        if state["click"] is not None and current is not None:
            x, y = state["click"]
            frame_labels[str(current)] = {"x": int(x), "y": int(y)}
            state["click"] = None
            continue
        if key in (ord("q"), 27):
            break
        elif key == ord("o") and current is not None:
            frame_labels[str(current)] = "offscreen"
        elif key == ord("u") and current is not None:
            frame_labels[str(current)] = "uncertain"
        elif key == ord("s") and current is not None:
            frame_labels[str(current)] = "skip"
        elif key in (ord("n"), ord(" ")):
            si += 1
        elif key == ord("b"):
            si -= 1

    save()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
