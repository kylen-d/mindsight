"""
yoloe_folder.py
YOLOE-26 Text-Prompted Batch Image Detector
--------------------------------------------
Runs YOLOE-26 open-vocabulary detection on every image inside a folder.
Class names are supplied as free-form text prompts — no re-training needed.

Requirements:
    pip install ultralytics opencv-python

Usage:
    # Detect cats and remote controls in all images under ./photos/
    python yoloe_folder.py --folder photos/ --prompts cat "remote control"

    # Use the medium model, lower confidence, save to a custom output dir
    python yoloe_folder.py --folder photos/ --prompts person dog \\
        --model yoloe-26m.pt --conf 0.25 --out results/

    # Dry-run: list matched images without running inference
    python yoloe_folder.py --folder photos/ --prompts cat --dry-run
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLOE

# ── Supported image extensions ────────────────────────────────────────────────
from ms.constants import IMAGE_EXTS

# ── Colour palette (one colour per prompt index, cycles for > 20 classes) ────
PALETTE = [
    (56, 56, 255),   (151, 157, 255), (31, 112, 255),  (29, 178, 255),
    (49, 210, 207),  (10, 249, 72),   (23, 204, 146),  (134, 219, 61),
    (52, 147, 26),   (187, 212, 0),   (168, 153, 44),  (255, 194, 0),
    (147, 69, 52),   (255, 115, 100), (236, 24, 0),    (255, 56, 132),
    (133, 0, 82),    (255, 56, 203),  (200, 149, 255), (199, 55, 255),
]


def get_colour(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(frame, results, names: list[str], conf_threshold: float) -> tuple:
    """Draw bounding boxes and labels on *frame* in-place.

    Returns (annotated_frame, detection_count, list_of_label_strings).
    """
    boxes = results[0].boxes
    if boxes is None:
        return frame, 0, []

    labels_found = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id    = int(box.cls[0])
        cls_name  = names[cls_id] if cls_id < len(names) else str(cls_id)
        label     = f"{cls_name} {conf:.2f}"
        colour    = get_colour(cls_id)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), colour, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        labels_found.append(cls_name)

    return frame, len(labels_found), labels_found


# ── Core batch runner ─────────────────────────────────────────────────────────

def run_folder(
    model: YOLOE,
    image_paths: list[Path],
    prompts: list[str],
    conf: float,
    out_dir: Path,
) -> list[dict]:
    """Run inference on every image; write annotated copies to *out_dir*.

    Returns a list of per-image result dicts for the summary table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    total_t = 0.0

    for i, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [{i}/{len(image_paths)}] SKIP  {img_path.name}  (unreadable)")
            continue

        t0      = time.perf_counter()
        results = model.predict(frame, conf=conf, verbose=False)
        elapsed = time.perf_counter() - t0
        total_t += elapsed

        frame, det_count, labels = draw_detections(frame, results, prompts, conf)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), frame)

        # Count per-class detections
        class_counts = {p: labels.count(p) for p in prompts if labels.count(p) > 0}
        class_str    = ", ".join(f"{v}× {k}" for k, v in class_counts.items()) or "—"

        print(
            f"  [{i:>{len(str(len(image_paths)))}}/{len(image_paths)}]"
            f"  {img_path.name:<40}"
            f"  {det_count:>3} det   {elapsed*1000:>6.1f} ms   {class_str}"
        )

        summary.append({
            "file":        img_path.name,
            "detections":  det_count,
            "ms":          elapsed * 1000,
            "classes":     class_counts,
        })

    if summary:
        avg_ms   = total_t * 1000 / len(summary)
        total_d  = sum(r["detections"] for r in summary)
        print()
        print(f"  Processed {len(summary)} image(s)  |  "
              f"{total_d} total detections  |  {avg_ms:.1f} ms avg per image")
        print(f"  Annotated images saved → {out_dir}/")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOE-26 text-prompted batch image detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder", required=True,
        help="Path to the folder of input images.",
    )
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        metavar="CLASS",
        help="Text class names to detect (space-separated). "
             'Quote multi-word names: --prompts person "coffee cup" laptop',
    )
    parser.add_argument(
        "--model", default="yoloe-26s.pt",
        help="YOLOE-26 weights file (default: yoloe-26s.pt). "
             "Options: yoloe-26s.pt / yoloe-26m.pt / yoloe-26l.pt — "
             "downloaded automatically on first run.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Minimum detection confidence (default: 0.3).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for annotated images "
             "(default: <folder>_detected/ next to the input folder).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List matched images and exit without running inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {folder}")

    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"[ERROR] No images found in {folder} "
                         f"(looked for {', '.join(sorted(IMAGE_EXTS))})")

    out_dir = Path(args.out) if args.out else folder.parent / (folder.name + "_detected")

    print("=" * 60)
    print("  YOLOE-26 — folder batch detector")
    print("=" * 60)
    print(f"  Folder  : {folder}  ({len(image_paths)} image(s))")
    print(f"  Prompts : {args.prompts}")
    print(f"  Model   : {args.model}")
    print(f"  Conf    : {args.conf}")
    print(f"  Output  : {out_dir}")
    print()

    if args.dry_run:
        print("  DRY RUN — matched images:")
        for p in image_paths:
            print(f"    {p}")
        return

    # Load model and set text-prompted classes
    print(f"Loading {args.model} …")
    model = YOLOE(args.model)
    model.set_classes(args.prompts)
    print(f"Classes set: {args.prompts}\n")

    run_folder(model, image_paths, args.prompts, args.conf, out_dir)


if __name__ == "__main__":
    main()
