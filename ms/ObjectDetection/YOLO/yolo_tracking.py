"""
YOLOv8 Object Detection
-----------------------
Supports three input modes:
  - Webcam / video device
  - Video file
  - Single image

Requirements:
    pip install ultralytics opencv-python

Usage:
    # Webcam (default)
    python yolov8_detector.py

    # Video file
    python yolov8_detector.py --source path/to/video.mp4

    # Image file
    python yolov8_detector.py --source path/to/image.jpg

    # Different model size (n / s / m / l / x)
    python yolov8_detector.py --model yolov8m.pt

    # Save output
    python yolov8_detector.py --source video.mp4 --save

    # Filter to specific classes (COCO class names)
    python yolov8_detector.py --classes person car dog

    # Blacklist specific classes (suppress from output)
    python yolov8_detector.py --blacklist person chair

    # Adjust confidence threshold
    python yolov8_detector.py --conf 0.4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from ms.constants import PALETTE_BGR as PALETTE  # noqa: F401
from ms.constants import get_colour

# ── Blacklist — class names to suppress from detection output ─────────────────
# Add any COCO class names here that you want permanently ignored.
# These are also merged with any names passed via --blacklist at runtime.
# Full COCO class list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
BLACKLISTED_CLASSES: set[str] = {
    # "person",
    # "chair",
    # "dining table",
}


def draw_detections(frame, results, names, conf_threshold: float, blacklist: set[str]):
    """Draw bounding boxes, labels and confidence scores on *frame* in-place.

    Returns the annotated frame and the number of visible (non-blacklisted) detections.
    """
    boxes = results[0].boxes
    if boxes is None:
        return frame, 0

    count = 0
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        class_name = names[cls_id]

        # Skip blacklisted classes (case-insensitive)
        if class_name.lower() in blacklist:
            continue

        count += 1
        label = f"{class_name} {conf:.2f}"
        colour = get_colour(cls_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            colour,
            -1,
        )

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame, count


def overlay_stats(frame, fps: float, det_count: int):
    """Overlay FPS and detection count in the top-left corner."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (200, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS : {fps:5.1f}",       (8, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Dets: {det_count:4d}",    (8, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame


def resolve_classes(model, requested: list[str] | None) -> list[int] | None:
    """Convert class name strings to integer IDs understood by YOLO."""
    if not requested:
        return None
    name_to_id = {v.lower(): k for k, v in model.names.items()}
    ids = []
    for name in requested:
        if name.lower() in name_to_id:
            ids.append(name_to_id[name.lower()])
        else:
            print(f"[WARNING] Class '{name}' not found in model — skipping.")
    return ids if ids else None


# ── Main detection routines ────────────────────────────────────────────────────

def run_on_image(model, source: str, conf: float, class_ids, blacklist: set[str], save: bool):
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    results = model(frame, conf=conf, classes=class_ids, verbose=False)
    frame, det_count = draw_detections(frame, results, model.names, conf, blacklist)

    print(f"Detections: {det_count}")

    if save:
        out_path = Path(source).stem + "_detected.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved → {out_path}")

    cv2.imshow("YOLOv8 Detection", frame)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(model, source, conf: float, class_ids, blacklist: set[str], save: bool):
    """Run detection on a webcam index (int) or video file path (str)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Optional: write output video
    writer = None
    if save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = "detected_output.mp4" if isinstance(source, int) else Path(source).stem + "_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_name, fourcc, fps_src, (w, h))
        print(f"Saving output → {out_name}")

    fps = 0.0
    frame_times = []

    print("Running — press Q to quit.")
    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, classes=class_ids, verbose=False)
        frame, det_count = draw_detections(frame, results, model.names, conf, blacklist)

        # Rolling FPS average over last 30 frames
        frame_times.append(time.perf_counter() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        overlay_stats(frame, fps, det_count)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "--source", default="0",
        help="Input source: '0' for webcam, path to video or image file (default: 0)"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt). "
             "Options: yolov8n/s/m/l/x.pt — downloads automatically on first run."
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Minimum confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Filter detections by class name, e.g. --classes person car dog"
    )
    parser.add_argument(
        "--blacklist", nargs="+", default=[],
        help="Class names to suppress, e.g. --blacklist person chair 'dining table'"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the annotated output to disk"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    model.set_classes([ "bottle", "banana", "plate", "recipe", "paper sheet", "knife", "plastic cutting board", "dark blue wastebin", ""])

    class_ids = resolve_classes(model, args.classes)
    if class_ids:
        print(f"Filtering to classes: {[model.names[i] for i in class_ids]}")

    # Merge hardcoded blacklist with any passed via --blacklist
    blacklist = set(BLACKLISTED_CLASSES) | {name.lower() for name in args.blacklist}
    if blacklist:
        print(f"Blacklisting classes: {sorted(blacklist)}")

    # Determine source type
    source = args.source
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if Path(source).suffix.lower() in image_exts and Path(source).exists():
        run_on_image(model, source, args.conf, class_ids, blacklist, args.save)
    else:
        # Webcam index or video file
        try:
            source = int(source)   # webcam index
        except ValueError:
            pass                   # keep as string (video file path)
        run_on_video(model, source, args.conf, class_ids, blacklist, args.save)


if __name__ == "__main__":
    main()
