"""
auto_crop.py -- LP1: derive a crop rectangle from object detections.

The Crop & Adjust tool can pre-place its crop rectangle automatically: run a
YOLOE detector on the video's middle frame -- prompted either by a TEXT LIST
of object names ("person, dining table") or by the study's own VISUAL PROMPT
file -- and fit the rectangle around everything found, plus a padding
tolerance.  The result lands in the SAME draggable rubber band as a manual
crop, so the user always reviews/adjusts before anything is re-encoded.
"""

from __future__ import annotations


def union_rect(boxes, pad, frame_w: int, frame_h: int):
    """Fit one rectangle around xyxy *boxes* + *pad*, clamped to the frame.

    ``pad`` is a uniform int or a ``(left, top, right, bottom)`` tuple;
    negative values shrink the fit inside the detections (eyes-on D3).
    Returns ``(x, y, w, h)`` in pixels, or None when there is nothing to fit
    (no boxes, or a degenerate result).
    """
    if not boxes:
        return None
    if isinstance(pad, (int, float)):
        pl = pt = pr = pb = int(pad)
    else:
        pl, pt, pr, pb = (int(v) for v in pad)
    x1 = max(0, int(min(b[0] for b in boxes) - pl))
    y1 = max(0, int(min(b[1] for b in boxes) - pt))
    x2 = min(int(frame_w), int(max(b[2] for b in boxes) + pr))
    y2 = min(int(frame_h), int(max(b[3] for b in boxes) + pb))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def detect_boxes(detector, frame, *, conf: float = 0.25) -> list[tuple]:
    """Run one prediction on *frame*; return xyxy tuples (may be empty)."""
    results = detector(frame, conf=conf, verbose=False)
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    return [tuple(float(v) for v in b) for b in boxes.xyxy.tolist()]


def load_landmark_detector(mode: str, *, classes=None, vp_file=None,
                           vp_model: str = "yoloe-26l-seg.pt",
                           device: str = "auto"):
    """Build the landmark detector for auto-crop.

    ``mode`` is ``"text"`` (YOLOE prompted with *classes*, a list of object
    names) or ``"vp"`` (YOLOE + the study's visual prompt file).  Reuses the
    pipeline's own ``create_yolo_detector`` so weight resolution, device
    fallback, and prompt wiring stay in one place.
    """
    from mindsight.ObjectDetection.model_factory import create_yolo_detector
    if mode == "vp":
        det, _, _ = create_yolo_detector(vp_file=vp_file, vp_model=vp_model,
                                         device=device)
    else:
        det, _, _ = create_yolo_detector(model_path=vp_model,
                                         classes=list(classes or []),
                                         device=device)
    return det
