"""
ObjectDetection/object_detection.py — Object detection wrappers.

Responsibilities
----------------
- YOLOEVPDetector: wraps YOLOE + a Visual Prompt file (.vp.json) with the
  same interface as a YOLO model.
- ObjectPersistenceCache: keeps detections alive for N frames after a miss
  (handles momentary occlusion / YOLO misses).
- parse_dets: converts raw YOLO result boxes to a list of object dicts.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from ObjectDetection.detection import Detection

# ══════════════════════════════════════════════════════════════════════════════
# YOLOE Visual-Prompt detector
# ══════════════════════════════════════════════════════════════════════════════

class YOLOEVPDetector:
    """
    Wraps YOLOE + a Visual Prompt file (.vp.json) to provide the same
    __call__ / .names interface as a YOLO model inside process_frame().
    """

    def __init__(self, model_path: str, vp_file: str, device: str | None = None):
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        self._device = device
        data = json.loads(Path(vp_file).read_text())
        refs  = data.get("references", [])
        if not refs:
            raise ValueError(f"VP file has no references: {vp_file}")
        ref  = refs[0]
        anns = ref.get("annotations", [])
        if not anns:
            raise ValueError(f"First reference in VP file has no annotations: {vp_file}")

        self._refer_image = str(ref["image"])
        self._visual_prompts = {
            "bboxes": np.array([a["bbox"]   for a in anns], dtype=float),
            "cls":    np.array([a["cls_id"] for a in anns], dtype=int),
        }
        self._predictor_cls = YOLOEVPSegPredictor
        self.names = {c["id"]: c["name"] for c in data.get("classes", [])}
        self._model = YOLOE(model_path)
        if device and device != "cpu":
            try:
                self._model.to(device)
            except Exception:
                pass
        self._initialized = False

    def set_classes(self, classes):
        try:
            self._model.set_classes(classes)
        except AttributeError:
            pass

    def __call__(self, frame, conf: float = 0.35, classes=None, verbose: bool = False):
        if not self._initialized:
            results = self._model.predict(
                frame,
                refer_image=self._refer_image,
                visual_prompts=self._visual_prompts,
                predictor=self._predictor_cls,
                conf=conf, verbose=verbose,
            )
            self._initialized = True
        else:
            results = self._model.predict(frame, conf=conf, verbose=verbose)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Object persistence cache
# ══════════════════════════════════════════════════════════════════════════════

class ObjectPersistenceCache:
    """
    Keeps YOLO-detected objects alive for up to `max_age` frames after they
    disappear from the detector (momentary occlusion, YOLO miss).

    On each call to ``update(current_dets)`` the cache:
      1. Matches incoming dets to existing slots by class + IoU.
      2. Ages unmatched slots; removes expired ones.
      3. Returns fresh + ghost objects combined (ghost objects have ``'_ghost': True``).
    """

    def __init__(self, max_age: int = 15, iou_threshold: float = 0.30):
        self.max_age       = max_age
        self.iou_threshold = iou_threshold
        self._slots: dict  = {}
        self._nid          = 0

    @staticmethod
    def _iou(a, b):
        ix1 = max(a['x1'], b['x1']); iy1 = max(a['y1'], b['y1'])
        ix2 = min(a['x2'], b['x2']); iy2 = min(a['y2'], b['y2'])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua    = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
        ub    = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def update(self, current_dets: list) -> list:
        used_slots    = set()
        unmatched_new = list(range(len(current_dets)))

        for sid, slot in self._slots.items():
            best_di, best_iou = None, self.iou_threshold
            for di in unmatched_new:
                if current_dets[di]['class_name'] != slot['det']['class_name']:
                    continue
                iou = self._iou(current_dets[di], slot['det'])
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_di is not None:
                det = dataclasses.replace(current_dets[best_di], ghost=False)
                slot['det'] = det
                slot['age'] = 0
                used_slots.add(sid)
                unmatched_new.remove(best_di)

        for di in unmatched_new:
            sid = self._nid; self._nid += 1
            det = dataclasses.replace(current_dets[di], ghost=False)
            self._slots[sid] = {'det': det, 'age': 0}
            used_slots.add(sid)

        for sid in list(self._slots):
            if sid not in used_slots:
                slot = self._slots[sid]
                slot['age'] += 1
                slot['det'] = dataclasses.replace(slot['det'], ghost=True)
                if slot['age'] > self.max_age:
                    del self._slots[sid]

        return [s['det'] for s in self._slots.values()]


# ══════════════════════════════════════════════════════════════════════════════
# Detection parsing
# ══════════════════════════════════════════════════════════════════════════════

def add_arguments(parser) -> None:
    """Register object-detection CLI flags onto *parser*."""
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Object Detection Model, defaults to yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Object detection confidence threshold, defaults to 0.35")
    parser.add_argument("--classes", nargs="+", default=[],
                        help="Specified YOLO Object Detection Classes Prompt")
    parser.add_argument("--blacklist", nargs="+", default=[],
                        help="Specified YOLO Object Detection Classes Blacklist")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Frames between object detection. Higher = faster but less accurate. "
                             "(Defaults to 1, i.e. process every frame)")
    parser.add_argument("--detect-scale", type=float, default=1.0,
                        help="Detection scale for Object Recognition")
    parser.add_argument("--vp-file", default=None,
                        help="Path to visual prompt file for use with YOLOE object detection models")
    parser.add_argument("--vp-model", default="yoloe-26l-seg.pt",
                        help="YOLOE model to use alongside visual prompting for object detection")
    parser.add_argument("--obj-persistence", type=int, default=0, metavar="N",
                        help="Dead-reckon object bboxes for N frames after they disappear (default 0).")


def parse_dets(results, names, conf_thr, blacklist):
    """Convert raw YOLO result boxes to a list of object dicts.

    Each dict has keys: class_name, cls_id, conf, x1, y1, x2, y2.
    Detections below conf_thr or whose class is in blacklist are dropped.
    """
    if not results or results[0].boxes is None:
        return []
    dets = []
    for b in results[0].boxes:
        c, cls = float(b.conf[0]), int(b.cls[0])
        if c >= conf_thr and names[cls].lower() not in blacklist:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            dets.append(Detection(class_name=names[cls], cls_id=cls, conf=c,
                                  x1=x1, y1=y1, x2=x2, y2=y2))
    return dets
