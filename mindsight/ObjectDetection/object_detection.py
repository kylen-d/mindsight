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

from mindsight.ObjectDetection.detection import Detection

# ══════════════════════════════════════════════════════════════════════════════
# YOLOE Visual-Prompt detector
# ══════════════════════════════════════════════════════════════════════════════

def parse_vp_references(data: dict) -> list[dict]:
    """Parse EVERY annotated reference from VP-file *data* (v1.1 W3.7).

    Returns a list of ``{'image': str, 'bboxes': ndarray, 'cls': ndarray}``
    (cls carries the VP file's class ids).  References without annotations
    are skipped.  1.0 used only ``references[0]`` and silently ignored the
    rest -- the extra reference images study leads added did nothing.
    """
    refs = []
    for ref in data.get("references", []):
        anns = ref.get("annotations", [])
        if not anns:
            continue
        refs.append({
            "image": str(ref["image"]),
            "bboxes": np.array([a["bbox"] for a in anns], dtype=float),
            "cls": np.array([a["cls_id"] for a in anns], dtype=int),
        })
    if not refs:
        raise ValueError("VP file has no annotated references")
    return refs


def prime_yoloe_multi_reference(model, references, predictor_cls,
                                class_names=None, log=print):
    """Prime YOLOE class embeddings from SEVERAL reference images.

    The pinned ultralytics ``get_vpe`` accepts a single image, so each
    reference primes individually (a predict on the reference image itself)
    and the resulting class-embedding tables (``model.model.pe``) are
    mean-pooled per class across the references that annotate it, then
    re-normalized and installed as the final class table.  Classes are the
    sorted union of every reference's ids; the returned dict maps the
    model's output class index to the VP class id.
    """
    import torch
    import torch.nn.functional as F

    union = sorted({int(c) for ref in references for c in ref["cls"]})
    sums: dict[int, "torch.Tensor"] = {}
    counts: dict[int, int] = {g: 0 for g in union}

    for ref in references:
        present = sorted({int(c) for c in ref["cls"]})
        remap = {g: i for i, g in enumerate(present)}
        compact = np.array([remap[int(c)] for c in ref["cls"]], dtype=int)
        model.predict(ref["image"],
                      refer_image=ref["image"],
                      visual_prompts={"bboxes": ref["bboxes"], "cls": compact},
                      predictor=predictor_cls,
                      conf=0.25, verbose=False)
        pe = model.model.pe.detach().float().cpu()      # (1, len(present), D)
        for g, i in remap.items():
            vec = pe[0, i]
            sums[g] = vec if g not in sums else sums[g] + vec
            counts[g] += 1

    pooled = torch.stack([F.normalize(sums[g] / counts[g], dim=-1)
                          for g in union]).unsqueeze(0)
    names = ([str(class_names.get(g, f"object{g}")) for g in union]
             if class_names else [f"object{g}" for g in union])
    device = next(model.model.parameters()).device
    model.model.set_classes(names, pooled.to(device))
    # The priming predicts leave ``model.predictor`` caching the LAST
    # reference's class table (the predictor's model wrapper copies
    # ``names`` at setup), and seg NMS sizes its mask-coefficient split
    # from ``len(predictor.model.names)``.  When the last reference
    # annotates only a SUBSET of the union, that stale count mis-splits
    # the head output and ``process_mask`` crashes on the first real
    # frame ("mat1 and mat2 shapes cannot be multiplied (Nx33 and
    # 32x16640)" -- 33 = 32 coefficients + 1 missing class).  Sync the
    # cached copy to the pooled table.
    predictor = getattr(model, "predictor", None)
    pred_model = getattr(predictor, "model", None)
    if pred_model is not None:
        pred_model.names = {i: n for i, n in enumerate(names)}
    log(f"YOLOE visual prompts: pooled {len(references)} reference image(s), "
        f"{len(union)} class(es)")
    return {idx: g for idx, g in enumerate(union)}


class YOLOEVPDetector:
    """
    Wraps YOLOE + a Visual Prompt file (.vp.json) to provide the same
    __call__ / .names interface as a YOLO model inside process_frame().

    Single-reference files use the 1.0 native priming path unchanged
    (byte-identical detections); files with several annotated references
    mean-pool the per-reference class embeddings (v1.1 W3.7).
    """

    def __init__(self, model_path: str, vp_file: str, device: str | None = None):
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        self._device = device
        data = json.loads(Path(vp_file).read_text())
        self._references = parse_vp_references(data)

        # Legacy single-reference fields (native priming path).
        ref = self._references[0]
        self._refer_image = ref["image"]
        self._visual_prompts = {"bboxes": ref["bboxes"], "cls": ref["cls"]}
        self._predictor_cls = YOLOEVPSegPredictor
        self._vp_names = {c["id"]: c["name"] for c in data.get("classes", [])}
        self.names = dict(self._vp_names)
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
            if len(self._references) > 1:
                idx_to_id = prime_yoloe_multi_reference(
                    self._model, self._references, self._predictor_cls,
                    class_names=self._vp_names)
                self.names = {idx: self._vp_names.get(g, f"object{g}")
                              for idx, g in idx_to_id.items()}
                self._initialized = True
                results = self._model.predict(frame, conf=conf, verbose=verbose)
            else:
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
