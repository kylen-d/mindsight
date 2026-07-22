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
# VP file format (v1 plain, v2 adds condition tags -- v1.3.1 item 3)
# ══════════════════════════════════════════════════════════════════════════════

VP_FORMAT_VERSION_MAX = 2


def ensure_vp_version(data: dict, source: str = "VP file") -> int:
    """Validate the payload's ``version`` key (absent = 1) and return it."""
    raw = data.get("version", 1)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} has an unreadable format version: {raw!r}") from None
    if v > VP_FORMAT_VERSION_MAX:
        raise ValueError(
            f"{source} uses VP format version {v}; this MindSight "
            f"understands up to version {VP_FORMAT_VERSION_MAX}. Update "
            "MindSight to open it.")
    return v


def load_vp_data(vp_file) -> dict:
    """Read a ``.vp.json`` with the format-version guard applied."""
    data = json.loads(Path(vp_file).read_text())
    ensure_vp_version(data, source=Path(vp_file).name)
    return data


def build_vp_payload(classes: list, references: list,
                     conditions: list | None = None) -> dict:
    """The serializable VP payload for *classes*/*references*.

    Emits version 1 -- byte-stable with pre-1.3.1 files -- unless any class
    carries condition tags or a *conditions* vocabulary is given, in which
    case version 2 adds a top-level ``conditions`` vocabulary (declared
    entries first, then any class tags missing from it).  Empty per-class
    ``conditions`` keys are stripped either way.
    """
    cleaned = [{k: v for k, v in c.items() if k != "conditions" or v}
               for c in classes]
    vocab = [str(t) for t in (conditions or [])]
    if not vocab and not any(c.get("conditions") for c in cleaned):
        return {"version": 1, "classes": cleaned, "references": references}
    for c in cleaned:
        for t in c.get("conditions", []):
            if t not in vocab:
                vocab.append(str(t))
    return {"version": 2, "conditions": vocab, "classes": cleaned,
            "references": references}


def vp_has_conditions(data: dict) -> bool:
    """True when any class is condition-tagged (the v2 feature in use)."""
    return any(c.get("conditions") for c in data.get("classes", []))


def vp_declared_conditions(data: dict) -> list[str]:
    """The condition vocabulary: declared list + any undeclared class tags."""
    declared = [str(t) for t in data.get("conditions", [])]
    for c in data.get("classes", []):
        for t in c.get("conditions") or []:
            if t not in declared:
                declared.append(str(t))
    return declared


def filter_vp_for_conditions(data: dict, tags) -> dict:
    """Effective VP payload for a video carrying condition *tags*.

    Classes with no condition tags are always active; a conditioned class is
    active when it shares at least one tag with the video (multi-tag videos
    therefore get the union).  A VP without condition tags is returned
    UNCHANGED (same object -- the v1 fast path).  Kept classes are
    renumbered contiguously, annotations remapped, and references left with
    no annotations dropped; class NAMES carry through, so downstream CSVs
    are unaffected by the renumbering.
    """
    if not vp_has_conditions(data):
        return data
    tag_set = {str(t) for t in (tags or [])}
    keep = [c for c in data.get("classes", [])
            if not c.get("conditions") or tag_set & set(c["conditions"])]
    id_map = {c["id"]: i for i, c in enumerate(keep)}
    classes = [{**{k: v for k, v in c.items() if k != "conditions"},
                "id": id_map[c["id"]]} for c in keep]
    refs = []
    for ref in data.get("references", []):
        anns = [{**a, "cls_id": id_map[a["cls_id"]]}
                for a in ref.get("annotations", [])
                if a.get("cls_id") in id_map]
        if anns:
            refs.append({**ref, "annotations": anns})
    return {"version": 1, "classes": classes, "references": refs}


def vp_content_digest(vp_file) -> str:
    """sha256 of the VP file's bytes (run-identity input since v1.3.1)."""
    import hashlib
    return hashlib.sha256(Path(vp_file).read_bytes()).hexdigest()


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

    def __init__(self, model_path: str, vp_file: str | None,
                 device: str | None = None, vp_data: dict | None = None):
        self._device = device
        # Parse the prompt BEFORE any model construction so a bad file (or
        # unsupported format version) fails with the format error, not a
        # model-load error.  *vp_data* takes an already-loaded (possibly
        # condition-filtered) payload -- no temp files needed (v1.3.1).
        data = vp_data if vp_data is not None else load_vp_data(vp_file)
        self._references = parse_vp_references(data)

        # Legacy single-reference fields (native priming path).
        ref = self._references[0]
        self._refer_image = ref["image"]
        self._visual_prompts = {"bboxes": ref["bboxes"], "cls": ref["cls"]}
        self._vp_names = {c["id"]: c["name"] for c in data.get("classes", [])}
        self.names = dict(self._vp_names)

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        self._predictor_cls = YOLOEVPSegPredictor
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
