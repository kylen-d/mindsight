"""GUI/region_suggest.py -- FastSAM click-to-suggest for the VP Builder.

v1.1 W3Z item 8; hybrid canvas since v1.3.1 item 2.  In the VP Builder a
click on an empty spot of the reference image runs a FastSAM point-prompt
and proposes the small set of regions containing that point (typically 1-3
nested candidates: the part, the whole object, sometimes a surrounding
surface).  The user accepts one into the active class -- the vp.json format
is untouched, so the runtime VP path needs zero changes.

The FastSAM-s weight is manifest-managed (Weights/SAM/FastSAM-s.pt,
non-required); :func:`fastsam_path` reports None until it is downloaded
from the Models tab, and the "Suggest on click" checkbox explains that in
plain English.

No Qt in this module -- the suggester is a plain object so it stays
testable headless and callable from a worker thread.
"""
from __future__ import annotations

from mindsight.weights import entry_dest, find_entry

FASTSAM_BACKEND = "SAM"
FASTSAM_FILENAME = "FastSAM-s.pt"

_MIN_AREA_FRAC = 0.0005      # drop speck masks (labels/screws/etc.)
_MAX_AREA_FRAC = 0.90        # drop whole-image "background" masks
_DEDUP_IOU = 0.90            # near-identical candidates collapse to one
MAX_SUGGESTIONS = 4


def fastsam_path():
    """Local FastSAM weight path, or None when not yet downloaded."""
    entry = find_entry(FASTSAM_FILENAME, backend=FASTSAM_BACKEND)
    if entry is None:
        return None
    dest = entry_dest(entry)
    return dest if dest.exists() else None


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


class RegionSuggester:
    """Lazy FastSAM point-prompt suggester.

    The model loads on the first :meth:`suggest` call and is kept for the
    session (~24 MB weight; ~0.4 s/click on Apple MPS, sub-second CPU).
    ``model_factory`` is injectable for tests: it must return a
    ``(model, device_str)`` pair where ``model(frame, device=..., points=...,
    labels=..., verbose=False)`` follows the Ultralytics predict API.
    """

    def __init__(self, model_factory=None):
        self._factory = model_factory or _default_factory
        self._loaded = None
        self.last_raw_count = 0   # FastSAM candidates BEFORE filtering

    @property
    def loaded(self) -> bool:
        """True once the FastSAM model has been constructed."""
        return self._loaded is not None

    def suggest(self, frame_bgr, x: int, y: int) -> list[list[int]]:
        """Candidate boxes ``[x1, y1, x2, y2]`` containing image point (x, y).

        Most-specific (smallest) first, capped at :data:`MAX_SUGGESTIONS`.
        ``last_raw_count`` records how many candidates FastSAM returned before
        area/point filtering, so the caller can distinguish "nothing found"
        from "everything filtered".  Raises whatever the model raises (the
        caller shows it as status).
        """
        self.last_raw_count = 0
        if self._loaded is None:
            self._loaded = self._factory()
        model, device = self._loaded
        res = model(frame_bgr, device=device, points=[[int(x), int(y)]],
                    labels=[1], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        self.last_raw_count = len(res.boxes)
        h, w = frame_bgr.shape[:2]
        area_img = float(h * w)
        cands = []
        for bb in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = (float(bb[0]), float(bb[1]),
                              float(bb[2]), float(bb[3]))
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if not (_MIN_AREA_FRAC * area_img < area < _MAX_AREA_FRAC * area_img):
                continue
            # The prompt should already restrict to point-related regions;
            # enforce it anyway so a stray mask never proposes elsewhere.
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            cands.append([int(x1), int(y1), int(x2), int(y2)])
        cands.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        kept: list[list[int]] = []
        for c in cands:
            if all(_iou(c, k) < _DEDUP_IOU for k in kept):
                kept.append(c)
            if len(kept) >= MAX_SUGGESTIONS:
                break
        return kept


def _default_factory():
    path = fastsam_path()
    if path is None:
        raise FileNotFoundError(
            "FastSAM-s.pt is not downloaded -- fetch it from the Models tab "
            "(SAM backend) to use Suggest on click.")
    from ultralytics import FastSAM

    from mindsight.utils.device import resolve_device
    model = FastSAM(str(path))
    dev = resolve_device("auto")
    return model, str(getattr(dev, "type", dev))
