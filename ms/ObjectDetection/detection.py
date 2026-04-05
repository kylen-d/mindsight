"""
ObjectDetection/detection.py — Formal Detection data type.

Replaces the implicit dict schema ({class_name, cls_id, conf, x1, y1, x2, y2,
_ghost}) with a typed dataclass.  Dict-style access (d['x1'], d.get('_ghost'))
is preserved so existing code continues to work unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np


@dataclasses.dataclass(slots=True)
class Detection:
    """A single object detection from YOLO (or equivalent)."""

    class_name: str
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    ghost: bool = False

    # Extra field used when faces are treated as objects.
    _face_idx: int | None = dataclasses.field(default=None, repr=False)

    # ── Dict-compatible access (backward compat) ─────────────────────────────

    _KEY_MAP: ClassVar[dict] = {
        '_ghost': 'ghost',
    }

    def __getitem__(self, key: str):
        attr = self._KEY_MAP.get(key, key)
        try:
            return getattr(self, attr)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        attr = self._KEY_MAP.get(key, key)
        return hasattr(self, attr)

    def __setitem__(self, key: str, value):
        attr = self._KEY_MAP.get(key, key)
        object.__setattr__(self, attr, value)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    # ── Geometry helpers ─────────────────────────────────────────────────────

    @property
    def center(self) -> np.ndarray:
        """Bounding-box centre as a float numpy array [cx, cy]."""
        return np.array([(self.x1 + self.x2) / 2,
                         (self.y1 + self.y2) / 2], float)

    # Allow iteration over keys (used by dict(det, _ghost=True) patterns).
    def keys(self):
        return [f.name for f in dataclasses.fields(self)
                if not f.name.startswith('_') or f.name == '_face_idx']

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]
