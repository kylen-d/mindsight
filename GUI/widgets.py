"""
widgets.py
----------
Shared widgets, layout helpers, and Visual Prompt file utilities for the
MindSight GUI.

Exports
-------
Constants : VP_EXT, _VP_PALETTE_BGR, _HERE
Functions : _palette_hex, _bgr_to_pixmap, save_vp_file, load_vp_file,
            vp_to_yoloe_args, _hrow, _browse_btn
Widgets   : ImageCanvas, CollapsibleGroupBox
"""

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Repo root (project root, one level above GUI/) ──────────────────────────
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from constants import IMAGE_EXTS  # noqa: F401  (re-export for convenience)

VP_EXT = ".vp.json"

_VP_PALETTE_BGR = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]


# ══════════════════════════════════════════════════════════════════════════════
# Colour / image helpers
# ══════════════════════════════════════════════════════════════════════════════

def _palette_hex(idx: int) -> str:
    """Return the hex colour string for palette index *idx*."""
    b, g, r = _VP_PALETTE_BGR[idx % len(_VP_PALETTE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _bgr_to_pixmap(bgr, max_w: int = 0, max_h: int = 0) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap, optionally scaled to fit."""
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
    px = QPixmap.fromImage(qimg)
    if max_w > 0 and max_h > 0:
        px = px.scaled(max_w, max_h,
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.FastTransformation)
    return px


# ══════════════════════════════════════════════════════════════════════════════
# Visual Prompt file utilities
# ══════════════════════════════════════════════════════════════════════════════

def save_vp_file(path: str, classes: list, references: list) -> None:
    """
    Save a Visual Prompt file.

    classes    : [{"id": int, "name": str}, ...]   IDs must be sequential from 0.
    references : [{"image": str,
                   "annotations": [{"cls_id": int, "bbox": [x1,y1,x2,y2]}, ...]
                  }, ...]
    """
    Path(path).write_text(json.dumps({"version": 1,
                                      "classes": classes,
                                      "references": references}, indent=2))


def load_vp_file(path: str) -> dict:
    """Load and return the contents of a Visual Prompt file."""
    return json.loads(Path(path).read_text())


def vp_to_yoloe_args(vp_data: dict):
    """
    Convert VP file data to YOLOE predict kwargs (refer_image mode).
    Uses the FIRST reference image's annotations.

    Returns (refer_image_path, visual_prompts_dict, class_names_list).
    """
    import numpy as np
    refs = vp_data.get("references", [])
    if not refs:
        raise ValueError("VP file contains no reference images")
    ref  = refs[0]
    anns = ref.get("annotations", [])
    if not anns:
        raise ValueError("First reference image has no annotations")
    return (
        str(ref["image"]),
        {"bboxes": np.array([a["bbox"]   for a in anns], dtype=float),
         "cls":    np.array([a["cls_id"] for a in anns], dtype=int)},
        [c["name"] for c in vp_data.get("classes", [])],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Qt layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hrow(*widgets):
    """Arrange *widgets* in a horizontal row.  Integer arguments become stretch."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    for wgt in widgets:
        if isinstance(wgt, int):
            lay.addStretch(wgt)
        else:
            lay.addWidget(wgt)
    return w


def _browse_btn(text="Browse"):
    """Return a small browse button with a minimum width."""
    btn = QPushButton(text)
    btn.setMinimumWidth(50)
    btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    return btn


# ══════════════════════════════════════════════════════════════════════════════
# ImageCanvas
# ══════════════════════════════════════════════════════════════════════════════

class ImageCanvas(QWidget):
    """Displays a BGR frame with detection overlays; supports click-to-toggle
    and drag-to-draw interaction."""

    box_toggled = pyqtSignal(int)
    crop_drawn  = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._frame        = None
        self._dets         = []
        self._manual_crops = []
        self._pixmap       = None
        self._scale        = 1.0
        self._off_x = self._off_y = 0
        self._drag_start   = None
        self._drag_current = None

    def set_image_data(self, frame, dets, manual_crops):
        """Initialize the canvas with a BGR *frame*, detections, and manual crops."""
        self._frame        = frame
        self._dets         = dets
        self._manual_crops = manual_crops
        self._rebuild_pixmap()
        self.update()

    def _rebuild_pixmap(self):
        if self._frame is None:
            self._pixmap = None
            return
        import cv2
        frame = self._frame.copy()
        for i, det in enumerate(self._dets):
            b, g, r = _VP_PALETTE_BGR[det["cls_id"] % len(_VP_PALETTE_BGR)]
            colour  = (b, g, r) if det.get("selected", True) else (90, 90, 90)
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            thick = 2 if det.get("selected", True) else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
            if det.get("selected", True):
                lbl = f"[{i}] {det['cls_name']} {det.get('conf', 1.0):.2f}"
                (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(frame, lbl, (x1 + 2, y1 - bl - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        for mc in self._manual_crops:
            x1, y1, x2, y2 = mc["x1"], mc["y1"], mc["x2"], mc["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, mc.get("label", ""), (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        self._pixmap = QPixmap.fromImage(
            QImage(rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#1a1a2e"))
        if self._pixmap:
            w, h   = self.width(), self.height()
            scaled = self._pixmap.scaled(w, h,
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            self._scale = scaled.width() / self._pixmap.width()
            self._off_x = (w - scaled.width())  // 2
            self._off_y = (h - scaled.height()) // 2
            painter.drawPixmap(self._off_x, self._off_y, scaled)
        if self._drag_start and self._drag_current:
            pen = QPen(QColor("#00ffff"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            x0, y0 = self._drag_start
            x1, y1 = self._drag_current
            painter.drawRect(QRect(QPoint(min(x0, x1), min(y0, y1)),
                                   QPoint(max(x0, x1), max(y0, y1))))

    def _widget_to_img(self, wx, wy):
        """Map widget coordinates to image coordinates."""
        if self._scale == 0:
            return 0, 0
        return (wx - self._off_x) / self._scale, (wy - self._off_y) / self._scale

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        ix, iy = self._widget_to_img(event.position().x(), event.position().y())
        for i, det in enumerate(self._dets):
            if det["x1"] <= ix <= det["x2"] and det["y1"] <= iy <= det["y2"]:
                self.box_toggled.emit(i)
                return
        self._drag_start   = (int(event.position().x()), int(event.position().y()))
        self._drag_current = self._drag_start

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_current = (int(event.position().x()), int(event.position().y()))
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1c = int(event.position().x())
        y1c = int(event.position().y())
        self._drag_start = self._drag_current = None
        self.update()
        if abs(x1c - x0) < 10 or abs(y1c - y0) < 10:
            return
        ix0, iy0 = self._widget_to_img(min(x0, x1c), min(y0, y1c))
        ix1, iy1 = self._widget_to_img(max(x0, x1c), max(y0, y1c))
        if self._frame is not None:
            hh, ww = self._frame.shape[:2]
            ix0 = max(0, int(ix0)); iy0 = max(0, int(iy0))
            ix1 = min(ww, int(ix1)); iy1 = min(hh, int(iy1))
        self.crop_drawn.emit(int(ix0), int(iy0), int(ix1), int(iy1))


# ══════════════════════════════════════════════════════════════════════════════
# CollapsibleGroupBox
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleGroupBox(QGroupBox):
    """A QGroupBox that starts collapsed and expands/collapses when its title
    checkbox is toggled.

    Usage::

        grp = CollapsibleGroupBox("Advanced Options")
        content = QWidget()
        # ... populate content ...
        grp.set_content(content)

    The group box is checkable; checking it expands the content area,
    unchecking it collapses it.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)

        self._content: QWidget | None = None

        # Internal layout that holds the content widget
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(0)

        self.toggled.connect(self._on_toggled)

    def set_content(self, widget: QWidget) -> None:
        """Set the collapsible content widget.  Replaces any previous content."""
        if self._content is not None:
            self._layout.removeWidget(self._content)
            self._content.setParent(None)
        self._content = widget
        self._layout.addWidget(self._content)
        # Initialize to collapsed state
        self._apply_collapsed_state(not self.isChecked())

    def _on_toggled(self, checked: bool) -> None:
        self._apply_collapsed_state(not checked)

    def _apply_collapsed_state(self, collapsed: bool) -> None:
        if self._content is None:
            return
        if collapsed:
            self._content.setMaximumHeight(0)
            self._content.setVisible(False)
        else:
            self._content.setMaximumHeight(16777215)
            self._content.setVisible(True)
