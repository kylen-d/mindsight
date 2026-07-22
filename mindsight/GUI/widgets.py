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

# ── Project root (mindsight/GUI/ -> mindsight/ -> project root) ──────────────
_HERE = Path(__file__).resolve().parent.parent.parent

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
    and drag-to-draw interaction.

    Suggest mode (v1.1 W3Z, validation wizard): while enabled, clicks stop
    toggling boxes / starting drags.  A click inside a suggestion box emits
    ``suggestion_accepted(index)``; any other click on the image emits
    ``point_clicked(ix, iy)`` in image coordinates.  Suggestions render as
    dashed overlays on top of the pixmap.

    Hybrid mode (v1.3.1 item 2, VP Builder): one always-on grammar decided on
    mouse RELEASE (<10 px movement = click): click a suggestion = accept,
    click a detection box = toggle, click empty image = ``point_clicked``,
    drag = ``crop_drawn``.  Esc or right-click dismisses suggestions
    (``suggestions_cleared``); hovering highlights the proposal under the
    cursor; digits and Ctrl/Cmd+arrows are re-emitted for the host widget's
    class switching (``digit_pressed`` / ``cycle_pressed``).
    """

    box_toggled = pyqtSignal(int)
    crop_drawn  = pyqtSignal(int, int, int, int)
    point_clicked       = pyqtSignal(int, int)
    suggestion_accepted = pyqtSignal(int)
    suggestions_cleared = pyqtSignal()
    digit_pressed       = pyqtSignal(int)   # 0-9
    cycle_pressed       = pyqtSignal(int)   # -1 / +1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._frame        = None
        self._dets         = []
        self._manual_crops = []
        self._pixmap       = None
        self._scale        = 1.0
        self._off_x = self._off_y = 0
        self._drag_start   = None
        self._drag_current = None
        self._suggest_mode = False
        self._hybrid       = False
        self._busy         = False
        self._hover_idx: int | None = None
        self._suggestions: list = []   # [[x1, y1, x2, y2], ...] image coords

    def set_suggest_mode(self, on: bool):
        self._suggest_mode = bool(on)
        if not on:
            self._suggestions = []
            self._hover_idx = None
        self.update()

    def set_hybrid_mode(self, on: bool):
        """Enable the release-based click/drag/suggest grammar (VP Builder)."""
        self._hybrid = bool(on)
        self.update()

    def set_busy(self, on: bool):
        """Show a wait cursor while the host runs background inference."""
        self._busy = bool(on)
        if on:
            self.setCursor(Qt.CursorShape.WaitCursor)
        else:
            self.unsetCursor()

    def set_suggestions(self, boxes):
        self._suggestions = [list(b) for b in (boxes or [])]
        self._hover_idx = None
        self.update()

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
        if self._suggestions and self._pixmap:
            for i, (ix1, iy1, ix2, iy2) in enumerate(self._suggestions):
                hovered = (i == self._hover_idx)
                pen = QPen(QColor("#ffb454"), 3 if hovered else 2,
                           Qt.PenStyle.SolidLine if hovered
                           else Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                x1 = int(ix1 * self._scale + self._off_x)
                y1 = int(iy1 * self._scale + self._off_y)
                x2 = int(ix2 * self._scale + self._off_x)
                y2 = int(iy2 * self._scale + self._off_y)
                painter.drawRect(QRect(QPoint(x1, y1), QPoint(x2, y2)))
                badge = QRect(x1, y1, 18, 16)
                painter.fillRect(badge, QColor("#ffb454"))
                painter.setPen(QColor("#1a1a2e"))
                painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(i + 1))

    def _widget_to_img(self, wx, wy):
        """Map widget coordinates to image coordinates."""
        if self._scale == 0:
            return 0, 0
        return (wx - self._off_x) / self._scale, (wy - self._off_y) / self._scale

    def _suggestion_order(self):
        """Indices smallest-area-first so nested clicks pick the most
        specific proposal (matches the suggestion sort order)."""
        return sorted(range(len(self._suggestions)),
                      key=lambda i: ((self._suggestions[i][2]
                                      - self._suggestions[i][0])
                                     * (self._suggestions[i][3]
                                        - self._suggestions[i][1])))

    def _suggestion_at(self, ix, iy) -> int | None:
        for i in self._suggestion_order():
            x1, y1, x2, y2 = self._suggestions[i]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return i
        return None

    def _dismiss_suggestions(self):
        if not self._suggestions:
            return
        self._suggestions = []
        self._hover_idx = None
        self.update()
        self.suggestions_cleared.emit()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape and self._suggestions:
            self._dismiss_suggestions()
            return
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            self.digit_pressed.emit(key - Qt.Key.Key_0)
            return
        if (key in (Qt.Key.Key_Left, Qt.Key.Key_Right)
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.cycle_pressed.emit(-1 if key == Qt.Key.Key_Left else 1)
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._dismiss_suggestions()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        ix, iy = self._widget_to_img(event.position().x(), event.position().y())
        if self._suggest_mode:
            i = self._suggestion_at(ix, iy)
            if i is not None:
                self.suggestion_accepted.emit(i)
                return
            if self._frame is not None:
                hh, ww = self._frame.shape[:2]
                if 0 <= ix < ww and 0 <= iy < hh:
                    self.point_clicked.emit(int(ix), int(iy))
            return
        if self._hybrid:
            # Click vs drag is decided on release (<10 px = click).
            self._drag_start   = (int(event.position().x()),
                                  int(event.position().y()))
            self._drag_current = self._drag_start
            return
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
            return
        if self._suggestions and not self._busy:
            ix, iy = self._widget_to_img(event.position().x(),
                                         event.position().y())
            hover = self._suggestion_at(ix, iy)
            if hover != self._hover_idx:
                self._hover_idx = hover
                self.setCursor(Qt.CursorShape.PointingHandCursor
                               if hover is not None
                               else Qt.CursorShape.ArrowCursor)
                self.update()

    def _hybrid_click(self, wx: int, wy: int):
        """Resolve a short press in hybrid mode: proposal > box > empty."""
        ix, iy = self._widget_to_img(wx, wy)
        i = self._suggestion_at(ix, iy)
        if i is not None:
            self.suggestion_accepted.emit(i)
            return
        for j, det in enumerate(self._dets):
            if det["x1"] <= ix <= det["x2"] and det["y1"] <= iy <= det["y2"]:
                self.box_toggled.emit(j)
                return
        if self._frame is not None:
            hh, ww = self._frame.shape[:2]
            if 0 <= ix < ww and 0 <= iy < hh:
                self.point_clicked.emit(int(ix), int(iy))

    def mouseReleaseEvent(self, event):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1c = int(event.position().x())
        y1c = int(event.position().y())
        self._drag_start = self._drag_current = None
        self.update()
        if abs(x1c - x0) < 10 or abs(y1c - y0) < 10:
            if self._hybrid:
                self._hybrid_click(x0, y0)
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


def middle_frame_pixmap(video_path, max_w: int = 640, max_h: int = 360):
    """Decode the MIDDLE frame of *video_path* as a scaled QPixmap, or None.

    The middle frame is the wizard's (and, later, the crop tool's) "what does
    this video look like" preview -- intros/fade-ins make the first frame a
    poor representative. Falls back to the first frame when the container
    reports no frame count; returns None when nothing decodes.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return _bgr_to_pixmap(frame, max_w, max_h)
    finally:
        cap.release()
