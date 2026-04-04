"""
MindSight_GUI.py
----------------
PyQt6 GUI wrapping all MindSight.py functionality, plus a YOLOE Visual
Prompt Builder for creating and testing .vp.json prompt files.

Tabs
----
1. Gaze Tracker        – configure and run the live gaze-intersection tracker.
                         Detector can be YOLO (text classes) or YOLOE Visual Prompt.
2. VP Builder          – draw bounding boxes on reference images to build a
                         .vp.json visual prompt file, then test inference with it.

Visual Prompt file format (.vp.json)
-------------------------------------
{
  "version": 1,
  "classes": [{"id": 0, "name": "knife"}, ...],   // sequential IDs from 0
  "references": [
    {
      "image": "/abs/path/ref.jpg",
      "annotations": [{"cls_id": 0, "bbox": [x1,y1,x2,y2]}, ...]
    }, ...
  ]
}

For YOLOE inference the FIRST reference image is used as `refer_image`
(sets class embeddings permanently); subsequent frames reuse cached embeddings.

Usage
-----
    python MindSight_GUI.py
"""

import json
import queue
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── Repo root ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent        # project root (one level above GUI/)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from constants import IMAGE_EXTS

VP_EXT     = ".vp.json"

_VP_PALETTE_BGR = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]


def _palette_hex(idx: int) -> str:
    b, g, r = _VP_PALETTE_BGR[idx % len(_VP_PALETTE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _bgr_to_pixmap(bgr, max_w: int = 0, max_h: int = 0) -> QPixmap:
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
    px = QPixmap.fromImage(qimg)
    if max_w > 0 and max_h > 0:
        px = px.scaled(max_w, max_h,
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
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
# Background worker: Gaze Tracker
# ══════════════════════════════════════════════════════════════════════════════

class GazeWorker(threading.Thread):
    """Loads models and runs gaze tracking; pushes BGR frames to frame_q."""

    def __init__(self, cfg: dict, frame_q: queue.Queue, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.cfg     = cfg
        self.frame_q = frame_q
        self.log_q   = log_q
        self._stop   = threading.Event()

    def stop(self):
        self._stop.set()

    def _log(self, msg):
        self.log_q.put(str(msg))

    def run(self):
        try:
            self._main()
        except Exception as exc:
            import traceback
            self._log(f"[ERROR] {exc}\n{traceback.format_exc()}")
        finally:
            self.frame_q.put(None)

    def _main(self):
        import time as _time

        import cv2

        from GazeTracking.gaze_factory import create_gaze_engine
        from MindSight import run
        from pipeline_config import DetectionConfig, GazeConfig, OutputConfig, TrackerConfig

        cfg = self.cfg

        # ── Object detector ───────────────────────────────────────────────────
        from ObjectDetection.model_factory import create_face_detector, create_yolo_detector
        if cfg.get("detection_mode") == "yoloe_vp":
            self._log(f"Loading YOLOE VP: {cfg['yoloe_model']}  +  {cfg['vp_file']}")
            yolo, class_ids, blacklist = create_yolo_detector(
                vp_file=cfg["vp_file"], vp_model=cfg["yoloe_model"])
        else:
            self._log(f"Loading YOLO: {cfg['yolo_model']}")
            yolo, class_ids, blacklist = create_yolo_detector(
                model_path=cfg["yolo_model"],
                classes=cfg.get("classes") or None,
                blacklist_names=cfg.get("blacklist", []))

        # ── Gaze backend ──────────────────────────────────────────────────────
        self._log("Loading RetinaFace…")
        face_det = create_face_detector()

        backend = cfg["backend"]
        self._log(f"Loading gaze backend: {backend}  ({cfg['mgaze_model']})")
        backend_kwargs = {}
        if backend == "gazelle":
            backend_kwargs = dict(
                model_name=cfg["gazelle_name"],
                ckpt_path=cfg["mgaze_model"],
                inout_threshold=float(cfg.get("gazelle_inout_threshold", 0.5)),
            )
        gaze_engine = create_gaze_engine(
            mgaze_model=cfg["mgaze_model"],
            mgaze_arch=cfg.get("mgaze_arch"),
            mgaze_dataset=cfg.get("mgaze_dataset", "gaze360"),
            backend=backend,
            **backend_kwargs,
        )

        self._log("Models loaded — starting…")

        # ── Redirect cv2 display into the GUI ─────────────────────────────────
        _orig_imshow      = cv2.imshow
        _orig_waitkey     = cv2.waitKey
        _orig_destroy_all = cv2.destroyAllWindows
        _orig_destroy_win = cv2.destroyWindow

        _frame_q = self.frame_q
        _stop_ev = self._stop

        def _gui_imshow(_, frame):
            try:
                _frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        def _gui_waitkey(delay):
            if delay == 0:
                while not _stop_ev.is_set():
                    _time.sleep(0.05)
                return ord('q')
            return ord('q') if _stop_ev.is_set() else 1

        cv2.imshow            = _gui_imshow
        cv2.waitKey           = _gui_waitkey
        cv2.destroyAllWindows = lambda: None
        cv2.destroyWindow     = lambda *_: None

        try:
            source = cfg["source"]
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass
            gaze_cfg = GazeConfig(
                ray_length=cfg["ray_length"],
                adaptive_ray=cfg["adaptive_ray"],
                snap_dist=cfg["snap_dist"],
                snap_bbox_scale=cfg.get("snap_bbox_scale", 0.5),
                snap_w_dist=cfg.get("snap_w_dist", 1.0),
                snap_w_size=cfg.get("snap_w_size", 0.3),
                snap_w_intersect=cfg.get("snap_w_intersect", 0.5),
                conf_ray=cfg["conf_ray"],
                gaze_tips=cfg["gaze_tips"],
                tip_radius=cfg["tip_radius"],
                gaze_debug=cfg["gaze_debug"],
            )
            det_cfg = DetectionConfig(
                conf=cfg["conf"],
                class_ids=class_ids,
                blacklist=blacklist,
                detect_scale=cfg["detect_scale"],
            )
            tracker_cfg = TrackerConfig(
                use_lock=not cfg["no_lock"],
                dwell_frames=cfg["dwell_frames"],
                lock_dist=cfg["lock_dist"],
                skip_frames=cfg["skip_frames"],
            )
            output_cfg = OutputConfig(
                save=cfg["save"],
                log_path=cfg.get("log"),
                summary_path=cfg.get("summary"),
            )
            run(
                source, yolo, face_det, gaze_engine,
                gaze_cfg, det_cfg, tracker_cfg, output_cfg,
            )
        finally:
            cv2.imshow            = _orig_imshow
            cv2.waitKey           = _orig_waitkey
            cv2.destroyAllWindows = _orig_destroy_all
            cv2.destroyWindow     = _orig_destroy_win


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: VP Inference (test visual prompts on a batch of images)
# ══════════════════════════════════════════════════════════════════════════════

class VPInferenceWorker(threading.Thread):
    """
    Run YOLOE inference on a list of images using a .vp.json visual prompt
    file.  Supports both VP mode (refer_image) and traditional text-class mode
    (set_classes) depending on whether a vp_file is supplied.

    result_q items: {"path": Path, "dets": [...], "frame": ndarray|None}
    Sentinel: None pushed when done.
    """

    def __init__(self, model_path: str, image_paths: list,
                 result_q: queue.Queue, log_q: queue.Queue,
                 conf: float = 0.30,
                 vp_file: str | None = None,
                 text_classes: list | None = None):
        super().__init__(daemon=True)
        self.model_path  = model_path
        self.image_paths = image_paths
        self.result_q    = result_q
        self.log_q       = log_q
        self.conf        = conf
        self.vp_file     = vp_file          # None → text-class mode
        self.text_classes = text_classes    # used when vp_file is None
        self._stop       = threading.Event()

    def stop(self):
        self._stop.set()

    def _log(self, msg):
        self.log_q.put(str(msg))

    def run(self):
        try:
            self._main()
        except Exception as exc:
            import traceback
            self._log(f"[ERROR] {exc}\n{traceback.format_exc()}")
        finally:
            self.result_q.put(None)

    def _main(self):
        import cv2
        from ultralytics import YOLOE

        self._log(f"Loading YOLOE: {self.model_path}")
        model = YOLOE(self.model_path)

        # ── Determine mode ────────────────────────────────────────────────────
        if self.vp_file:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            vp_data = load_vp_file(self.vp_file)
            refer_image, visual_prompts, class_names = vp_to_yoloe_args(vp_data)
            self._log(f"VP classes: {class_names}  (ref: {Path(refer_image).name})")
            classes_set = False

            def _predict(frame):
                nonlocal classes_set
                if not classes_set:
                    r = model.predict(frame,
                                      refer_image=refer_image,
                                      visual_prompts=visual_prompts,
                                      predictor=YOLOEVPSegPredictor,
                                      conf=self.conf, verbose=False)
                    classes_set = True
                    return r
                return model.predict(frame, conf=self.conf, verbose=False)

        else:
            # Text-class mode (traditional YOLOE)
            prompts = self.text_classes or []
            model.set_classes(prompts)
            class_names = prompts
            self._log(f"Text classes: {prompts}")

            def _predict(frame):
                return model.predict(frame, conf=self.conf, verbose=False)

        # ── Inference loop ────────────────────────────────────────────────────
        for i, img_path in enumerate(self.image_paths):
            if self._stop.is_set():
                break
            frame = cv2.imread(str(img_path))
            if frame is None:
                self._log(f"SKIP {img_path.name}")
                self.result_q.put({"path": img_path, "dets": [], "frame": None})
                continue

            try:
                results = _predict(frame)
            except Exception as e:
                self._log(f"[WARN] {img_path.name}: {e}")
                self.result_q.put({"path": img_path, "dets": [], "frame": frame.copy()})
                continue

            dets = []
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    c      = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if c < self.conf:
                        continue
                    cls_name = (class_names[cls_id]
                                if cls_id < len(class_names) else str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({"cls_name": cls_name, "cls_id": cls_id, "conf": c,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "selected": True})

            self._log(f"[{i+1}/{len(self.image_paths)}] {img_path.name} → {len(dets)} det(s)")
            self.result_q.put({"path": img_path, "dets": dets, "frame": frame.copy()})


# ══════════════════════════════════════════════════════════════════════════════
# Custom image canvas widget
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
# Qt layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _hrow(*widgets):
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
    btn = QPushButton(text)
    btn.setFixedWidth(64)
    return btn


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Gaze Tracker
# ══════════════════════════════════════════════════════════════════════════════

class GazeTab(QWidget):

    GAZE_ARCHS = [
        "resnet18", "resnet34", "resnet50", "mobilenetv2",
        "mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3", "mobileone_s4",
    ]
    GAZE_DATASETS = ["gaze360", "mpiigaze"]
    GAZELLE_NAMES = sorted([
        "gazelle_dinov2_vitb14", "gazelle_dinov2_vitl14",
        "gazelle_dinov2_vitb14_inout", "gazelle_dinov2_vitl14_inout",
    ])

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: GazeWorker | None = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q:   queue.Queue = queue.Queue()
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_w   = QWidget()
        settings_lay = QVBoxLayout(settings_w)
        settings_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        settings_lay.setSpacing(6)
        scroll.setWidget(settings_w)
        outer.addWidget(scroll)

        self._build_settings(settings_lay)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(right, stretch=1)

        self._preview = QLabel()
        self._preview.setStyleSheet("background:#1a1a2e;")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self._preview, stretch=3)

        log_group = QGroupBox("Log")
        log_lay   = QVBoxLayout(log_group)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(130)
        self._log_box.setFont(QFont("Courier", 10))
        log_lay.addWidget(self._log_box)
        right_lay.addWidget(log_group)

    def _build_settings(self, lay):
        # ── Source ────────────────────────────────────────────────────────────
        g = QGroupBox("Source")
        f = QFormLayout(g)
        self._src = QLineEdit("0")
        btn = _browse_btn()
        btn.clicked.connect(self._browse_source)
        f.addRow("Source:", _hrow(self._src, btn))
        lay.addWidget(g)

        # ── Object Detection (YOLO or YOLOE VP) ───────────────────────────────
        g = QGroupBox("Object Detection")
        vl = QVBoxLayout(g)

        mode_row = _hrow()
        self._rb_det_yolo = QRadioButton("YOLO (text classes)")
        self._rb_det_vp   = QRadioButton("YOLOE Visual Prompt")
        self._rb_det_yolo.setChecked(True)
        mode_row.layout().addWidget(self._rb_det_yolo)
        mode_row.layout().addWidget(self._rb_det_vp)
        mode_row.layout().addStretch(1)
        vl.addWidget(mode_row)

        # YOLO sub-panel
        self._yolo_det_panel = QWidget()
        fp = QFormLayout(self._yolo_det_panel)
        fp.setContentsMargins(0, 0, 0, 0)
        self._yolo_model = QLineEdit("yolov8n.pt")
        btn2 = _browse_btn()
        btn2.clicked.connect(lambda: self._browse_to(self._yolo_model, "*.pt"))
        fp.addRow("Model:", _hrow(self._yolo_model, btn2))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95); self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.35); self._conf_spin.setDecimals(2)
        fp.addRow("Conf:", self._conf_spin)
        self._classes = QLineEdit()
        self._classes.setPlaceholderText("all (comma-separated to filter)")
        fp.addRow("Classes:", self._classes)
        self._blacklist = QLineEdit()
        self._blacklist.setPlaceholderText("none (comma-separated to suppress)")
        fp.addRow("Blacklist:", self._blacklist)
        vl.addWidget(self._yolo_det_panel)

        # YOLOE VP sub-panel
        self._vp_det_panel = QWidget()
        fv = QFormLayout(self._vp_det_panel)
        fv.setContentsMargins(0, 0, 0, 0)
        self._vp_file = QLineEdit()
        self._vp_file.setPlaceholderText("Select .vp.json (build in VP Builder tab)")
        vp_btn = _browse_btn()
        vp_btn.clicked.connect(self._browse_vp_file)
        fv.addRow("VP file:", _hrow(self._vp_file, vp_btn))
        self._yoloe_model = QLineEdit("yoloe-26l-seg.pt")
        yoloe_btn = _browse_btn()
        yoloe_btn.clicked.connect(lambda: self._browse_to(self._yoloe_model, "*.pt"))
        fv.addRow("YOLOE model:", _hrow(self._yoloe_model, yoloe_btn))
        self._vp_conf_spin = QDoubleSpinBox()
        self._vp_conf_spin.setRange(0.05, 0.95); self._vp_conf_spin.setSingleStep(0.05)
        self._vp_conf_spin.setValue(0.35); self._vp_conf_spin.setDecimals(2)
        fv.addRow("Conf:", self._vp_conf_spin)
        self._vp_det_panel.setVisible(False)
        vl.addWidget(self._vp_det_panel)

        self._rb_det_yolo.toggled.connect(self._refresh_det_mode)
        self._rb_det_vp.toggled.connect(self._refresh_det_mode)
        lay.addWidget(g)

        # ── Gaze Backend ──────────────────────────────────────────────────────
        g = QGroupBox("Gaze Backend")
        vl = QVBoxLayout(g)
        rb_row = _hrow()
        self._rb_mgaze   = QRadioButton("MGaze")
        self._rb_gazelle = QRadioButton("Gazelle")
        self._rb_mgaze.setChecked(True)
        rb_row.layout().addWidget(self._rb_mgaze)
        rb_row.layout().addWidget(self._rb_gazelle)
        rb_row.layout().addStretch(1)
        vl.addWidget(rb_row)

        self._gaze_model = QLineEdit(
            str(_HERE / "GazeTracking" / "Backends" / "MGaze" / "gaze-estimation" / "weights" / "mobileone_s0_gaze.onnx"))
        gm_btn = _browse_btn()
        gm_btn.clicked.connect(lambda: self._browse_to(self._gaze_model, "*.onnx *.pt"))
        model_row = QWidget(); model_lay = QFormLayout(model_row)
        model_lay.setContentsMargins(0, 0, 0, 0)
        model_lay.addRow("Model:", _hrow(self._gaze_model, gm_btn))
        self._gaze_model.textChanged.connect(self._refresh_backend)
        vl.addWidget(model_row)

        self._arch_widget = QWidget()
        awl = QFormLayout(self._arch_widget)
        awl.setContentsMargins(0, 0, 0, 0)
        self._gaze_arch = QComboBox(); self._gaze_arch.addItems(self.GAZE_ARCHS)
        self._gaze_arch.setCurrentText("mobileone_s0")
        self._gaze_dataset = QComboBox(); self._gaze_dataset.addItems(self.GAZE_DATASETS)
        awl.addRow("Arch:", self._gaze_arch)
        awl.addRow("Dataset:", self._gaze_dataset)
        self._arch_widget.setVisible(False)
        vl.addWidget(self._arch_widget)

        self._gazelle_widget = QWidget()
        gwl = QFormLayout(self._gazelle_widget)
        gwl.setContentsMargins(0, 0, 0, 0)
        self._gazelle_name = QComboBox(); self._gazelle_name.addItems(self.GAZELLE_NAMES)
        self._gazelle_ckpt = QLineEdit()
        gc_btn = _browse_btn()
        gc_btn.clicked.connect(lambda: self._browse_to(self._gazelle_ckpt, "*.pt"))
        self._gazelle_inout = QDoubleSpinBox()
        self._gazelle_inout.setRange(0.0, 1.0); self._gazelle_inout.setSingleStep(0.05)
        self._gazelle_inout.setValue(0.5); self._gazelle_inout.setDecimals(2)
        gwl.addRow("Variant:", self._gazelle_name)
        gwl.addRow("Checkpoint:", _hrow(self._gazelle_ckpt, gc_btn))
        gwl.addRow("InOut thr:", self._gazelle_inout)
        self._gazelle_widget.setVisible(False)
        vl.addWidget(self._gazelle_widget)

        self._rb_mgaze.toggled.connect(self._refresh_backend)
        self._rb_gazelle.toggled.connect(self._refresh_backend)
        lay.addWidget(g)

        # ── Ray & Tracking ────────────────────────────────────────────────────
        g = QGroupBox("Gaze Ray & Tracking")
        f = QFormLayout(g)
        self._ray_length = QDoubleSpinBox()
        self._ray_length.setRange(0.2, 5.0); self._ray_length.setSingleStep(0.1)
        self._ray_length.setValue(1.0); self._ray_length.setDecimals(1)
        f.addRow("Ray length:", self._ray_length)
        self._skip_frames = QSpinBox()
        self._skip_frames.setRange(1, 10); self._skip_frames.setValue(1)
        f.addRow("Skip frames:", self._skip_frames)
        self._detect_scale = QDoubleSpinBox()
        self._detect_scale.setRange(0.25, 1.0); self._detect_scale.setSingleStep(0.05)
        self._detect_scale.setValue(1.0); self._detect_scale.setDecimals(2)
        f.addRow("Detect scale:", self._detect_scale)
        lay.addWidget(g)

        # ── Options ───────────────────────────────────────────────────────────
        g = QGroupBox("Options")
        fl = QFormLayout(g)
        self._cb_gaze_debug = QCheckBox("Show pitch/yaw debug overlay")
        self._cb_conf_ray   = QCheckBox("Scale ray length by confidence")
        self._cb_save       = QCheckBox("Save output to file")
        for cb in [self._cb_gaze_debug, self._cb_conf_ray, self._cb_save]:
            fl.addRow(cb)
        lay.addWidget(g)

        # ── Gaze tips ─────────────────────────────────────────────────────────
        self._gaze_tips_group = QGroupBox("Gaze tips (virtual objects)")
        self._gaze_tips_group.setCheckable(True); self._gaze_tips_group.setChecked(False)
        ft = QFormLayout(self._gaze_tips_group)
        self._tip_radius = QSpinBox(); self._tip_radius.setRange(20, 300); self._tip_radius.setValue(80)
        ft.addRow("Tip radius:", self._tip_radius)
        lay.addWidget(self._gaze_tips_group)

        # ── Adaptive ray ──────────────────────────────────────────────────────
        self._adaptive_group = QGroupBox("Adaptive ray")
        fa = QFormLayout(self._adaptive_group)
        self._adaptive_ray_combo = QComboBox()
        self._adaptive_ray_combo.addItems(["Off", "Extend", "Snap"])
        fa.addRow("Mode:", self._adaptive_ray_combo)
        self._snap_dist = QSpinBox(); self._snap_dist.setRange(20, 500); self._snap_dist.setValue(150)
        fa.addRow("Snap dist:", self._snap_dist)
        self._snap_bbox_scale = QDoubleSpinBox(); self._snap_bbox_scale.setRange(0.0, 2.0)
        self._snap_bbox_scale.setSingleStep(0.1); self._snap_bbox_scale.setValue(0.0)
        fa.addRow("Bbox scale:", self._snap_bbox_scale)
        self._snap_w_dist = QDoubleSpinBox(); self._snap_w_dist.setRange(0.0, 3.0)
        self._snap_w_dist.setSingleStep(0.1); self._snap_w_dist.setValue(1.0)
        fa.addRow("W dist:", self._snap_w_dist)
        self._snap_w_size = QDoubleSpinBox(); self._snap_w_size.setRange(0.0, 3.0)
        self._snap_w_size.setSingleStep(0.1); self._snap_w_size.setValue(0.0)
        fa.addRow("W size:", self._snap_w_size)
        self._snap_w_intersect = QDoubleSpinBox(); self._snap_w_intersect.setRange(0.0, 3.0)
        self._snap_w_intersect.setSingleStep(0.1); self._snap_w_intersect.setValue(0.5)
        fa.addRow("W intersect:", self._snap_w_intersect)
        lay.addWidget(self._adaptive_group)

        # ── Lock-on ───────────────────────────────────────────────────────────
        self._lock_group = QGroupBox("Lock-on")
        self._lock_group.setCheckable(True); self._lock_group.setChecked(True)
        fl2 = QFormLayout(self._lock_group)
        self._dwell_frames = QSpinBox(); self._dwell_frames.setRange(1, 120); self._dwell_frames.setValue(15)
        fl2.addRow("Dwell frames:", self._dwell_frames)
        self._lock_dist = QSpinBox(); self._lock_dist.setRange(20, 400); self._lock_dist.setValue(100)
        fl2.addRow("Lock dist:", self._lock_dist)
        lay.addWidget(self._lock_group)

        # ── CSV Output ────────────────────────────────────────────────────────
        g = QGroupBox("CSV Output (video only)")
        f = QFormLayout(g)
        self._log_path = QLineEdit()
        self._log_path.setPlaceholderText("optional — click Browse")
        lb = _browse_btn()
        lb.clicked.connect(lambda: self._browse_save(self._log_path, "CSV (*.csv)"))
        f.addRow("Event log:", _hrow(self._log_path, lb))
        self._summary_path = QLineEdit()
        self._summary_path.setPlaceholderText("optional — click Browse")
        sb = _browse_btn()
        sb.clicked.connect(lambda: self._browse_save(self._summary_path, "CSV (*.csv)"))
        f.addRow("Summary CSV:", _hrow(self._summary_path, sb))
        lay.addWidget(g)

        # ── Start / Stop ──────────────────────────────────────────────────────
        btn_row = _hrow()
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setStyleSheet(
            "QPushButton{background:#2a7a2a;color:white;font-weight:bold;padding:6px;}")
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:white;font-weight:bold;padding:6px;}")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.layout().addWidget(self._start_btn, 1)
        btn_row.layout().addWidget(self._stop_btn, 1)
        lay.addWidget(btn_row)
        lay.addStretch(1)

    # ── Visibility helpers ────────────────────────────────────────────────────

    def _refresh_det_mode(self):
        vp = self._rb_det_vp.isChecked()
        self._yolo_det_panel.setVisible(not vp)
        self._vp_det_panel.setVisible(vp)

    def _refresh_backend(self):
        is_mgaze = self._rb_mgaze.isChecked()
        mgaze_path = self._gaze_model.text().strip().lower()
        self._arch_widget.setVisible(is_mgaze and mgaze_path.endswith('.pt'))
        self._gazelle_widget.setVisible(self._rb_gazelle.isChecked())

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source", "",
            "Video/Image (*.mp4 *.mov *.avi *.jpg *.jpeg *.png *.bmp);;All (*)")
        if path:
            self._src.setText(path)

    def _browse_vp_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select VP file", "",
            f"Visual Prompt (*{VP_EXT});;JSON (*.json);;All (*)")
        if path:
            self._vp_file.setText(path)

    def _browse_to(self, line_edit: QLineEdit, filt: str = "*"):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", "", f"Files ({filt});;All (*)")
        if path:
            line_edit.setText(path)

    def _browse_save(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", "", filt)
        if path:
            line_edit.setText(path)

    # ── Config ────────────────────────────────────────────────────────────────

    def _build_cfg(self) -> dict:
        use_vp  = self._rb_det_vp.isChecked()
        backend = ("gazelle" if self._rb_gazelle.isChecked() else "mgaze")
        gaze_model = (self._gazelle_ckpt.text() if backend == "gazelle"
                      else self._gaze_model.text())
        cls_raw = self._classes.text().strip()
        bl_raw  = self._blacklist.text().strip()
        return {
            "source":         self._src.text().strip(),
            # detection
            "detection_mode": "yoloe_vp" if use_vp else "yolo",
            "yolo_model":     self._yolo_model.text().strip(),
            "yoloe_model":    self._yoloe_model.text().strip(),
            "vp_file":        self._vp_file.text().strip(),
            "conf":           self._vp_conf_spin.value() if use_vp else self._conf_spin.value(),
            "classes":        [c.strip() for c in cls_raw.split(",") if c.strip()] or None,
            "blacklist":      [c.strip() for c in bl_raw.split(",") if c.strip()],
            # gaze
            "backend":        backend,
            "mgaze_model":    gaze_model.strip(),
            "mgaze_arch":     self._gaze_arch.currentText(),
            "mgaze_dataset":  self._gaze_dataset.currentText(),
            "gazelle_name":   self._gazelle_name.currentText(),
            "gazelle_inout_threshold": self._gazelle_inout.value(),
            # ray
            "ray_length":     self._ray_length.value(),
            "skip_frames":    self._skip_frames.value(),
            "detect_scale":   self._detect_scale.value(),
            "tip_radius":     self._tip_radius.value(),
            "snap_dist":      self._snap_dist.value(),
            "snap_bbox_scale": self._snap_bbox_scale.value(),
            "snap_w_dist":    self._snap_w_dist.value(),
            "snap_w_size":    self._snap_w_size.value(),
            "snap_w_intersect": self._snap_w_intersect.value(),
            "dwell_frames":   self._dwell_frames.value(),
            "lock_dist":      self._lock_dist.value(),
            # flags
            "gaze_debug":     self._cb_gaze_debug.isChecked(),
            "conf_ray":       self._cb_conf_ray.isChecked(),
            "gaze_tips":      self._gaze_tips_group.isChecked(),
            "adaptive_ray":   self._adaptive_ray_combo.currentText().lower(),
            "no_lock":        not self._lock_group.isChecked(),
            "save":           self._cb_save.isChecked(),
            "log":            self._log_path.text().strip() or None,
            "summary":        self._summary_path.text().strip() or None,
        }

    # ── Start / Stop / Poll ───────────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.is_alive():
            return
        cfg = self._build_cfg()
        if not cfg["source"]:
            QMessageBox.critical(self, "Error", "Source is required."); return
        if not cfg["mgaze_model"]:
            QMessageBox.critical(self, "Error", "Gaze model path is required."); return
        if cfg["detection_mode"] == "yoloe_vp":
            if not cfg["vp_file"]:
                QMessageBox.critical(self, "Error", "Select a VP file or switch to YOLO mode.")
                return
            if not Path(cfg["vp_file"]).exists():
                QMessageBox.critical(self, "Error", f"VP file not found:\n{cfg['vp_file']}")
                return
        self._frame_q = queue.Queue(maxsize=4)
        self._log_q   = queue.Queue()
        self._worker  = GazeWorker(cfg, self._frame_q, self._log_q)
        self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._append_log("Starting…")
        self._poll_timer.start(30)

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _poll(self):
        try:
            while True:
                self._append_log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            frame = self._frame_q.get_nowait()
            if frame is None:
                self._poll_timer.stop()
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                self._append_log("Stopped.")
                return
            pw = self._preview.width() or 640
            ph = self._preview.height() or 480
            self._preview.setPixmap(_bgr_to_pixmap(frame, pw, ph))
        except queue.Empty:
            pass
        if self._worker and not self._worker.is_alive():
            self._poll_timer.stop()
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    def _append_log(self, msg: str):
        self._log_box.append(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())

    # ── Called externally to pre-fill a VP file (e.g. from VP Builder tab) ──

    def set_vp_file(self, path: str):
        self._vp_file.setText(path)
        self._rb_det_vp.setChecked(True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Visual Prompt Builder
# ══════════════════════════════════════════════════════════════════════════════

class VisualPromptBuilderTab(QWidget):
    """
    Build YOLOE Visual Prompt files (.vp.json) from manual bounding-box
    annotations on reference images, then test inference with those prompts.

    Workflow
    --------
    1. Load reference images (folder or individual files).
    2. Define classes in the Classes panel (right).  Each class gets the next
       sequential ID automatically.
    3. Select a class in the list, then drag on the image canvas to draw a box.
       Click an existing box to delete it.
    4. [Save VP File]  →  saves a .vp.json file.
       [Load VP File]  →  restores a previously saved file for editing.

    Testing
    -------
    5. In the "Test Inference" section (bottom) select a YOLOE model and an
       optional separate target folder, then click "▶ Test".
       Results are shown on the canvas alongside annotation boxes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # {path_str: {"frame": ndarray|None, "annotations": [{"cls_id":int, "cls_name":str, "bbox":[x1,y1,x2,y2]}, ...]}}
        self._images: dict[str, dict] = {}
        self._current_path: str | None = None
        self._classes: list[dict] = []   # [{"id": int, "name": str}, ...]
        self._test_dets: dict[str, list] = {}   # inference results per image
        self._vp_worker: VPInferenceWorker | None = None
        self._result_q: queue.Queue = queue.Queue()
        self._log_q:    queue.Queue = queue.Queue()
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_worker)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Top toolbar ───────────────────────────────────────────────────────
        tb = QWidget()
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(6)

        load_dir_btn = QPushButton("Load Folder…")
        load_dir_btn.clicked.connect(self._load_folder)
        load_files_btn = QPushButton("Add Images…")
        load_files_btn.clicked.connect(self._load_files)
        tbl.addWidget(load_dir_btn)
        tbl.addWidget(load_files_btn)

        tbl.addWidget(QFrame(frameShape=QFrame.Shape.VLine))

        load_vp_btn = QPushButton("Load VP File…")
        load_vp_btn.clicked.connect(self._load_vp_file)
        self._save_vp_btn = QPushButton("Save VP File…")
        self._save_vp_btn.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:5px 10px;}")
        self._save_vp_btn.clicked.connect(self._save_vp_file)
        tbl.addWidget(load_vp_btn)
        tbl.addWidget(self._save_vp_btn)

        tbl.addStretch(1)
        outer.addWidget(tb)

        # ── Main splitter (images | canvas | classes+annotations) ─────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # Left: reference image list
        left_grp = QGroupBox("Reference Images")
        left_lay = QVBoxLayout(left_grp)
        self._file_list = QListWidget()
        self._file_list.setFixedWidth(190)
        self._file_list.currentRowChanged.connect(self._on_file_select)
        left_lay.addWidget(self._file_list)
        rm_img_btn = QPushButton("Remove Selected")
        rm_img_btn.clicked.connect(self._remove_current_image)
        left_lay.addWidget(rm_img_btn)
        splitter.addWidget(left_grp)

        # Centre: canvas
        canvas_grp = QGroupBox(
            "Image  [select class → drag to draw box · click box to delete]")
        canvas_lay = QVBoxLayout(canvas_grp)
        self._canvas = ImageCanvas()
        self._canvas.box_toggled.connect(self._on_box_delete)
        self._canvas.crop_drawn.connect(self._on_box_drawn)
        canvas_lay.addWidget(self._canvas)
        self._canvas_status = QLabel("Load reference images to begin.")
        self._canvas_status.setStyleSheet("color:#aaa;font-size:11px;")
        canvas_lay.addWidget(self._canvas_status)
        splitter.addWidget(canvas_grp)

        # Right: classes panel + annotations panel
        right_vbox = QWidget()
        right_vlay = QVBoxLayout(right_vbox)
        right_vlay.setContentsMargins(0, 0, 0, 0)
        right_vlay.setSpacing(4)
        right_vbox.setFixedWidth(230)

        cls_grp = QGroupBox("Classes  (select before drawing)")
        cls_lay = QVBoxLayout(cls_grp)
        self._class_list = QListWidget()
        self._class_list.setFixedHeight(140)
        cls_lay.addWidget(self._class_list)
        cls_btn_row = _hrow()
        add_cls_btn = QPushButton("+ Add")
        add_cls_btn.clicked.connect(self._add_class)
        ren_cls_btn = QPushButton("Rename")
        ren_cls_btn.clicked.connect(self._rename_class)
        del_cls_btn = QPushButton("Remove")
        del_cls_btn.clicked.connect(self._remove_class)
        cls_btn_row.layout().addWidget(add_cls_btn)
        cls_btn_row.layout().addWidget(ren_cls_btn)
        cls_btn_row.layout().addWidget(del_cls_btn)
        cls_lay.addWidget(cls_btn_row)
        right_vlay.addWidget(cls_grp)

        ann_grp = QGroupBox("Annotations  (current image)")
        ann_lay = QVBoxLayout(ann_grp)
        self._ann_scroll = QScrollArea()
        self._ann_scroll.setWidgetResizable(True)
        self._ann_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ann_widget = QWidget()
        self._ann_lay    = QVBoxLayout(self._ann_widget)
        self._ann_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._ann_lay.setSpacing(2)
        self._ann_scroll.setWidget(self._ann_widget)
        ann_lay.addWidget(self._ann_scroll)
        clear_btn = QPushButton("Clear Image Annotations")
        clear_btn.clicked.connect(self._clear_image_annotations)
        ann_lay.addWidget(clear_btn)
        right_vlay.addWidget(ann_grp, stretch=1)
        splitter.addWidget(right_vbox)

        splitter.setSizes([190, 700, 230])

        # ── Test Inference section ─────────────────────────────────────────────
        test_grp = QGroupBox("Test Inference  (optional — verify your VP file)")
        test_grp.setCheckable(True)
        test_grp.setChecked(False)
        test_lay = QHBoxLayout(test_grp)
        test_lay.setSpacing(6)

        test_lay.addWidget(QLabel("YOLOE model:"))
        self._test_model = QComboBox()
        seg_models = sorted(str(p.name) for p in _HERE.glob("yoloe-*.pt"))
        self._test_model.addItems(seg_models or ["yoloe-26l-seg.pt"])
        self._test_model.setEditable(True)
        test_lay.addWidget(self._test_model, 2)
        tm_btn = QPushButton("…")
        tm_btn.setFixedWidth(28)
        tm_btn.clicked.connect(self._browse_test_model)
        test_lay.addWidget(tm_btn)

        test_lay.addWidget(QFrame(frameShape=QFrame.Shape.VLine))
        test_lay.addWidget(QLabel("Target folder (opt.):"))
        self._test_folder = QLineEdit()
        self._test_folder.setPlaceholderText("leave blank → test on reference images")
        test_lay.addWidget(self._test_folder, 2)
        tf_btn = QPushButton("Browse")
        tf_btn.clicked.connect(self._browse_test_folder)
        test_lay.addWidget(tf_btn)

        test_lay.addWidget(QLabel("Conf:"))
        self._test_conf = QDoubleSpinBox()
        self._test_conf.setRange(0.05, 0.95); self._test_conf.setSingleStep(0.05)
        self._test_conf.setValue(0.30); self._test_conf.setDecimals(2)
        self._test_conf.setFixedWidth(64)
        test_lay.addWidget(self._test_conf)

        test_lay.addWidget(QFrame(frameShape=QFrame.Shape.VLine))
        self._test_run_btn = QPushButton("▶  Test")
        self._test_run_btn.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:5px 10px;}")
        self._test_run_btn.clicked.connect(self._run_test)
        self._test_stop_btn = QPushButton("■  Stop")
        self._test_stop_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:white;font-weight:bold;padding:5px 10px;}")
        self._test_stop_btn.setEnabled(False)
        self._test_stop_btn.clicked.connect(self._stop_test)
        test_lay.addWidget(self._test_run_btn)
        test_lay.addWidget(self._test_stop_btn)
        outer.addWidget(test_grp)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = QLabel("Ready.")
        self._status.setStyleSheet("color:#888;font-size:11px;padding:2px;")
        outer.addWidget(self._status)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select reference image folder")
        if not folder:
            return
        imgs = sorted(p for p in Path(folder).iterdir()
                      if p.suffix.lower() in IMAGE_EXTS)
        added = 0
        for p in imgs:
            if str(p) not in self._images:
                self._images[str(p)] = {"frame": None, "annotations": []}
                self._file_list.addItem(p.name)
                added += 1
        self._set_status(f"Loaded {added} image(s) from {Path(folder).name}/")
        if self._file_list.count() > 0 and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)

    def _load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select reference images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;All (*)")
        added = 0
        for p in paths:
            if p not in self._images:
                self._images[p] = {"frame": None, "annotations": []}
                self._file_list.addItem(Path(p).name)
                added += 1
        if added:
            self._set_status(f"Added {added} image(s)")
        if self._file_list.count() > 0 and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)

    def _remove_current_image(self):
        row = self._file_list.currentRow()
        if row < 0:
            return
        path = self._path_at_row(row)
        if path:
            del self._images[path]
        self._file_list.takeItem(row)
        self._current_path = None
        self._canvas.set_image_data(None, [], []) if hasattr(self._canvas, '_frame') else None
        self._refresh_annotations_panel()

    def _path_at_row(self, row: int) -> str | None:
        if row < 0 or row >= self._file_list.count():
            return None
        name = self._file_list.item(row).text()
        for p in self._images:
            if Path(p).name == name:
                return p
        return None

    # ── File selection ────────────────────────────────────────────────────────

    def _on_file_select(self, idx: int):
        if idx < 0:
            return
        path = self._path_at_row(idx)
        if path is None:
            return
        self._current_path = path
        info = self._images[path]
        if info["frame"] is None:
            try:
                import cv2
                info["frame"] = cv2.imread(path)
            except Exception:
                pass
        self._refresh_canvas()
        self._refresh_annotations_panel()

    # ── Class management ──────────────────────────────────────────────────────

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(c["name"] == name for c in self._classes):
            QMessageBox.warning(self, "Duplicate", f"Class '{name}' already exists.")
            return
        new_id = len(self._classes)
        self._classes.append({"id": new_id, "name": name})
        self._refresh_class_list()
        self._class_list.setCurrentRow(len(self._classes) - 1)

    def _rename_class(self):
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self._classes):
            return
        old_name = self._classes[row]["name"]
        new_name, ok = QInputDialog.getText(
            self, "Rename Class", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        self._classes[row]["name"] = new_name
        # Update existing annotations
        for info in self._images.values():
            for ann in info["annotations"]:
                if ann["cls_name"] == old_name:
                    ann["cls_name"] = new_name
        self._refresh_class_list()
        self._class_list.setCurrentRow(row)
        self._refresh_canvas()
        self._refresh_annotations_panel()

    def _remove_class(self):
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self._classes):
            return
        cls = self._classes[row]
        reply = QMessageBox.question(
            self, "Remove Class",
            f"Remove class '{cls['name']}' (ID {cls['id']}) and all its annotations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._classes.pop(row)
        # Re-assign sequential IDs
        for i, c in enumerate(self._classes):
            c["id"] = i
        # Remove annotations and re-map cls_ids
        for info in self._images.values():
            info["annotations"] = [
                {**a, "cls_id": next(c["id"] for c in self._classes
                                     if c["name"] == a["cls_name"])}
                for a in info["annotations"]
                if a["cls_name"] != cls["name"]
                   and any(c["name"] == a["cls_name"] for c in self._classes)
            ]
        self._refresh_class_list()
        self._refresh_canvas()
        self._refresh_annotations_panel()

    def _refresh_class_list(self):
        self._class_list.clear()
        for c in self._classes:
            col = _palette_hex(c["id"])
            item = QListWidgetItem(f"  [{c['id']}]  {c['name']}")
            item.setForeground(QColor(col))
            self._class_list.addItem(item)

    # ── Canvas / annotation management ───────────────────────────────────────

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        if x2 <= x1 or y2 <= y1:
            return
        if self._current_path is None:
            QMessageBox.warning(self, "No image", "Load a reference image first.")
            return
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self._classes):
            QMessageBox.warning(self, "No class selected",
                                "Add and select a class before drawing boxes.")
            return
        cls = self._classes[row]
        self._images[self._current_path]["annotations"].append({
            "cls_id":   cls["id"],
            "cls_name": cls["name"],
            "bbox":     [x1, y1, x2, y2],
        })
        self._refresh_canvas()
        self._refresh_annotations_panel()

    def _on_box_delete(self, idx: int):
        """Click on a box in VP builder mode → delete it."""
        if self._current_path is None:
            return
        anns = self._images[self._current_path]["annotations"]
        if 0 <= idx < len(anns):
            anns.pop(idx)
            self._refresh_canvas()
            self._refresh_annotations_panel()

    def _clear_image_annotations(self):
        if self._current_path is None:
            return
        self._images[self._current_path]["annotations"].clear()
        self._refresh_canvas()
        self._refresh_annotations_panel()

    def _refresh_canvas(self):
        if self._current_path is None:
            return
        info = self._images.get(self._current_path)
        if info is None or info["frame"] is None:
            return
        # Convert VP annotations to canvas det format
        dets = [{"cls_id":   a["cls_id"],
                 "cls_name": a["cls_name"],
                 "x1": int(a["bbox"][0]), "y1": int(a["bbox"][1]),
                 "x2": int(a["bbox"][2]), "y2": int(a["bbox"][3]),
                 "selected": True, "conf": 1.0}
                for a in info["annotations"]]
        # Overlay test inference detections (different palette entry — show at full opacity)
        test = self._test_dets.get(self._current_path, [])
        self._canvas.set_image_data(info["frame"], dets + test, [])
        n_ann  = len(info["annotations"])
        n_test = len(test)
        extras = f"  |  {n_test} test det(s)" if n_test else ""
        self._canvas_status.setText(
            f"{Path(self._current_path).name}  —  {n_ann} annotation(s){extras}")

    def _refresh_annotations_panel(self):
        while self._ann_lay.count():
            item = self._ann_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self._current_path is None:
            return
        anns = self._images.get(self._current_path, {}).get("annotations", [])
        for i, a in enumerate(anns):
            col   = _palette_hex(a["cls_id"])
            x1, y1, x2, y2 = [int(v) for v in a["bbox"]]
            label = (f'<font color="{col}">■</font>  '
                     f'<b>[{a["cls_id"]}] {a["cls_name"]}</b>  '
                     f'<small>({x1},{y1})–({x2},{y2})</small>')
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(label)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            row_l.addWidget(lbl, 1)
            del_btn = QPushButton("✕")
            del_btn.setFixedWidth(22)
            del_btn.clicked.connect(lambda _, idx=i: self._on_box_delete(idx))
            row_l.addWidget(del_btn)
            self._ann_lay.addWidget(row_w)

    # ── Save / Load VP file ───────────────────────────────────────────────────

    def _save_vp_file(self):
        if not self._classes:
            QMessageBox.warning(self, "Empty", "Define at least one class."); return
        has_anns = any(info["annotations"] for info in self._images.values())
        if not has_anns:
            QMessageBox.warning(self, "Empty", "Draw at least one annotation."); return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save VP file", "",
            f"Visual Prompt (*{VP_EXT});;JSON (*.json)")
        if not path:
            return
        if not path.endswith(VP_EXT) and not path.endswith(".json"):
            path += VP_EXT

        refs = [
            {"image": str(Path(img_path).resolve()),
             "annotations": [{"cls_id": a["cls_id"], "bbox": a["bbox"]}
                              for a in info["annotations"]]}
            for img_path, info in self._images.items()
            if info["annotations"]
        ]
        save_vp_file(path, self._classes, refs)
        self._last_saved_vp = path
        self._set_status(f"Saved → {Path(path).name}")
        QMessageBox.information(
            self, "Saved",
            f"Visual Prompt file saved:\n{path}\n\n"
            f"{len(self._classes)} class(es), {len(refs)} reference image(s)")

    def _load_vp_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load VP file", "",
            f"Visual Prompt (*{VP_EXT});;JSON (*.json);;All (*)")
        if not path:
            return
        try:
            data = load_vp_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load VP file:\n{e}"); return

        self._classes = list(data.get("classes", []))
        self._refresh_class_list()

        self._images.clear()
        self._file_list.clear()
        self._test_dets.clear()

        for ref in data.get("references", []):
            img_path = ref["image"]
            name_map = {c["id"]: c["name"] for c in self._classes}
            anns = [{"cls_id":   a["cls_id"],
                     "cls_name": name_map.get(a["cls_id"], str(a["cls_id"])),
                     "bbox":     a["bbox"]}
                    for a in ref.get("annotations", [])]
            self._images[img_path] = {"frame": None, "annotations": anns}
            self._file_list.addItem(Path(img_path).name)

        self._set_status(
            f"Loaded {Path(path).name}  —  "
            f"{len(self._classes)} class(es), {len(data.get('references',[]))} reference(s)")
        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    # ── Test inference ────────────────────────────────────────────────────────

    def _browse_test_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLOE model", "", "Weights (*.pt)")
        if path:
            self._test_model.setCurrentText(path)

    def _browse_test_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select target folder")
        if path:
            self._test_folder.setText(path)

    def _run_test(self):
        if not self._classes:
            QMessageBox.warning(self, "Empty", "No classes defined."); return
        has_anns = any(info["annotations"] for info in self._images.values())
        if not has_anns:
            QMessageBox.warning(self, "Empty", "No annotations drawn."); return

        # Build a temporary VP file
        import tempfile
        refs = [
            {"image": str(Path(img_path).resolve()),
             "annotations": [{"cls_id": a["cls_id"], "bbox": a["bbox"]}
                              for a in info["annotations"]]}
            for img_path, info in self._images.items()
            if info["annotations"]
        ]
        tmp = tempfile.NamedTemporaryFile(suffix=VP_EXT, delete=False, mode="w")
        tmp.write(json.dumps({"version": 1, "classes": self._classes,
                              "references": refs}, indent=2))
        tmp.close()
        self._tmp_vp = tmp.name

        # Determine target images
        tf = self._test_folder.text().strip()
        if tf and Path(tf).is_dir():
            image_paths = sorted(p for p in Path(tf).iterdir()
                                 if p.suffix.lower() in IMAGE_EXTS)
        else:
            image_paths = [Path(p) for p in self._images if Path(p).exists()]

        if not image_paths:
            QMessageBox.warning(self, "No images", "No target images found."); return

        self._test_dets.clear()
        self._result_q = queue.Queue()
        self._log_q    = queue.Queue()
        self._vp_worker = VPInferenceWorker(
            model_path=self._test_model.currentText().strip(),
            image_paths=image_paths,
            result_q=self._result_q,
            log_q=self._log_q,
            conf=self._test_conf.value(),
            vp_file=self._tmp_vp,
        )
        self._vp_worker.start()
        self._test_run_btn.setEnabled(False)
        self._test_stop_btn.setEnabled(True)
        self._set_status("Running inference…")
        self._poll_timer.start(100)

    def _stop_test(self):
        if self._vp_worker:
            self._vp_worker.stop()
        self._poll_timer.stop()
        self._test_run_btn.setEnabled(True)
        self._test_stop_btn.setEnabled(False)

    def _poll_worker(self):
        try:
            while True:
                self._set_status(self._log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                item = self._result_q.get_nowait()
                if item is None:
                    self._poll_timer.stop()
                    self._test_run_btn.setEnabled(True)
                    self._test_stop_btn.setEnabled(False)
                    self._set_status("Inference done.")
                    self._refresh_canvas()
                    # Clean up temp file
                    if hasattr(self, "_tmp_vp"):
                        try:
                            Path(self._tmp_vp).unlink()
                        except Exception:
                            pass
                    return
                path_str = str(item["path"])
                if item["dets"]:
                    self._test_dets[path_str] = item["dets"]
                if path_str == self._current_path:
                    self._refresh_canvas()
        except queue.Empty:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status.setText(msg)

    def current_vp_path(self) -> str | None:
        """Return the last saved VP file path (for passing to GazeTab)."""
        return getattr(self, "_last_saved_vp", None)


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MindSight — Gaze Tracker + Visual Prompt Builder")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self._gaze_tab = GazeTab()
        self._vp_tab   = VisualPromptBuilderTab()
        tabs.addTab(self._gaze_tab, "  Gaze Tracker  ")
        tabs.addTab(self._vp_tab,   "  VP Builder  ")

        # Convenience: "Use in Gaze Tracker" button surfaced via status-bar button
        self._use_vp_btn = QPushButton("Use saved VP in Gaze Tracker")
        self._use_vp_btn.setStyleSheet(
            "QPushButton{background:#5a2a7a;color:white;font-weight:bold;padding:4px 10px;}")
        self._use_vp_btn.clicked.connect(self._push_vp_to_gaze)
        self.statusBar().addPermanentWidget(self._use_vp_btn)

    def _push_vp_to_gaze(self):
        path = getattr(self._vp_tab, "_last_saved_vp", None)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No VP file",
                                "Save a VP file in the VP Builder tab first.")
            return
        self._gaze_tab.set_vp_file(path)
        self.centralWidget().setCurrentIndex(0)  # switch to Gaze tab

    def closeEvent(self, event):
        if self._gaze_tab._worker and self._gaze_tab._worker.is_alive():
            self._gaze_tab._worker.stop()
        if self._vp_tab._vp_worker and self._vp_tab._vp_worker.is_alive():
            self._vp_tab._vp_worker.stop()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
