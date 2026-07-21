"""
Visual Prompt Builder tab for MindSight.

This module provides the :class:`VisualPromptBuilderTab` widget, which lets
users:

1. Load reference images from a folder or individual files.
2. Define object classes with automatically sequenced identifiers.
3. Draw bounding-box annotations on the image canvas.
4. Save and load ``.vp.json`` visual prompt files.
5. Run test YOLOE inference to verify the prompt before committing to a
   full gaze-tracking session.

Colour palette entries follow the same order used throughout the rest of
the MindSight GUI so that class colours remain consistent.

"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mindsight.constants import IMAGE_EXTS

from .widgets import (
    VP_EXT,
    ImageCanvas,
    _hrow,
    _palette_hex,
    load_vp_file,
    save_vp_file,
)
from .workers import VPInferenceWorker


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
    4. [Save VP File]  ->  saves a .vp.json file.
       [Load VP File]  ->  restores a previously saved file for editing.

    Testing
    -------
    5. In the "Test Inference" section (bottom) select a YOLOE model and an
       optional separate target folder, then click "Test".
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
        # Suggest mode (W3Z): lazy FastSAM suggester + its own poll loop.
        self._suggester = None
        self._suggest_q: queue.Queue = queue.Queue()
        self._suggest_busy = False
        self._pending_suggestions: list = []
        self._suggest_timer = QTimer()
        self._suggest_timer.timeout.connect(self._poll_suggest)
        self._build_ui()

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # -- Top toolbar ---------------------------------------------------
        tb = QWidget()
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(6)

        load_dir_btn = QPushButton("Load Folder\u2026")
        load_dir_btn.clicked.connect(self._load_folder)
        load_files_btn = QPushButton("Add Images\u2026")
        load_files_btn.clicked.connect(self._load_files)
        extract_btn = QPushButton("Extract Frames\u2026")
        extract_btn.setToolTip(
            "Pull still frames out of videos (files, a folder, or a "
            "project) to annotate here")
        extract_btn.clicked.connect(self._extract_frames)
        tbl.addWidget(load_dir_btn)
        tbl.addWidget(load_files_btn)
        tbl.addWidget(extract_btn)

        tbl.addWidget(QFrame(frameShape=QFrame.Shape.VLine))

        load_vp_btn = QPushButton("Load VP File\u2026")
        load_vp_btn.setToolTip(
            "Load a .vp.json, or a portable .vp.zip archive (extracted "
            "beside the archive)")
        load_vp_btn.clicked.connect(self._load_vp_file)
        self._save_vp_btn = QPushButton("Save VP File\u2026")
        self._save_vp_btn.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:5px 10px;}")
        self._save_vp_btn.clicked.connect(self._save_vp_file)
        export_btn = QPushButton("Export Portable\u2026")
        export_btn.setToolTip(
            "Bundle this prompt AND its images into one .vp.zip that works "
            "on any machine")
        export_btn.clicked.connect(self._export_portable)
        tbl.addWidget(load_vp_btn)
        tbl.addWidget(self._save_vp_btn)
        tbl.addWidget(export_btn)

        tbl.addStretch(1)
        outer.addWidget(tb)

        # -- Main splitter (images | canvas | classes+annotations) ---------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # Left: reference image list
        left_grp = QGroupBox("Reference Images")
        left_lay = QVBoxLayout(left_grp)
        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(140)
        self._file_list.currentRowChanged.connect(self._on_file_select)
        left_lay.addWidget(self._file_list)
        rm_img_btn = QPushButton("Remove Selected")
        rm_img_btn.clicked.connect(self._remove_current_image)
        left_lay.addWidget(rm_img_btn)
        splitter.addWidget(left_grp)

        # Centre: canvas
        canvas_grp = QGroupBox(
            "Image  [select class \u2192 drag to draw box \u00b7 click box to delete]")
        canvas_lay = QVBoxLayout(canvas_grp)
        self._canvas = ImageCanvas()
        self._canvas.box_toggled.connect(self._on_box_delete)
        self._canvas.crop_drawn.connect(self._on_box_drawn)
        self._canvas.point_clicked.connect(self._on_suggest_point)
        self._canvas.suggestion_accepted.connect(self._on_suggestion_accepted)
        canvas_lay.addWidget(self._canvas)
        suggest_row = QHBoxLayout()
        self._suggest_btn = QPushButton("Suggest mode")
        self._suggest_btn.setCheckable(True)
        self._suggest_btn.setToolTip(
            "Click an object in the image and accept a proposed box instead "
            "of drawing it by hand (FastSAM segmentation). Needs the "
            "FastSAM-s weight from the Models tab.")
        self._suggest_btn.toggled.connect(self._on_suggest_toggled)
        suggest_row.addWidget(self._suggest_btn)
        suggest_row.addStretch(1)
        canvas_lay.addLayout(suggest_row)
        self._canvas_status = QLabel("Load reference images to begin.")
        self._canvas_status.setStyleSheet("color:#aaa;font-size:11px;")
        canvas_lay.addWidget(self._canvas_status)
        splitter.addWidget(canvas_grp)

        # Right: classes panel + annotations panel
        right_vbox = QWidget()
        right_vlay = QVBoxLayout(right_vbox)
        right_vlay.setContentsMargins(0, 0, 0, 0)
        right_vlay.setSpacing(4)
        right_vbox.setMinimumWidth(180)

        cls_grp = QGroupBox("Classes  (select before drawing)")
        cls_lay = QVBoxLayout(cls_grp)
        self._class_list = QListWidget()
        self._class_list.setMinimumHeight(100)
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

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        # -- Test Inference section ----------------------------------------
        test_grp = QGroupBox("Test Inference  (optional \u2014 verify your VP file)")
        test_grp.setCheckable(True)
        test_grp.setChecked(False)
        test_lay = QHBoxLayout(test_grp)
        test_lay.setSpacing(6)

        test_lay.addWidget(QLabel("YOLOE model:"))
        self._test_model = QComboBox()
        # Read the shared Weights folder (WEIGHTS_ROOT honors MINDSIGHT_HOME), not
        # the package dir -- in a release install the YOLOE weights live under the
        # app's data home, not beside the source tree.
        from mindsight import weights as _weights
        _yolo_dir = _weights.WEIGHTS_ROOT / "YOLO"
        seg_models = sorted(str(p.name) for p in _yolo_dir.glob("yoloe-*.pt")) if _yolo_dir.is_dir() else []
        self._test_model.addItems(seg_models or ["yoloe-26l-seg.pt"])
        self._test_model.setEditable(True)
        test_lay.addWidget(self._test_model, 2)
        tm_btn = QPushButton("\u2026")
        tm_btn.setFixedWidth(28)
        tm_btn.clicked.connect(self._browse_test_model)
        test_lay.addWidget(tm_btn)

        test_lay.addWidget(QFrame(frameShape=QFrame.Shape.VLine))
        test_lay.addWidget(QLabel("Target folder (opt.):"))
        self._test_folder = QLineEdit()
        self._test_folder.setPlaceholderText("leave blank \u2192 test on reference images")
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
        self._test_run_btn = QPushButton("\u25b6  Test")
        self._test_run_btn.setStyleSheet(
            "QPushButton{background:#2a5a8a;color:white;font-weight:bold;padding:5px 10px;}")
        self._test_run_btn.clicked.connect(self._run_test)
        self._test_stop_btn = QPushButton("\u25a0  Stop")
        self._test_stop_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:white;font-weight:bold;padding:5px 10px;}")
        self._test_stop_btn.setEnabled(False)
        self._test_stop_btn.clicked.connect(self._stop_test)
        test_lay.addWidget(self._test_run_btn)
        test_lay.addWidget(self._test_stop_btn)
        outer.addWidget(test_grp)

        # -- Status bar ----------------------------------------------------
        self._status = QLabel("Ready.")
        self._status.setStyleSheet("color:#888;font-size:11px;padding:2px;")
        outer.addWidget(self._status)

    # -- Image loading -----------------------------------------------------

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
        added = self.add_images(paths)
        if added:
            self._set_status(f"Added {added} image(s)")

    def add_images(self, paths) -> int:
        """Programmatic reference-image add (frame extraction, tests)."""
        added = 0
        for p in paths or []:
            p = str(p)
            if p not in self._images:
                self._images[p] = {"frame": None, "annotations": []}
                self._file_list.addItem(Path(p).name)
                added += 1
        if self._file_list.count() > 0 and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)
        return added

    def _extract_frames(self):
        """MP2: extract stills from footage, then offer them as references."""
        from .frame_extract_dialog import FrameExtractDialog
        dlg = FrameExtractDialog(self)
        if not dlg.exec() or not dlg.extracted:
            return
        reply = QMessageBox.question(
            self, "Extracted",
            f"Extracted {len(dlg.extracted)} frame(s). Add them to the "
            "reference image list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            added = self.add_images(dlg.extracted)
            self._set_status(f"Added {added} extracted frame(s)")

    def _export_portable(self):
        """MP4: bundle the prompt + its images into one portable .vp.zip."""
        from .vp_archive import VP_ARCHIVE_EXT, export_vp_archive
        if not self._classes:
            QMessageBox.warning(self, "Empty", "Define at least one class.")
            return
        refs = [
            {"image": str(Path(img_path).resolve()),
             "annotations": [{"cls_id": a["cls_id"], "bbox": a["bbox"]}
                             for a in info["annotations"]]}
            for img_path, info in self._images.items()
            if info["annotations"]
        ]
        if not refs:
            QMessageBox.warning(self, "Empty", "Draw at least one annotation.")
            return
        from mindsight.GUI.path_picker import remember_vp_dir, vp_default_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Export portable VP archive", vp_default_dir(),
            f"Portable Visual Prompt (*{VP_ARCHIVE_EXT})")
        if not path:
            return
        remember_vp_dir(path)
        if not path.endswith(VP_ARCHIVE_EXT):
            path += VP_ARCHIVE_EXT
        try:
            export_vp_archive(path, self._classes, refs)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._set_status(f"Exported → {Path(path).name}")
        QMessageBox.information(
            self, "Exported",
            f"Portable archive saved:\n{path}\n\nIt contains the prompt AND "
            "all its images -- share it as one file; load it anywhere via "
            "Load VP File.")

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

    # -- File selection ----------------------------------------------------

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
        # Proposals belong to the image they were suggested on.
        self._pending_suggestions = []
        self._canvas.set_suggestions([])
        self._refresh_canvas()
        self._refresh_annotations_panel()

    # -- Class management --------------------------------------------------

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

    # -- Canvas / annotation management ------------------------------------

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
        """Click on a box in VP builder mode -- delete it."""
        if self._current_path is None:
            return
        anns = self._images[self._current_path]["annotations"]
        if 0 <= idx < len(anns):
            anns.pop(idx)
            self._refresh_canvas()
            self._refresh_annotations_panel()

    # -- Suggest mode (W3Z: FastSAM click-to-suggest) ----------------------

    def _on_suggest_toggled(self, checked: bool):
        if checked:
            from mindsight.GUI.region_suggest import fastsam_path
            if fastsam_path() is None:
                QMessageBox.information(
                    self, "Weight needed",
                    "Suggest mode needs the FastSAM-s segmentation weight "
                    "(24 MB).\n\nDownload it on the Models tab (SAM row), "
                    "then turn Suggest mode on again.")
                self._suggest_btn.setChecked(False)
                return
            self._set_status(
                "Suggest mode: click an object to get box proposals.")
        else:
            self._pending_suggestions = []
        self._canvas.set_suggest_mode(checked)

    def _on_suggest_point(self, ix: int, iy: int):
        if self._suggest_busy or self._current_path is None:
            return
        info = self._images.get(self._current_path)
        if info is None or info["frame"] is None:
            return
        if self._suggester is None:
            from mindsight.GUI.region_suggest import RegionSuggester
            self._suggester = RegionSuggester()
        frame, suggester = info["frame"], self._suggester
        self._suggest_busy = True
        self._set_status("Suggesting…")

        def work():
            try:
                self._suggest_q.put(("ok", suggester.suggest(frame, ix, iy)))
            except Exception as exc:                       # pragma: no cover
                self._suggest_q.put(("err", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self._suggest_timer.start(100)

    def _poll_suggest(self):
        try:
            kind, payload = self._suggest_q.get_nowait()
        except queue.Empty:
            return
        self._suggest_timer.stop()
        self._suggest_busy = False
        if kind == "err":
            self._set_status(f"Suggest failed: {payload}")
            return
        self._pending_suggestions = payload
        self._canvas.set_suggestions(payload)
        if payload:
            self._set_status(
                f"{len(payload)} proposal(s) — click one to accept, or "
                "click elsewhere to re-suggest.")
        else:
            self._set_status(
                "No region found there — try another point or draw the "
                "box by hand.")

    def _on_suggestion_accepted(self, idx: int):
        boxes = self._pending_suggestions
        if not (0 <= idx < len(boxes)) or self._current_path is None:
            return
        row = self._class_list.currentRow()
        if row < 0 or row >= len(self._classes):
            QMessageBox.warning(self, "No class selected",
                                "Add and select a class before accepting a "
                                "suggestion.")
            return
        x1, y1, x2, y2 = boxes[idx]
        cls = self._classes[row]
        self._images[self._current_path]["annotations"].append({
            "cls_id":   cls["id"],
            "cls_name": cls["name"],
            "bbox":     [int(x1), int(y1), int(x2), int(y2)],
        })
        self._pending_suggestions = []
        self._canvas.set_suggestions([])
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
        # Overlay test inference detections (different palette entry -- show at full opacity)
        test = self._test_dets.get(self._current_path, [])
        self._canvas.set_image_data(info["frame"], dets + test, [])
        n_ann  = len(info["annotations"])
        n_test = len(test)
        extras = f"  |  {n_test} test det(s)" if n_test else ""
        self._canvas_status.setText(
            f"{Path(self._current_path).name}  \u2014  {n_ann} annotation(s){extras}")

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
            label = (f'<font color="{col}">\u25a0</font>  '
                     f'<b>[{a["cls_id"]}] {a["cls_name"]}</b>  '
                     f'<small>({x1},{y1})\u2013({x2},{y2})</small>')
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(label)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            row_l.addWidget(lbl, 1)
            del_btn = QPushButton("\u2715")
            del_btn.setFixedWidth(22)
            del_btn.clicked.connect(lambda _, idx=i: self._on_box_delete(idx))
            row_l.addWidget(del_btn)
            self._ann_lay.addWidget(row_w)

    # -- Save / Load VP file -----------------------------------------------

    def _save_vp_file(self):
        if not self._classes:
            QMessageBox.warning(self, "Empty", "Define at least one class."); return
        has_anns = any(info["annotations"] for info in self._images.values())
        if not has_anns:
            QMessageBox.warning(self, "Empty", "Draw at least one annotation."); return

        from mindsight.GUI.path_picker import remember_vp_dir, vp_default_dir
        path, _ = QFileDialog.getSaveFileName(
            self, "Save VP file", vp_default_dir(),
            f"Visual Prompt (*{VP_EXT});;JSON (*.json)")
        if not path:
            return
        if not path.endswith(VP_EXT) and not path.endswith(".json"):
            path += VP_EXT
        remember_vp_dir(path)

        refs = [
            {"image": str(Path(img_path).resolve()),
             "annotations": [{"cls_id": a["cls_id"], "bbox": a["bbox"]}
                              for a in info["annotations"]]}
            for img_path, info in self._images.items()
            if info["annotations"]
        ]
        save_vp_file(path, self._classes, refs)
        self._last_saved_vp = path
        self._set_status(f"Saved \u2192 {Path(path).name}")
        QMessageBox.information(
            self, "Saved",
            f"Visual Prompt file saved:\n{path}\n\n"
            f"{len(self._classes)} class(es), {len(refs)} reference image(s)")

    def _load_vp_file(self):
        from .vp_archive import VP_ARCHIVE_EXT, import_vp_archive
        path, _ = QFileDialog.getOpenFileName(
            self, "Load VP file", "",
            f"Visual Prompt (*{VP_EXT} *{VP_ARCHIVE_EXT});;JSON (*.json);;"
            "All (*)")
        if not path:
            return
        try:
            if path.endswith(VP_ARCHIVE_EXT):
                # MP4: a portable archive extracts beside itself and yields a
                # normal .vp.json with absolute paths.
                path = str(import_vp_archive(path))
                self._set_status(f"Archive extracted → {Path(path).parent}")
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
            f"Loaded {Path(path).name}  \u2014  "
            f"{len(self._classes)} class(es), {len(data.get('references', []))} reference(s)")
        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    # -- Test inference ----------------------------------------------------

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
        self._set_status("Running inference\u2026")
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

    # -- Helpers -----------------------------------------------------------

    def _set_status(self, msg: str):
        self._status.setText(msg)

    def current_vp_path(self) -> str | None:
        """Return the last saved VP file path (for passing to GazeTab)."""
        return getattr(self, "_last_saved_vp", None)
