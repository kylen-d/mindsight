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

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QCursor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
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
    QMenu,
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
    3. Tag objects on the canvas (hybrid grammar, v1.3.1): drag to draw a
       box, click an existing box to delete it, click an EMPTY spot to get
       FastSAM box proposals (numbered; click one to accept, Esc or
       right-click to dismiss).  The active class is switched with digits
       0-9 / Ctrl(Cmd)+Left/Right over the image, or by clicking the class
       list; accepting with no active class pops a class menu at the cursor.
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
        self._last_saved_vp: str | None = None
        self._test_dets: dict[str, list] = {}   # inference results per image
        self._vp_worker: VPInferenceWorker | None = None
        self._result_q: queue.Queue = queue.Queue()
        self._log_q:    queue.Queue = queue.Queue()
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_worker)
        # Suggest-on-click (W3Z, hybrid since v1.3.1): lazy FastSAM suggester
        # + its own poll loop.
        self._suggester = None
        self._suggest_q: queue.Queue = queue.Queue()
        self._suggest_busy = False
        self._suggest_for_path: str | None = None
        self._pending_suggestions: list = []
        self._suggest_timer = QTimer()
        self._suggest_timer.timeout.connect(self._poll_suggest)
        self._class_press_was_current = False
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

        tbl.addWidget(QFrame(frameShape=QFrame.Shape.VLine))

        fresh_btn = QPushButton("Start Fresh")
        fresh_btn.setToolTip(
            "Clear all reference images, annotations and classes and start "
            "a new visual prompt from scratch")
        fresh_btn.clicked.connect(self._start_fresh)
        tbl.addWidget(fresh_btn)

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
            "Image  [drag = draw box \u00b7 click box = delete \u00b7 "
            "click empty spot = suggest]")
        canvas_lay = QVBoxLayout(canvas_grp)
        self._canvas = ImageCanvas()
        self._canvas.set_hybrid_mode(True)
        self._canvas.box_toggled.connect(self._on_box_delete)
        self._canvas.crop_drawn.connect(self._on_box_drawn)
        self._canvas.point_clicked.connect(self._on_suggest_point)
        self._canvas.suggestion_accepted.connect(self._on_suggestion_accepted)
        self._canvas.suggestions_cleared.connect(self._on_suggestions_cleared)
        self._canvas.digit_pressed.connect(self._on_class_digit)
        self._canvas.cycle_pressed.connect(self._on_class_cycle)
        canvas_lay.addWidget(self._canvas)
        suggest_row = QHBoxLayout()
        self._active_chip = QLabel()
        self._active_chip.setTextFormat(Qt.TextFormat.RichText)
        self._active_chip.setStyleSheet("font-size:11px;")
        self._active_chip.setToolTip(
            "The class new boxes are tagged with. Over the image: press a "
            "class's number (0-9) to pick it, the same number again to "
            "clear, Ctrl/Cmd+Left/Right to cycle. Clicking the selected "
            "class in the list also clears it.")
        suggest_row.addWidget(self._active_chip)
        suggest_row.addStretch(1)
        self._suggest_chk = QCheckBox("Suggest on click")
        self._suggest_chk.setToolTip(
            "Click an empty spot on the image to get FastSAM box proposals "
            "instead of drawing by hand. Needs the FastSAM-s weight (24 MB) "
            "from the Models tab.")
        from mindsight.GUI.region_suggest import fastsam_path
        self._suggest_chk.setChecked(fastsam_path() is not None)
        self._suggest_chk.toggled.connect(self._on_suggest_toggled)
        suggest_row.addWidget(self._suggest_chk)
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

        cls_grp = QGroupBox("Classes  (pick by number key over the image)")
        cls_lay = QVBoxLayout(cls_grp)
        self._class_list = QListWidget()
        self._class_list.setMinimumHeight(100)
        self._class_list.currentRowChanged.connect(
            lambda _row: self._update_active_chip())
        self._class_list.itemClicked.connect(self._on_class_item_clicked)
        self._class_list.viewport().installEventFilter(self)
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
        self._update_active_chip()

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
        self._canvas.set_image_data(None, [], [])
        self._refresh_annotations_panel()

    # -- Start fresh --------------------------------------------------------

    def _start_fresh(self):
        """Clear the whole session (images, annotations, classes) in one go."""
        if self._vp_worker is not None and self._vp_worker.is_alive():
            self._set_status("Stop the test inference before starting fresh.")
            return
        if self._images or self._classes:
            reply = QMessageBox.question(
                self, "Start Fresh",
                "Start fresh? This clears all reference images, annotations "
                "and classes.\nUnsaved work is lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._reset_all()
        self._set_status("Started fresh.")

    def _reset_all(self):
        """Reset all session content; keeps the loaded FastSAM suggester and
        the remembered VP folder (tool state, not prompt content)."""
        self._images.clear()
        self._classes.clear()
        self._test_dets.clear()
        self._current_path = None
        self._pending_suggestions = []
        self._last_saved_vp = None
        self._file_list.clear()
        self._refresh_class_list()
        self._class_list.setCurrentRow(-1)
        self._canvas.set_image_data(None, [], [])
        self._canvas.set_suggestions([])
        self._refresh_annotations_panel()
        self._canvas_status.setText("Load reference images to begin.")
        self._update_active_chip()

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
        cls = self._require_class()
        if cls is None:
            self._set_status("Pick a class to tag the box, then redraw it.")
            return
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

    # -- Active class (v1.3.1 item 2: keys + chip + fallback popup) --------

    def eventFilter(self, obj, event):
        # Track whether a class-list click landed on the ALREADY-selected row
        # (that click deselects; a selection-changing click must not).
        if (obj is self._class_list.viewport()
                and event.type() == QEvent.Type.MouseButtonPress):
            item = self._class_list.itemAt(event.position().toPoint())
            row = self._class_list.row(item) if item is not None else -1
            self._class_press_was_current = (
                row >= 0 and row == self._class_list.currentRow())
        return super().eventFilter(obj, event)

    def _on_class_item_clicked(self, item):
        row = self._class_list.row(item)
        if self._class_press_was_current and row == self._class_list.currentRow():
            self._class_list.setCurrentRow(-1)
        self._class_press_was_current = False

    def _on_class_digit(self, digit: int):
        """Digit key over the canvas: pick class by its shown ID; same digit
        again clears the selection."""
        if not (0 <= digit < len(self._classes)):
            return
        if self._class_list.currentRow() == digit:
            self._class_list.setCurrentRow(-1)
        else:
            self._class_list.setCurrentRow(digit)

    def _on_class_cycle(self, step: int):
        n = len(self._classes)
        if n == 0:
            return
        cur = self._class_list.currentRow()
        self._class_list.setCurrentRow(0 if cur < 0 else (cur + step) % n)

    def _update_active_chip(self):
        row = self._class_list.currentRow()
        if 0 <= row < len(self._classes):
            c = self._classes[row]
            col = _palette_hex(c["id"])
            self._active_chip.setText(
                f'Active class: <span style="color:{col}"><b>&#9632; '
                f'[{c["id"]}] {c["name"]}</b></span>')
        else:
            self._active_chip.setText(
                'Active class: <i>none</i> — press its number over the '
                'image, or click a class')

    def _require_class(self) -> dict | None:
        """The active class, or a cursor popup to pick/create one."""
        row = self._class_list.currentRow()
        if 0 <= row < len(self._classes):
            return self._classes[row]
        return self._popup_class_choice()

    def _popup_class_choice(self) -> dict | None:
        menu = QMenu(self)
        acts = []
        for c in self._classes:
            pm = QPixmap(12, 12)
            pm.fill(QColor(_palette_hex(c["id"])))
            acts.append((menu.addAction(QIcon(pm),
                                        f'[{c["id"]}] {c["name"]}'), c))
        if self._classes:
            menu.addSeparator()
        new_act = menu.addAction("New class…")
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return None
        if chosen is new_act:
            before = len(self._classes)
            self._add_class()
            if len(self._classes) > before:
                row = self._class_list.currentRow()
                if 0 <= row < len(self._classes):
                    return self._classes[row]
            return None
        for act, c in acts:
            if act is chosen:
                self._class_list.setCurrentRow(c["id"])
                return c
        return None

    # -- Suggest on click (W3Z FastSAM, hybrid since v1.3.1) ---------------

    def _on_suggest_toggled(self, checked: bool):
        if checked:
            from mindsight.GUI.region_suggest import fastsam_path
            if fastsam_path() is None:
                QMessageBox.information(
                    self, "Weight needed",
                    "Suggest on click needs the FastSAM-s segmentation "
                    "weight (24 MB).\n\nDownload it on the Models tab (SAM "
                    "row), then turn Suggest on click back on.")
                self._suggest_chk.setChecked(False)
                return
            self._set_status(
                "Suggest on click: click an empty spot on the image to get "
                "box proposals.")
        else:
            self._pending_suggestions = []
            self._canvas.set_suggestions([])

    def _on_suggestions_cleared(self):
        self._pending_suggestions = []

    def _on_suggest_point(self, ix: int, iy: int):
        if not self._suggest_chk.isChecked():
            return
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
        self._suggest_for_path = self._current_path
        self._canvas.set_busy(True)
        self._set_status(
            "Suggesting…" if suggester.loaded
            else "Loading FastSAM (first use, one-time)…")

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
        self._canvas.set_busy(False)
        if kind == "err":
            self._set_status(f"Suggest failed: {payload}")
            return
        if self._suggest_for_path != self._current_path:
            return   # image changed (or session reset) while inferring
        self._pending_suggestions = payload
        self._canvas.set_suggestions(payload)
        if payload:
            self._set_status(
                f"{len(payload)} proposal(s) — click one to accept, Esc "
                "to dismiss, or click elsewhere to re-suggest.")
        elif getattr(self._suggester, "last_raw_count", 0):
            self._set_status(
                "FastSAM only found regions too large or too small to be "
                "useful here — draw the box by hand.")
        else:
            self._set_status(
                "No region found at that point — try another spot or "
                "draw the box by hand.")

    def _on_suggestion_accepted(self, idx: int):
        boxes = self._pending_suggestions
        if not (0 <= idx < len(boxes)) or self._current_path is None:
            return
        cls = self._require_class()
        if cls is None:
            self._set_status(
                "Pick a class to tag the proposal (it is still shown).")
            return
        x1, y1, x2, y2 = boxes[idx]
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

        self._reset_all()
        self._classes = list(data.get("classes", []))
        self._refresh_class_list()

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
        return self._last_saved_vp
