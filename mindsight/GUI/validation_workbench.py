"""
GUI/validation_workbench.py — Validation workbench pane (Layout B).

Lives on the Inference Tuning tab's right-hand splitter: the
tune -> validate loop against the settings CURRENTLY dialed into the
tab.  Set management (new / annotate / delete), a [Validate] button that
drives the ordinary GazeWorker over the set's video with the tab's
namespace (streams into a fresh run dir, detections stream on), and a
metrics table comparing the fresh score against the previous run's.

The tab injects ``namespace_provider`` (its ``_build_namespace``) and a
``worker_factory`` seam lets tests replace the real pipeline worker.
"""
from __future__ import annotations

import queue

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mindsight.validation import (
    ValidationSet,
    ValidationSetError,
    ValidationStore,
    allocate_run_dir,
    latest_score,
    prepare_validation_namespace,
    score_and_persist,
    validation_root,
)

_METRICS = [
    ("endpoint_px_mean", "mean px error", "{:.1f}"),
    ("endpoint_px_median", "median px", "{:.1f}"),
    ("endpoint_px_p95", "p95 px", "{:.0f}"),
    ("hit_rate", "gaze hit rate", "{:.0%}"),
    ("mae_deg_mean", "MAE (degrees)", "{:.1f}"),
    ("object_iou_mean", "object IoU", "{:.2f}"),
    ("offscreen_auc", "off-screen AUC", "{:.2f}"),
]


def _default_worker_factory(ns, frame_q, log_q):
    from mindsight.GUI.workers import GazeWorker
    return GazeWorker(ns, frame_q, log_q)


class ValidationWorkbench(QWidget):
    """Sets row · metrics table (run vs prev) · Validate button."""

    def __init__(self, namespace_provider, parent=None, store=None,
                 worker_factory=_default_worker_factory):
        super().__init__(parent)
        self._namespace_provider = namespace_provider
        self._store = store or ValidationStore(validation_root())
        self._worker_factory = worker_factory
        self._worker = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q: queue.Queue = queue.Queue()
        self._pending = None          # (vset, run_dir, ns) while running
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._on_poll)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Set:"))
        self._set_combo = QComboBox()
        self._set_combo.setMinimumWidth(140)
        row.addWidget(self._set_combo, 1)
        for text, slot, tip in (
                ("New…", self._on_new, "Create a validation set from a video"),
                ("Annotate…", self._on_annotate,
                 "Label gaze targets and object boxes for this set"),
                ("Delete", self._on_delete, "Delete the selected set")):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        lay.addLayout(row)

        self._table = QTableWidget(len(_METRICS), 2)
        self._table.setHorizontalHeaderLabels(["run", "prev"])
        self._table.setVerticalHeaderLabels([m[1] for m in _METRICS])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        run_row = QHBoxLayout()
        self._validate_btn = QPushButton("▶ Validate (current settings)")
        self._validate_btn.setToolTip(
            "Run the settings currently dialed into this tab over the "
            "set's video and score against its labels.")
        self._validate_btn.clicked.connect(self._on_validate)
        run_row.addWidget(self._validate_btn)
        self._status = QLabel("")
        run_row.addWidget(self._status, 1)
        lay.addLayout(run_row)

        self.refresh_sets()

    # ── Set management ───────────────────────────────────────────────────────

    def refresh_sets(self, select: str | None = None):
        current = select or self._set_combo.currentText()
        self._set_combo.blockSignals(True)
        self._set_combo.clear()
        for info in self._store.list_sets():
            self._set_combo.addItem(
                f"{info['name']}", info["name"])
        self._set_combo.blockSignals(False)
        if current:
            idx = self._set_combo.findData(current)
            if idx >= 0:
                self._set_combo.setCurrentIndex(idx)

    def _selected_name(self) -> str | None:
        return self._set_combo.currentData()

    def _on_new(self):
        name, ok = QInputDialog.getText(self, "New validation set",
                                        "Set name:")
        if not ok or not name.strip():
            return
        video, _ = QFileDialog.getOpenFileName(
            self, "Choose the set's source video", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if not video:
            return
        try:
            self._store.save(ValidationSet(name=name.strip(), video=video))
        except ValidationSetError as exc:
            QMessageBox.warning(self, "Cannot create set", str(exc))
            return
        self.refresh_sets(select=name.strip())

    def _on_annotate(self):
        name = self._selected_name()
        if not name:
            return
        try:
            vset = self._store.load(name)
        except ValidationSetError as exc:
            QMessageBox.warning(self, "Cannot open set", str(exc))
            return
        from mindsight.GUI.validation_annotator import (
            ValidationAnnotatorDialog,
        )
        ValidationAnnotatorDialog(vset, self._store, self).exec()
        self.refresh_sets(select=name)

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            return
        if QMessageBox.question(
                self, "Delete set",
                f"Delete validation set '{name}'? Labels are removed; "
                "past run results are kept.") \
                != QMessageBox.StandardButton.Yes:
            return
        self._store.delete(name)
        self.refresh_sets()

    # ── Validate ─────────────────────────────────────────────────────────────

    def _on_validate(self):
        if self._worker is not None:
            return
        name = self._selected_name()
        if not name:
            self._status.setText("Create a validation set first.")
            return
        try:
            vset = self._store.load(name)
            run_dir = allocate_run_dir(self._store.root, name)
            ns = prepare_validation_namespace(
                self._namespace_provider(), vset, run_dir)
        except ValidationSetError as exc:
            self._status.setText(str(exc))
            return
        self._pending = (vset, run_dir, ns)
        self._worker = self._worker_factory(ns, self._frame_q, self._log_q)
        self._validate_btn.setEnabled(False)
        self._status.setText(f"Validating '{name}'…")
        self._worker.start()
        self._poll.start(100)

    def _on_poll(self):
        done = False
        try:
            while True:
                item = self._frame_q.get_nowait()
                if item is None:            # worker sentinel
                    done = True
        except queue.Empty:
            pass
        while not self._log_q.empty():
            self._log_q.get_nowait()
        if not done:
            return
        self._poll.stop()
        self._worker = None
        vset, run_dir, ns = self._pending
        self._pending = None
        self._validate_btn.setEnabled(True)
        prev = latest_score(self._store.root, vset.name)
        try:
            result = score_and_persist(vset, run_dir, ns=ns)
        except ValidationSetError as exc:
            self._status.setText(str(exc))
            return
        self._show_scores(result, prev)
        self._status.setText(f"Scored {result['scored_points']} labels "
                             f"→ {run_dir.name}")

    def stop(self):
        """Cancel a running validation (tab teardown)."""
        if self._worker is not None and hasattr(self._worker, "stop"):
            self._worker.stop()

    # ── Report table ─────────────────────────────────────────────────────────

    def _show_scores(self, result: dict, prev: dict | None):
        for r, (key, _label, fmt) in enumerate(_METRICS):
            for c, source in enumerate((result, prev)):
                v = (source or {}).get(key)
                text = fmt.format(v) if v is not None else "—"
                self._table.setItem(r, c, QTableWidgetItem(text))
