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
import time
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mindsight.validation import (
    ValidationSetError,
    ValidationStore,
    allocate_run_dir,
    latest_score,
    prepare_validation_namespace,
    score_and_persist,
    validation_root,
)

# Gaze-targets-only v1 (user ruling): the object-IoU row is shelved with
# the object annotator; score.json still records it for sets that carry
# boxes from earlier versions.
_METRICS = [
    ("endpoint_px_mean", "mean px error", "{:.1f}"),
    ("endpoint_px_median", "median px", "{:.1f}"),
    ("endpoint_px_p95", "p95 px", "{:.0f}"),
    ("hit_rate", "gaze hit rate", "{:.0%}"),
    ("mae_deg_mean", "MAE (degrees)", "{:.1f}"),
    ("offscreen_auc", "off-screen AUC", "{:.2f}"),
    ("avg_fps", "avg fps", "{:.1f}"),
]


def _default_worker_factory(ns, frame_q, log_q):
    from mindsight.GUI.workers import GazeWorker
    return GazeWorker(ns, frame_q, log_q)


class ValidationWorkbench(QWidget):
    """Sets row · metrics table (run vs prev) · Validate button."""

    def __init__(self, namespace_provider, parent=None, store=None,
                 worker_factory=_default_worker_factory,
                 namespace_applier=None):
        super().__init__(parent)
        self._namespace_provider = namespace_provider
        self._namespace_applier = namespace_applier
        self._store = store or ValidationStore(validation_root())
        self._worker_factory = worker_factory
        self._worker = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q: queue.Queue = queue.Queue()
        self._pending = None          # (vset, run_dir, ns) while running
        self._frames_done = 0
        self._t_first_frame = None
        self._total_frames = 0
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
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.stop)
        run_row.addWidget(self._cancel_btn)
        autotune_btn = QPushButton("Auto-tune…")
        autotune_btn.setToolTip(
            "Sweep one or two knobs over this set and apply the best "
            "combination back to the tab.")
        autotune_btn.clicked.connect(self._on_autotune)
        run_row.addWidget(autotune_btn)
        history_btn = QPushButton("History…")
        history_btn.setToolTip(
            "All scored runs for this set, with the settings that "
            "changed between runs.")
        history_btn.clicked.connect(self._on_history)
        run_row.addWidget(history_btn)
        embed_btn = QPushButton("Embed…")
        embed_btn.setToolTip(
            "Write this set's latest score into a pipeline YAML's "
            "validation: block (metadata only -- never affects runs or "
            "resume).")
        embed_btn.clicked.connect(self._on_embed)
        run_row.addWidget(embed_btn)
        run_row.addStretch(1)
        lay.addLayout(run_row)

        # Live run feedback: frame counter / fps / ETA + progress bar.
        progress_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        progress_row.addWidget(self._progress, 1)
        lay.addLayout(progress_row)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

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
        from mindsight.GUI.validation_wizard import ValidationSetWizard
        wizard = ValidationSetWizard(self._store, parent=self)
        wizard.exec()
        created = wizard._vset.name if wizard._vset is not None else None
        self.refresh_sets(select=created)

    def _on_annotate(self):
        name = self._selected_name()
        if not name:
            return
        try:
            vset = self._store.load(name)
        except ValidationSetError as exc:
            QMessageBox.warning(self, "Cannot open set", str(exc))
            return
        from mindsight.GUI.validation_wizard import ValidationSetWizard
        ValidationSetWizard(self._store, vset, parent=self).exec()
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

    def _on_history(self):
        name = self._selected_name()
        if not name:
            return
        from mindsight.GUI.validation_history import ValidationHistoryDialog
        ValidationHistoryDialog(self._store, name, self).exec()

    def _on_autotune(self):
        if self._worker is not None:      # never two pipelines at once
            return
        name = self._selected_name()
        if not name:
            self._status.setText("Create a validation set first.")
            return
        from mindsight.GUI.validation_autotune import AutoTuneDialog
        AutoTuneDialog(
            self._store, name, self._namespace_provider,
            self._namespace_applier or (lambda ns: None),
            parent=self, worker_factory=self._worker_factory).exec()

    def _on_embed(self):
        name = self._selected_name()
        if not name:
            return
        from mindsight.validation import (
            embed_validation_summary,
            validation_summary_block,
        )
        score = latest_score(self._store.root, name)
        if score is None:
            self._status.setText("Validate first — no score to embed.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Pipeline file to embed the summary into", "",
            "YAML (*.yaml *.yml);;All files (*)")
        if not path:
            return
        import datetime
        try:
            vset = self._store.load(name)
            block = validation_summary_block(
                vset, score,
                date=datetime.date.today().isoformat())
            embed_validation_summary(path, block)
        except ValidationSetError as exc:
            QMessageBox.warning(self, "Cannot embed", str(exc))
            return
        self._status.setText(f"Embedded '{name}' summary into "
                             f"{Path(path).name}")

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
        self._frames_done = 0
        self._t_first_frame = None
        self._total_frames = self._probe_frame_count(vset.video)
        self._progress.setRange(0, self._total_frames or 0)  # 0,0 = busy bar
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._worker = self._worker_factory(ns, self._frame_q, self._log_q)
        self._validate_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status.setText(f"Validating '{name}' — loading models…")
        self._worker.start()
        self._poll.start(100)

    @staticmethod
    def _probe_frame_count(video: str) -> int:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return max(total, 0)
        except Exception:
            return 0

    def _live_fps(self) -> float | None:
        if self._t_first_frame is None or self._frames_done < 2:
            return None
        elapsed = time.monotonic() - self._t_first_frame
        return (self._frames_done - 1) / elapsed if elapsed > 0 else None

    def _on_poll(self):
        done = False
        try:
            while True:
                item = self._frame_q.get_nowait()
                if item is None:            # worker sentinel
                    done = True
                else:
                    if self._t_first_frame is None:
                        self._t_first_frame = time.monotonic()
                    self._frames_done += 1
        except queue.Empty:
            pass
        while not self._log_q.empty():
            self._log_q.get_nowait()
        if not done:
            if self._frames_done:
                self._progress.setValue(
                    min(self._frames_done, self._total_frames)
                    if self._total_frames else 0)
                fps = self._live_fps()
                total = (f"/{self._total_frames}"
                         if self._total_frames else "")
                parts = [f"frame {self._frames_done}{total}"]
                if fps:
                    parts.append(f"{fps:.1f} fps")
                    if self._total_frames:
                        remaining = self._total_frames - self._frames_done
                        if remaining > 0:
                            parts.append(f"ETA {remaining / fps:.0f}s")
                self._status.setText("Validating — " + " · ".join(parts))
            return
        self._poll.stop()
        self._worker = None
        vset, run_dir, ns = self._pending
        self._pending = None
        self._validate_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        avg_fps = self._live_fps()
        prev = latest_score(self._store.root, vset.name)
        try:
            result = score_and_persist(
                vset, run_dir, ns=ns,
                extra={"avg_fps": avg_fps} if avg_fps else None)
        except ValidationSetError as exc:
            self._status.setText(str(exc))
            return
        self._show_scores(result, prev)
        self._status.setText(f"Scored {result['scored_points']} labels "
                             f"→ {run_dir.name}")

    def stop(self):
        """Cancel a running validation (Cancel button / tab teardown).

        The worker finishes the current frame and sends its sentinel;
        whatever ran still gets scored, so a cancelled run is a shorter
        run, not a lost one."""
        if self._worker is not None and hasattr(self._worker, "stop"):
            self._worker.stop()
            self._status.setText("Cancelling — finishing the current frame…")

    # ── Report table ─────────────────────────────────────────────────────────

    def _show_scores(self, result: dict, prev: dict | None):
        for r, (key, _label, fmt) in enumerate(_METRICS):
            for c, source in enumerate((result, prev)):
                v = (source or {}).get(key)
                text = fmt.format(v) if v is not None else "—"
                self._table.setItem(r, c, QTableWidgetItem(text))
