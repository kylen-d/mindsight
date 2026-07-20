"""
GUI/validation_autotune.py — Auto-tune sweep dialog (W4C item 1).

"Sweep these knobs over this set", in-app: pick one or two knobs from
the curated list (ruling R6), give each a comma-separated value list,
and every combination runs sequentially through the ordinary validation
runner — each an ordinary ``run-NNN`` History already understands.
Results land in a table sorted by mean px error; [Apply best to tab]
writes the winning knob values back through the tab's
``apply_namespace`` (the census seam) so [Validate] immediately
reproduces the winner.

The sweep manifest (``.runs/<set>/sweep-NNN.json``) persists after
every combo, so a cancelled sweep keeps its completed scores and the
dialog reopens on the last sweep's table.
"""
from __future__ import annotations

import queue
import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from mindsight.validation import (
    CURATED_KNOBS,
    ValidationSetError,
    allocate_run_dir,
    allocate_sweep_path,
    estimate_seconds,
    expand_combos,
    latest_score,
    latest_sweep,
    new_sweep_manifest,
    pick_winner,
    prepare_sweep_namespace,
    save_sweep,
    score_and_persist,
)

_NONE_LABEL = "— none —"

#: Score columns after the knob-value columns.
_SCORE_COLS = [
    ("endpoint_px_mean", "mean px", "{:.1f}"),
    ("endpoint_px_median", "median px", "{:.1f}"),
    ("hit_rate", "hit rate", "{:.0%}"),
    ("avg_fps", "avg fps", "{:.1f}"),
]


def _default_worker_factory(ns, frame_q, log_q):
    from mindsight.GUI.workers import GazeWorker
    return GazeWorker(ns, frame_q, log_q)


class AutoTuneDialog(QDialog):
    """Knob picker · run/cancel with per-combo progress · results table
    sorted by mean px · Apply best to tab."""

    def __init__(self, store, set_name, namespace_provider,
                 namespace_applier, parent=None,
                 worker_factory=_default_worker_factory):
        super().__init__(parent)
        self.setWindowTitle(f"Auto-tune — {set_name}")
        self._store = store
        self._set_name = set_name
        self._namespace_provider = namespace_provider
        self._namespace_applier = namespace_applier
        self._worker_factory = worker_factory

        self._worker = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=4)
        self._log_q: queue.Queue = queue.Queue()
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._on_poll)
        self._cancelled = False
        self._combos: list[dict] = []
        self._combo_idx = 0
        self._base_ns = None
        self._vset = None
        self._manifest = None
        self._manifest_path = None
        self._pending = None            # (run_dir, ns) while a combo runs
        self._frames_done = 0
        self._t_first_frame = None
        self._total_frames = 0

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Sweep one or two knobs over this set. Every combination "
            "runs with the settings currently dialed into the tab; only "
            "the swept knobs change."))

        self._knob_rows = []
        for i in range(2):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Knob {i + 1}:"))
            combo = QComboBox()
            if i == 1:
                combo.addItem(_NONE_LABEL, None)
            for dest, label, _cast in CURATED_KNOBS:
                combo.addItem(label, dest)
            values = QLineEdit()
            values.setPlaceholderText("values, comma-separated — e.g. "
                                      "1.0, 1.1, 1.2")
            combo.currentIndexChanged.connect(self._update_estimate)
            values.textChanged.connect(self._update_estimate)
            row.addWidget(combo)
            row.addWidget(values, 1)
            lay.addLayout(row)
            self._knob_rows.append((combo, values))

        self._estimate = QLabel("")
        lay.addWidget(self._estimate)

        run_row = QHBoxLayout()
        self._start_btn = QPushButton("▶ Run sweep")
        self._start_btn.clicked.connect(self._on_start)
        run_row.addWidget(self._start_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setToolTip(
            "Finish the current combination and stop; completed "
            "combinations keep their scores.")
        self._cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch(1)
        self._apply_btn = QPushButton("Apply best to tab")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Write the winning knob values into the tab's settings so "
            "Validate reproduces the best run.")
        self._apply_btn.clicked.connect(self._on_apply_best)
        run_row.addWidget(self._apply_btn)
        lay.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        self._update_estimate()
        previous = latest_sweep(self._store.root, self._set_name)
        if previous:
            self._manifest = previous
            self._show_results(previous)
            self._status.setText("Showing the set's last sweep.")

    # ── Knob parsing / estimate ──────────────────────────────────────────────

    def _parse_knobs(self) -> list[tuple[str, list]]:
        """[(dest, [typed values…]), …] from the picker rows; raises
        ValidationSetError with a plain-English message on bad input."""
        casts = {dest: cast for dest, _label, cast in CURATED_KNOBS}
        labels = {dest: label for dest, label, _cast in CURATED_KNOBS}
        knobs = []
        for combo, values in self._knob_rows:
            dest = combo.currentData()
            if dest is None:
                continue
            text = values.text().strip()
            if not text:
                raise ValidationSetError(
                    f"Knob {labels[dest]!r} has no values.")
            try:
                parsed = [casts[dest](tok.strip())
                          for tok in text.split(",") if tok.strip()]
            except ValueError as exc:
                raise ValidationSetError(
                    f"Bad value for {labels[dest]!r}: {exc}") from exc
            knobs.append((dest, parsed))
        return knobs

    def _update_estimate(self):
        try:
            combos = expand_combos(self._parse_knobs())
        except ValidationSetError as exc:
            self._estimate.setText(str(exc))
            return
        parts = [f"{len(combos)} combination"
                 + ("s" if len(combos) != 1 else "")]
        frames = self._clip_frames()
        score = latest_score(self._store.root, self._set_name)
        secs = estimate_seconds(len(combos), frames,
                                (score or {}).get("avg_fps"))
        if secs is not None:
            parts.append(f"est. ~{secs / 60:.0f} min"
                         if secs >= 90 else f"est. ~{secs:.0f} s")
        else:
            parts.append("time unknown — Validate once to measure fps")
        self._estimate.setText(" · ".join(parts))

    def _clip_frames(self) -> int:
        try:
            vset = self._vset or self._store.load(self._set_name)
        except ValidationSetError:
            return 0
        from mindsight.GUI.validation_workbench import ValidationWorkbench
        return ValidationWorkbench._probe_frame_count(vset.video)

    # ── Sweep loop ───────────────────────────────────────────────────────────

    def _on_start(self):
        if self._worker is not None:
            return
        try:
            knobs = self._parse_knobs()
            self._combos = expand_combos(knobs)
            self._vset = self._store.load(self._set_name)
            self._base_ns = self._namespace_provider()
        except ValidationSetError as exc:
            self._status.setText(str(exc))
            return
        self._manifest = new_sweep_manifest(self._set_name, knobs)
        self._manifest_path = allocate_sweep_path(
            self._store.root, self._set_name)
        save_sweep(self._manifest_path, self._manifest)
        self._cancelled = False
        self._combo_idx = 0
        self._total_frames = self._clip_frames()
        self._table.setRowCount(0)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._next_combo()

    def _next_combo(self):
        if self._cancelled or self._combo_idx >= len(self._combos):
            self._finish()
            return
        overrides = self._combos[self._combo_idx]
        try:
            run_dir = allocate_run_dir(self._store.root, self._set_name)
            ns = prepare_sweep_namespace(
                self._base_ns, self._vset, run_dir, overrides)
        except ValidationSetError as exc:
            self._record_result(overrides, None, None, str(exc))
            self._combo_idx += 1
            self._next_combo()
            return
        self._pending = (run_dir, ns)
        self._frames_done = 0
        self._t_first_frame = None
        self._progress.setRange(0, self._total_frames or 0)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText(
            f"Combo {self._combo_idx + 1}/{len(self._combos)} "
            f"({self._describe(overrides)}) — loading models…")
        self._worker = self._worker_factory(ns, self._frame_q, self._log_q)
        self._worker.start()
        self._poll.start(100)

    @staticmethod
    def _describe(overrides: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in overrides.items())

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
                if item is None:
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
            if self._frames_done and self._total_frames:
                self._progress.setValue(
                    min(self._frames_done, self._total_frames))
            return
        self._poll.stop()
        self._worker = None
        run_dir, ns = self._pending
        self._pending = None
        overrides = self._combos[self._combo_idx]
        avg_fps = self._live_fps()
        try:
            score = score_and_persist(
                self._vset, run_dir, ns=ns,
                extra={"avg_fps": avg_fps} if avg_fps else None)
            self._record_result(overrides, run_dir.name, score, None)
        except ValidationSetError as exc:
            self._record_result(overrides, run_dir.name, None, str(exc))
        self._combo_idx += 1
        self._next_combo()

    def _record_result(self, overrides, run_name, score, error):
        self._manifest["results"].append({
            "overrides": overrides, "run": run_name,
            "score": score, "error": error})
        save_sweep(self._manifest_path, self._manifest)

    def _on_cancel(self):
        self._cancelled = True
        if self._worker is not None and hasattr(self._worker, "stop"):
            self._worker.stop()
            self._status.setText(
                "Cancelling — finishing the current combination…")

    def _finish(self):
        results = self._manifest["results"]
        self._manifest["winner"] = pick_winner(results)
        save_sweep(self._manifest_path, self._manifest)
        self._progress.setVisible(False)
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._show_results(self._manifest)
        scored = sum(1 for r in results if r.get("score"))
        note = ("cancelled — " if self._cancelled
                and len(results) < len(self._combos) else "")
        self._status.setText(
            f"Sweep {note}{scored}/{len(results)} combinations scored."
            + ("" if scored else " Nothing scored — check the set."))

    # ── Results table / apply ────────────────────────────────────────────────

    def _winner_overrides(self) -> dict | None:
        if not self._manifest:
            return None
        winner = self._manifest.get("winner")
        if winner is None:
            return None
        return self._manifest["results"][winner]["overrides"]

    def _show_results(self, manifest: dict):
        knob_dests = [k[0] for k in manifest.get("knobs", [])]
        results = manifest.get("results", [])
        winner = manifest.get("winner")
        headers = knob_dests + [c[1] for c in _SCORE_COLS] + ["run"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(results))

        def sort_key(pair):
            score = pair[1].get("score") or {}
            mean = score.get("endpoint_px_mean")
            return (mean is None, mean if mean is not None else 0.0)

        for row, (idx, entry) in enumerate(
                sorted(enumerate(results), key=sort_key)):
            score = entry.get("score") or {}
            cells = [str(entry["overrides"].get(d, "")) for d in knob_dests]
            for key, _label, fmt in _SCORE_COLS:
                v = score.get(key)
                cells.append(fmt.format(v) if v is not None else "—")
            cells.append(entry.get("run") or (entry.get("error") or "—"))
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if idx == winner:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._apply_btn.setEnabled(self._winner_overrides() is not None)

    def _on_apply_best(self):
        overrides = self._winner_overrides()
        if overrides is None:
            return
        ns = self._namespace_provider()
        for dest, value in overrides.items():
            setattr(ns, dest, value)
        self._namespace_applier(ns)
        self._status.setText(
            f"Applied to tab: {self._describe(overrides)} — Validate "
            "now reproduces the winning run.")

    # ── Teardown ─────────────────────────────────────────────────────────────

    # Esc / window close while a combo runs = cancel, not abandon: the
    # running combo can only finish scoring through _on_poll, so the
    # dialog must stay open until the worker sends its sentinel.

    def reject(self):
        if self._worker is not None:
            self._on_cancel()
            return
        super().reject()

    def closeEvent(self, event):  # noqa: N802 (Qt override)
        if self._worker is not None:
            self._on_cancel()
            event.ignore()
            return
        super().closeEvent(event)
