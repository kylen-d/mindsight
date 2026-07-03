"""
models_tab.py
-------------
The **Models** tab -- a manifest-driven weight manager (SP4.1 Batch F, T10).

One row per ``weights_manifest.json`` entry: paper-term label (Gaze-LLE /
MobileGaze), backend, required flag, a "needed now" marker for the weights the
current Gaze Tuning config resolves (``provenance.collect_weights`` -- the read-
only view the SP4 stub showed is kept as this column), on-disk size, and a
verification state (OK / mismatch / missing / auto-fetch).  Missing downloadable
weights get an **Install** button; present ones get **Verify** and **Re-download**.
All downloads and checksum hashing run OFF the GUI thread
(:class:`~mindsight.GUI.workers.WeightsDownloadWorker` + a poll timer) so the tab
stays responsive; failures surface as a readable status line, never a dialog
storm (G-OFFLINE UX).

Config-needed weights that are NOT manifest entries render as read-only
``unmanaged`` rows (custom/user weights) so the current-config view survives.

The tab writes NO namespace dests (GUI namespace census UNCHANGED, T10).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mindsight import weights
from .workers import WeightsDownloadWorker

# Row state -> (display text, colour).
_STATE_STYLE = {
    "ok": ("OK", "#2a7a2a"),
    "mismatch": ("mismatch", "#b8860b"),
    "missing": ("MISSING", "#b22222"),
    "present": ("present (unverified)", "#888"),
    "auto-fetch": ("auto-fetch on first use", "#888"),
    "unmanaged": ("unmanaged (custom weight)", "#888"),
}

_COLS = ["Model", "Backend", "Required", "Needed now", "State", "On disk", "Actions"]


class ModelsTab(QWidget):
    """Manifest-driven weight manager: install, verify, re-download."""

    def __init__(self, gaze_tab=None, parent=None, *,
                 manifest_path=None, weights_root=None):
        super().__init__(parent)
        self._gaze_tab = gaze_tab
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._weights_root = Path(weights_root) if weights_root else None
        self._q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._row_info: list[dict] = []      # per-row: entry/dest/kind
        self._entry_row: dict = {}           # (backend, filename) -> row
        self._build_ui()
        self._timer = None
        self.refresh()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        note = QLabel(
            "Model weights from the checksummed manifest. Install missing "
            "weights, verify them against the published checksum, or "
            "re-download a file that no longer matches.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        lay.addWidget(note)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        self._disk = QLabel("")
        self._disk.setStyleSheet("color: #888;")
        lay.addWidget(self._disk)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        bar = QHBoxLayout()
        verify_all = QPushButton("Verify all")
        verify_all.clicked.connect(self._verify_all)
        bar.addWidget(verify_all)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)
        bar.addStretch(1)
        lay.addLayout(bar)

    # ── Path / manifest helpers ──────────────────────────────────────────────

    def _entries(self) -> list[dict]:
        try:
            return weights.manifest_entries(self._manifest_path)
        except weights.WeightsError:
            return []

    def _entry_dest(self, entry: dict) -> Path:
        if self._weights_root is not None:
            return self._weights_root / entry["backend"] / entry["filename"]
        return weights.entry_dest(entry)

    def _needed_now(self) -> set[str]:
        """Basenames of the weights the current Gaze Tuning config resolves."""
        if self._gaze_tab is None:
            return set()
        try:
            from mindsight.outputs import provenance
            ns = self._gaze_tab._build_namespace()
            collected = provenance.collect_weights(ns)
        except Exception:
            return set()
        return {Path(w.get("resolved", "")).name for w in collected.values()}

    def _state_for(self, entry: dict, dest: Path) -> str:
        if entry.get("source") == weights.SOURCE_ULTRALYTICS_AUTO:
            return "auto-fetch"
        if not dest.exists():
            return "missing"
        return "present"

    # ── Rendering ────────────────────────────────────────────────────────────

    def refresh(self):
        """Rebuild every row from the manifest (presence only -- no hashing)."""
        self._status.setText("")
        self._status.setStyleSheet("")
        entries = self._entries()
        needed = self._needed_now()
        manifest_names = {e["filename"] for e in entries}

        self._row_info = []
        self._entry_row = {}
        for e in entries:
            dest = self._entry_dest(e)
            self._row_info.append({
                "entry": e, "dest": dest, "kind": "managed",
                "state": self._state_for(e, dest),
            })

        # Config-needed custom weights with no manifest entry -> unmanaged rows.
        for name in sorted(needed - manifest_names):
            self._row_info.append({
                "entry": None, "dest": None, "kind": "unmanaged",
                "state": "unmanaged", "name": name,
            })

        self._table.setRowCount(len(self._row_info))
        for row, info in enumerate(self._row_info):
            self._render_row(row, info, needed)
        self._update_disk()

    def _render_row(self, row: int, info: dict, needed: set[str]):
        if info["kind"] == "unmanaged":
            name = info["name"]
            self._set(row, 0, name)
            self._set(row, 1, "custom")
            self._set(row, 2, "")
            self._set(row, 3, "yes")
            self._set_state(row, "unmanaged")
            self._set(row, 5, "")
            self._table.removeCellWidget(row, 6)
            return

        entry = info["entry"]
        self._entry_row[(entry["backend"], entry["filename"])] = row
        self._set(row, 0, entry.get("label", entry["filename"]))
        self._set(row, 1, entry["backend"])
        self._set(row, 2, "required" if entry.get("required") else "")
        self._set(row, 3, "yes" if entry["filename"] in needed else "")
        self._set_state(row, info["state"])
        self._set(row, 5, self._disk_text(info["dest"]))
        self._set_actions(row, info)

    def _disk_text(self, dest: Path) -> str:
        try:
            return f"{dest.stat().st_size / 1e6:.1f} MB"
        except OSError:
            return "-"

    def _set_actions(self, row: int, info: dict):
        entry, state = info["entry"], info["state"]
        holder = QWidget()
        hb = QHBoxLayout(holder)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)
        downloadable = (entry.get("url")
                        and entry.get("source") != weights.SOURCE_ULTRALYTICS_AUTO)
        if state == "auto-fetch":
            pass  # fetched by Ultralytics on first use; nothing to do here
        elif state == "missing":
            if downloadable:
                btn = QPushButton("Install")
                btn.clicked.connect(lambda _=False, r=row: self._start_download(r))
                hb.addWidget(btn)
        else:  # present / ok / mismatch
            vbtn = QPushButton("Verify")
            vbtn.clicked.connect(lambda _=False, r=row: self._verify_row(r))
            hb.addWidget(vbtn)
            if downloadable:
                rbtn = QPushButton("Re-download")
                rbtn.clicked.connect(lambda _=False, r=row: self._start_download(r))
                hb.addWidget(rbtn)
        hb.addStretch(1)
        self._table.setCellWidget(row, 6, holder)

    def _update_disk(self):
        root = self._weights_root or weights.WEIGHTS_ROOT
        total = 0
        try:
            for p in Path(root).rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except OSError:
            pass
        self._disk.setText(f"Weights on disk: {total / 1e6:.1f} MB total")

    # ── Cell setters ─────────────────────────────────────────────────────────

    def _set(self, row, col, text, colour=None):
        item = QTableWidgetItem(str(text))
        if colour:
            from PyQt6.QtGui import QColor
            item.setForeground(QColor(colour))
        self._table.setItem(row, col, item)

    def _set_state(self, row, state):
        text, colour = _STATE_STYLE.get(state, (state, "#888"))
        self._set(row, 4, text, colour)
        if row < len(self._row_info):
            self._row_info[row]["state"] = state

    # ── Verify (async, off the GUI thread) ───────────────────────────────────

    def _managed_present_rows(self) -> list[int]:
        return [r for r, info in enumerate(self._row_info)
                if info["kind"] == "managed"
                and info["state"] in ("present", "ok", "mismatch")]

    def _verify_all(self):
        rows = self._managed_present_rows()
        if not rows:
            self._status.setText("No present weights to verify.")
            return
        self._status.setText(f"Verifying {len(rows)} weight(s)...")
        t = threading.Thread(target=self._verify_worker, args=(rows,), daemon=True)
        self._threads.append(t)
        t.start()
        self._ensure_timer()

    def _verify_row(self, row: int):
        info = self._row_info[row]
        if info["kind"] != "managed":
            return
        self._status.setText(f"Verifying {info['entry']['filename']}...")
        t = threading.Thread(target=self._verify_worker, args=([row],), daemon=True)
        self._threads.append(t)
        t.start()
        self._ensure_timer()

    def _verify_worker(self, rows):
        for row in rows:
            info = self._row_info[row]
            state = weights.verify(info["dest"], info["entry"])
            self._q.put(("vstate", row, state))
        self._q.put(("vdone", None, None))

    # ── Download (async) ─────────────────────────────────────────────────────

    def _start_download(self, row: int):
        info = self._row_info[row]
        entry = info["entry"]
        self._set_state(row, "present")  # optimistic "working" cue
        self._set(row, 4, "downloading...", "#888")
        self._status.setStyleSheet("")
        self._status.setText(f"Downloading {entry['filename']} ...")
        worker = WeightsDownloadWorker(
            [entry], self._q, dest_for=self._entry_dest)
        self._threads.append(worker)
        worker.start()
        self._ensure_timer()

    # ── Result pump (queue drained on the GUI thread) ────────────────────────

    def _ensure_timer(self):
        if self._timer is None:
            from PyQt6.QtCore import QTimer
            self._timer = QTimer(self)
            self._timer.setInterval(150)
            self._timer.timeout.connect(self._drain)
            self._timer.start()

    def _drain(self):
        """Apply worker results on the GUI thread. Safe to call directly (tests)."""
        try:
            while True:
                kind, ref, payload = self._q.get_nowait()
                if kind == "log":
                    self._status.setText(str(payload))
                elif kind == "vstate":
                    self._set_state(ref, payload)
                    if payload in ("ok", "mismatch"):
                        self._set_actions(ref, self._row_info[ref])
                elif kind == "done":
                    self._on_download_done(ref)
                elif kind == "error":
                    self._on_download_error(ref, str(payload))
                elif kind in ("finished", "vdone"):
                    pass
        except queue.Empty:
            pass

    def _on_download_done(self, entry: dict):
        row = self._entry_row.get((entry["backend"], entry["filename"]))
        if row is None:
            return
        info = self._row_info[row]
        state = weights.verify(info["dest"], entry)   # small: just-fetched file
        self._set_state(row, state)
        self._set(row, 5, self._disk_text(info["dest"]))
        self._set_actions(row, info)
        self._status.setText(f"{entry['filename']}: downloaded and {state}.")
        self._update_disk()

    def _on_download_error(self, entry: dict, msg: str):
        row = self._entry_row.get((entry["backend"], entry["filename"]))
        if row is not None:
            info = self._row_info[row]
            self._set_state(row, self._state_for(entry, info["dest"]))
            self._set_actions(row, info)
        self._status.setText(f"Download failed: {msg}")
        self._status.setStyleSheet("color: #b22222;")
