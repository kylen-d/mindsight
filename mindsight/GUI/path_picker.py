"""GUI/path_picker.py -- known-location file pickers (v1.1 W3Y item 5).

Model path fields across the GUI share the same shape: a free-text path
plus a Browse... button that used to open in the process CWD.  This module
adds the two missing conveniences without new dialogs:

* ``known_candidates(dest)`` -- the bare filenames already sitting in the
  dest's natural home (``Weights/<backend>/``), offered as an editable
  dropdown.  Bare names are what the pipeline stores (the no-absolute-
  paths rule); ``mindsight.weights.resolve_weight`` resolves them at use.
* ``default_browse_dir(dest)`` -- the directory the Browse dialog opens
  in (the same natural home; empty string keeps Qt's default).

``KnownPathCombo`` is a drop-in replacement for the ``QLineEdit`` these
fields used: editable, candidates pre-listed, and ``text``/``setText``/
``textChanged``/``setPlaceholderText`` kept API-compatible so the
surrounding collect/apply code does not change.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox

#: dest -> Weights/<backend> subdirectory holding its candidates.
_DEST_BACKEND = {
    "model": "YOLO",
    "vp_model": "YOLO",
    "mgaze_model": "MGaze",
    "rf_gazelle_model": "Gazelle",
    "gazelle_model": "Gazelle",
}

#: File suffixes that count as weights when enumerating a backend dir.
_WEIGHT_SUFFIXES = {".pt", ".onnx", ".ts", ".engine", ".mlpackage"}


def _backend_dir(dest: str) -> Path | None:
    backend = _DEST_BACKEND.get(dest)
    if backend is None:
        return None
    from mindsight.weights import WEIGHTS_ROOT
    d = Path(WEIGHTS_ROOT) / backend
    return d if d.is_dir() else None


def known_candidates(dest: str) -> list[str]:
    """Bare filenames available for *dest* in its Weights/<backend> home."""
    d = _backend_dir(dest)
    if d is None:
        return []
    try:
        return sorted(p.name for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES
                      and not p.name.startswith("."))
    except OSError:
        return []


def default_browse_dir(dest: str) -> str:
    """Directory the Browse dialog should open in ('' = Qt default)."""
    if dest == "vp_file":
        return vp_default_dir()
    d = _backend_dir(dest)
    return str(d) if d is not None else ""


def projects_default_dir() -> str:
    """Where project-folder dialogs open: beside the most recent project."""
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        for p in SettingsManager().list_recent_projects():
            parent = Path(p).parent
            if parent.is_dir():
                return str(parent)
    except Exception:
        pass
    return ""


def vp_default_dir() -> str:
    """Where VP-file dialogs open: the last dir a .vp.json was saved to or
    picked from (recorded via remember_vp_dir), '' = Qt default."""
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        d = SettingsManager().load_gui_state().get("vp_last_dir", "")
        return d if d and Path(d).is_dir() else ""
    except Exception:
        return ""


def remember_vp_dir(path: str) -> None:
    """Record the directory of a just-saved/just-picked VP file."""
    from mindsight.GUI.settings_manager import SettingsManager
    try:
        SettingsManager().save_gui_state(
            {"vp_last_dir": str(Path(path).resolve().parent)})
    except Exception:
        pass


class KnownPathCombo(QComboBox):
    """Editable path field with a dropdown of known candidates.

    API-compatible with the QLineEdit it replaces: ``text()``,
    ``setText()``, ``textChanged`` and ``setPlaceholderText`` behave the
    same, so existing collect/apply/browse code keeps working unchanged.
    """

    textChanged = pyqtSignal(str)

    def __init__(self, candidates: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        if candidates:
            self.addItems(candidates)
        self.setEditText("")
        self.editTextChanged.connect(self.textChanged.emit)

    # -- QLineEdit-compatible surface ----------------------------------------

    def text(self) -> str:
        return self.currentText()

    def setText(self, value) -> None:
        self.setEditText("" if value is None else str(value))

    def setPlaceholderText(self, text: str) -> None:
        le = self.lineEdit()
        if le is not None:
            le.setPlaceholderText(text)
