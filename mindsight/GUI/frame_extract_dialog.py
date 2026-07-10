"""
frame_extract_dialog.py -- extract still frames from footage for VP building
(MP2), launched from the VP Builder toolbar.

Sources: individual video files, every video in a folder, or a project's run
videos.  Extraction takes N evenly spaced frames per video (segment
midpoints, so fade-ins and duplicate end frames are skipped) into an output
folder; the VP Builder then offers to add the stills to its reference list.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

_GO_GREEN = ("QPushButton{background:#2a7a2a;color:white;"
             "font-weight:bold;padding:4px 26px;}"
             "QPushButton:disabled{background:#33333f;color:#777;}")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


class FrameExtractDialog(QDialog):
    """Pick videos -> frames per video -> output folder -> Extract.

    Extracted paths are collected in ``self.extracted`` for the caller (the
    VP Builder offers to add them as reference images).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract Frames for Visual Prompts")
        self.resize(560, 420)
        self.extracted: list[Path] = []

        lay = QVBoxLayout(self)
        hint = QLabel(
            "Pull still frames out of study footage to annotate as visual "
            "prompts. Frames are taken evenly across each video.")
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        add_files = QPushButton("Add Videos...")
        add_files.clicked.connect(self._add_videos)
        add_dir = QPushButton("Add Folder...")
        add_dir.clicked.connect(self._add_folder)
        add_proj = QPushButton("From Project...")
        add_proj.setToolTip("Add every run video from a MindSight project")
        add_proj.clicked.connect(self._add_project)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove_selected)
        for b in (add_files, add_dir, add_proj, rm):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._list = QListWidget()
        lay.addWidget(self._list, 1)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Frames per video:"))
        self._count = QSpinBox()
        self._count.setRange(1, 100)
        self._count.setValue(8)
        self._count.setToolTip("Evenly spaced across each video")
        opt_row.addWidget(self._count)
        opt_row.addStretch(1)
        lay.addLayout(opt_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save frames to:"))
        self._out = QLineEdit()
        from mindsight.constants import PROJECT_ROOT
        self._out.setText(str(PROJECT_ROOT / "VP_Frames"))
        out_row.addWidget(self._out, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_out)
        out_row.addWidget(browse)
        lay.addLayout(out_row)

        go_row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888;")
        go_row.addWidget(self._status)
        go_row.addStretch(1)
        extract = QPushButton("▶  Extract")
        extract.setStyleSheet(_GO_GREEN)
        extract.setMinimumHeight(32)
        extract.clicked.connect(self._extract)
        go_row.addWidget(extract)
        lay.addLayout(go_row)

    # -- source list ------------------------------------------------------

    def _paths(self) -> list[Path]:
        return [Path(self._list.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self._list.count())]

    def _append(self, paths):
        existing = set(self._paths())
        for p in paths or []:
            p = Path(p)
            if p.is_file() and p not in existing:
                item = QListWidgetItem(p.name)
                item.setToolTip(str(p))
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                self._list.addItem(item)
                existing.add(p)

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add videos", "",
            "Video (*.mp4 *.mov *.avi *.mkv);;All (*)")
        self._append(paths)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add a video folder")
        if folder:
            self._append(sorted(
                p for p in Path(folder).iterdir()
                if p.suffix.lower() in _VIDEO_EXTS and p.is_file()))

    def _add_project(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a MindSight project folder")
        if not folder:
            return
        try:
            from mindsight.project.project import Project
            sources = [Path(s.source) for s in Project.open(folder).runs()]
        except Exception as exc:  # noqa: BLE001 -- plain-English, not a crash
            QMessageBox.warning(self, "From Project",
                                f"Could not read that project: {exc}")
            return
        if not sources:
            QMessageBox.information(self, "From Project",
                                    "That project has no videos yet.")
            return
        self._append(sources)

    def _remove_selected(self):
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Save frames to")
        if path:
            self._out.setText(path)

    # -- extraction ---------------------------------------------------------

    def _extract(self):
        videos = self._paths()
        if not videos:
            QMessageBox.warning(self, "Extract frames",
                                "Add at least one video.")
            return
        out_root = self._out.text().strip()
        if not out_root:
            QMessageBox.warning(self, "Extract frames",
                                "Choose an output folder.")
            return
        from mindsight.io.video_edit import extract_frames
        progress = QProgressDialog("Extracting...", None, 0, len(videos),
                                   self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        failed = []
        self.extracted = []
        for i, video in enumerate(videos):
            progress.setValue(i)
            progress.setLabelText(
                f"Extracting from {video.name} ({i + 1}/{len(videos)})")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            try:
                self.extracted.extend(extract_frames(
                    video, Path(out_root) / video.stem,
                    count=self._count.value()))
            except (ValueError, OSError) as exc:
                failed.append(f"{video.name}: {exc}")
        progress.setValue(len(videos))
        if failed:
            QMessageBox.warning(
                self, "Some videos failed",
                f"Extracted {len(self.extracted)} frame(s).\nProblems:\n"
                + "\n".join(failed))
        if self.extracted:
            self.accept()
        elif not failed:
            self._status.setText("Nothing extracted.")
