"""UP4 follow-up: mid-run quit guard + zero-frame camera hint.

Both found in a real user session (2026-07-10): closing the window mid-run
killed the daemon worker before finalization (a camera run lost its summary
CSV), and a camera that opens but delivers no frames produced empty outputs
with no explanation.
"""

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeWorker:
    """Alive until stop() is called, like a well-behaved run thread."""

    def __init__(self):
        self.stopped = False

    def is_alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


@pytest.mark.slow
def test_close_mid_run_prompts_and_finalizes(qapp, monkeypatch):
    from PyQt6.QtGui import QCloseEvent
    from PyQt6.QtWidgets import QMessageBox

    from mindsight.GUI.main_window import MainWindow

    win = MainWindow()
    worker = _FakeWorker()
    win._run_study_tab._one_off_worker = worker

    # Cancel keeps the window open and the run alive.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Cancel)
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()
    assert not worker.stopped

    # Yes stops the worker and waits for it before closing.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()
    assert worker.stopped


def test_zero_frame_quick_run_gets_plain_english_hint(qapp, tmp_path):
    import cv2

    from mindsight.GUI.run_study_tab import RunStudyTab

    out = tmp_path / "camera0"
    out.mkdir()
    # A 0-frame recording: writer opened and released without frames --
    # exactly what a placeholder camera produced in the real session.
    wr = cv2.VideoWriter(str(out / "cam_Video_Output.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    wr.release()
    (out / "cam_Events.csv").write_text("frame,t_seconds\n")

    tab = RunStudyTab()
    tab._last_one_off = ("cam", str(out))
    tab._register_one_off_outputs()
    assert "captured NO frames" in tab._log_box.toPlainText()

    # A real recording does NOT trigger the hint.
    out2 = tmp_path / "camera1"
    out2.mkdir()
    import numpy as np
    wr = cv2.VideoWriter(str(out2 / "cam2_Video_Output.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(5):
        wr.write(np.zeros((48, 64, 3), dtype=np.uint8))
    wr.release()
    (out2 / "cam2_Events.csv").write_text("frame,t_seconds\n1,0.033\n")
    tab2 = RunStudyTab()
    tab2._last_one_off = ("cam2", str(out2))
    tab2._register_one_off_outputs()
    assert "captured NO frames" not in tab2._log_box.toPlainText()
