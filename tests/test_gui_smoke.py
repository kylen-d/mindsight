"""Offscreen-Qt smoke for the MindSight GUI (SP1.4 regression net).

Written BEFORE the GUI is rewired off the cv2-monkeypatch worker, so it pins the
behavior that must survive: the main window launches headless, a pipeline YAML
round-trips through the Gaze tab widgets, the GazeWorker streams annotated frames
into its queue and cancels cleanly (finalizing the summary CSV) while feeding the
live-dashboard queue, and -- once the pre-existing project-mode import bug is
fixed in step 4 -- the ProjectWorker does the same per video.

Heavy (loads real models + runs the clip), so slow-marked and self-skipping when
Qt, the sample video, or the model weights are unavailable.  Runs entirely under
``QT_QPA_PLATFORM=offscreen`` -- no display, no cv2 window.
"""

import os
import queue
import time
from argparse import Namespace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO = REPO_ROOT / "test_data" / "trimmed.mp4"
MGAZE_ONNX = REPO_ROOT / "Weights" / "MGaze" / "resnet50_gaze.onnx"
YOLO_WEIGHT = REPO_ROOT / "Weights" / "YOLO" / "yolov8n.pt"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not VIDEO.exists(), reason="test_data/trimmed.mp4 missing"),
    pytest.mark.skipif(not MGAZE_ONNX.exists(), reason="MGaze onnx weight missing"),
    pytest.mark.skipif(not YOLO_WEIGHT.exists(), reason="YOLO weight missing"),
]


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app  # never quit -- shared across the module


def _collect_frames(frame_q, log_q, *, want, deadline_s):
    """Pull ndarray frames off frame_q until ``want`` collected or deadline.
    Returns (frames, logs) so a failure can surface the worker's [ERROR] logs."""
    import numpy as np
    frames = []
    logs = []
    end = time.monotonic() + deadline_s
    while len(frames) < want and time.monotonic() < end:
        try:
            while True:
                logs.append(log_q.get_nowait())
        except queue.Empty:
            pass
        try:
            f = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if f is None:            # end-of-run sentinel
            break
        if isinstance(f, np.ndarray):
            frames.append(f)
    try:
        while True:
            logs.append(log_q.get_nowait())
    except queue.Empty:
        pass
    return frames, logs


def test_main_window_launches_and_loads_config(qapp):
    from mindsight.config_compat import load_pipeline
    from mindsight.GUI.main_window import MainWindow

    win = MainWindow()
    try:
        assert win._gaze_tab is not None
        assert win._vp_tab is not None
        assert win._project_tab is not None

        ns = load_pipeline(REPO_ROOT / "test_pipeline.yaml", Namespace())
        win._gaze_tab.apply_namespace(ns)
        ns2 = win._gaze_tab._build_namespace()
        # values from test_pipeline.yaml round-trip through the widgets
        assert ns2.model == "yoloe-26s-seg.pt"
        assert ns2.ray_length == 1.5
        assert ns2.gaze_tips is True
    finally:
        win.close()


def test_gaze_worker_streams_and_cancels(qapp, tmp_path):
    from mindsight.GUI.main_window import MainWindow
    from mindsight.GUI.workers import GazeWorker

    win = MainWindow()
    try:
        tab = win._gaze_tab
        tab._detection._src.setText(str(VIDEO))
        tab._gaze_backend._gaze_model.setText(str(MGAZE_ONNX))
        ns = tab._build_namespace()
        ns.log = str(tmp_path / "events.csv")
        ns.summary = str(tmp_path / "summary.csv")

        frame_q: queue.Queue = queue.Queue(maxsize=4)
        log_q: queue.Queue = queue.Queue()
        dash_q: queue.Queue = queue.Queue(maxsize=30)
        worker = GazeWorker(ns, frame_q, log_q, dashboard_q=dash_q)
        worker.start()

        frames, logs = _collect_frames(frame_q, log_q, want=10, deadline_s=180)
        assert len(frames) >= 10, (
            f"only {len(frames)} frames; worker log:\n" + "\n".join(map(str, logs)))

        worker.stop()
        worker.join(timeout=120)
        assert not worker.is_alive(), "worker did not stop after cancel"

        # end-of-run None sentinel eventually arrives
        got_sentinel = False
        end = time.monotonic() + 10
        while time.monotonic() < end:
            try:
                if frame_q.get(timeout=1.0) is None:
                    got_sentinel = True
                    break
            except queue.Empty:
                break
        assert got_sentinel, "no None end sentinel on frame_q"

        # cancel finalized outputs through the normal post-run path
        summary = tmp_path / "summary.csv"
        assert summary.exists() and summary.stat().st_size > 0

        # the live-dashboard bridge was fed
        saw_dash = False
        try:
            while True:
                snap = dash_q.get_nowait()
                if isinstance(snap, dict) and "fps" in snap:
                    saw_dash = True
        except queue.Empty:
            pass
        assert saw_dash, "no dashboard snapshot with an 'fps' key"
    finally:
        if win._gaze_tab._worker and win._gaze_tab._worker.is_alive():
            win._gaze_tab._worker.stop()
        win.close()


def test_project_worker_runs_and_cancels(qapp, tmp_path):
    import shutil

    from mindsight.GUI.main_window import MainWindow
    from mindsight.GUI.workers import ProjectWorker

    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    shutil.copy(VIDEO, proj / "Inputs" / "Videos" / "trimmed.mp4")

    win = MainWindow()
    try:
        tab = win._gaze_tab
        tab._gaze_backend._gaze_model.setText(str(MGAZE_ONNX))
        ns = tab._build_namespace()

        progress_q: queue.Queue = queue.Queue()
        log_q: queue.Queue = queue.Queue()
        frame_q: queue.Queue = queue.Queue(maxsize=2)
        worker = ProjectWorker(str(proj), ns, progress_q, log_q, frame_q,
                               project_cfg=None)
        worker.start()

        first = progress_q.get(timeout=180)
        assert isinstance(first, dict) and first.get("type") == "start"

        # let a video get going, then cancel
        _collect_frames(frame_q, log_q, want=5, deadline_s=180)
        worker.stop()
        worker.join(timeout=180)
        assert not worker.is_alive()

        # a done event + None sentinel arrive; per-video summary written
        events = []
        try:
            while True:
                events.append(progress_q.get(timeout=1.0))
        except queue.Empty:
            pass
        assert None in events, "no None sentinel on progress_q"
        assert any(isinstance(e, dict) and e.get("type") == "done"
                   for e in events)
        summ = proj / "Outputs" / "CSV Files" / "trimmed_summary.csv"
        assert summ.exists()
    finally:
        if win._project_tab._worker and win._project_tab._worker.is_alive():
            win._project_tab._worker.stop()
        win.close()
