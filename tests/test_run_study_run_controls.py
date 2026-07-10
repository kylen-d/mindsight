"""Analyze Footage run-control wiring (SP3.1 Batch G fix-forward, G-FIX-1/2).

Fast, offscreen, model-free: a fake ProjectWorker (monkeypatched into
``mindsight.GUI.workers``) streams frames + progress events through the REAL
tab queues so the poll/paint/stop wiring is exercised without loading models.

Pins:
- G-FIX-1: frames pulled off frame_q are PAINTED into the preview label during
  a run (the paint must survive the queue.Empty drain exit).
- G-FIX-2: Stop visibly transitions (Stop disabled + "Cancelling"/"Cancelled"
  log lines, Run re-enabled at the sentinel) and a following Start launches a
  FRESH worker.
"""

import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakeProjectWorker(threading.Thread):
    """Streams a start event + frames, then waits for stop() like the real one."""

    instances: list = []

    def __init__(self, project_dir, ns, progress_q, log_q, frame_q, *,
                 project_cfg=None, dashboard_q=None):
        super().__init__(daemon=True)
        self.progress_q = progress_q
        self.log_q = log_q
        self.frame_q = frame_q
        # ``_stop_event`` (never ``_stop``): a bare ``_stop`` attribute would
        # shadow ``threading.Thread._stop`` and abort on ``is_alive``/``join``
        # under CPython < 3.14 -- mirror the real worker's fixed naming.
        self._stop_event = threading.Event()
        FakeProjectWorker.instances.append(self)

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.progress_q.put({"type": "start", "total": 1})
        self.progress_q.put({"type": "progress", "current": 1, "total": 1,
                             "source_name": "a.mp4"})
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        # stream frames until stopped (bounded so a test bug cannot hang)
        for _ in range(600):
            try:
                self.frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass
            if self._stop_event.wait(timeout=0.01):
                break
        self.log_q.put("fake worker: finishing")
        self.progress_q.put({"type": "done"})
        self.progress_q.put(None)
        self.frame_q.put(None)


def _make_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "Inputs" / "Videos").mkdir(parents=True)
    (proj / "Inputs" / "Videos" / "a.mp4").write_bytes(b"\x00" * 32)
    return proj


@pytest.fixture()
def tab(qapp, tmp_path, monkeypatch):
    import mindsight.GUI.workers as workers_mod
    from mindsight.GUI.run_study_tab import RunStudyTab
    monkeypatch.setattr(workers_mod, "ProjectWorker", FakeProjectWorker)
    FakeProjectWorker.instances = []
    t = RunStudyTab()
    t._open_project(str(_make_project(tmp_path)))
    yield t
    if t._worker and t._worker.is_alive():
        t._worker.stop()
        t._worker.join(timeout=5)


def _poll_until(tab, predicate, deadline_s=10.0):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        tab._poll()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_preview_paints_streamed_frames(tab):
    """G-FIX-1: the preview label receives a pixmap while frames stream."""
    assert tab._preview.pixmap() is None or tab._preview.pixmap().isNull()
    tab._start()
    worker = tab._worker
    assert worker is not None and worker.is_alive()
    painted = _poll_until(
        tab, lambda: tab._preview.pixmap() is not None
        and not tab._preview.pixmap().isNull())
    assert painted, "preview never received a painted frame during the run"
    tab._stop()
    _poll_until(tab, lambda: tab._worker is None)


def test_stop_transitions_visibly_and_start_relaunches(tab):
    """G-FIX-2: Stop -> visible cancelling/cancelled states; Start -> fresh worker."""
    tab._start()
    first_worker = tab._worker
    assert first_worker is not None
    _poll_until(tab, lambda: "Starting run..." in tab._log_box.toPlainText())

    tab._stop()
    # immediate visible feedback
    assert not tab._stop_btn.isEnabled()
    assert "Cancelling" in tab._log_box.toPlainText()

    # the sentinel arrives -> terminal transition
    finished = _poll_until(
        tab, lambda: tab._worker is None and tab._run_btn.isEnabled())
    assert finished, "run never reached the cancelled terminal state"
    assert "Cancelled." in tab._log_box.toPlainText()
    first_worker.join(timeout=5)
    assert not first_worker.is_alive()

    # Start after stop launches a FRESH worker
    tab._start()
    second_worker = tab._worker
    assert second_worker is not None and second_worker is not first_worker
    assert second_worker.is_alive()
    assert len(FakeProjectWorker.instances) == 2
    tab._stop()
    _poll_until(tab, lambda: tab._worker is None)


def test_start_while_finishing_logs_feedback(tab):
    """Start during an alive worker logs feedback instead of a silent no-op."""
    tab._start()
    alive_worker = tab._worker
    tab._start()   # second click while running
    assert "still finishing" in tab._log_box.toPlainText()
    assert tab._worker is alive_worker      # no replacement mid-run
    tab._stop()
    _poll_until(tab, lambda: tab._worker is None)


# ── B1 F1/F2: one-off "Run now" wiring + shared any-worker guard ──────────────


class FakeGazeWorker(threading.Thread):
    """Captures the ns; does NOT run the pipeline (start() spawns nothing)."""

    instances: list = []

    def __init__(self, ns, frame_q, log_q, dashboard_q=None):
        super().__init__(daemon=True)
        self.ns = ns
        self._alive = False
        FakeGazeWorker.instances.append(self)

    def start(self):        # override: never launch a real pipeline thread
        self._alive = True

    def is_alive(self):
        return self._alive

    def stop(self):
        self._alive = False


class _AliveStub:
    """A stand-in worker that reports itself alive (for the guard tests)."""

    def is_alive(self):
        return True

    def stop(self):
        pass


def _one_off_dlg(project, *, output_dir=None):
    return SimpleNamespace(
        video=str(project / "Inputs" / "Videos" / "a.mp4"),
        meta={}, output_dir=output_dir, move=False)


def test_run_single_run_defaults_output_to_project(tab, tmp_path, monkeypatch):
    """B1 F1: an omitted output_dir lands under the open project's Outputs root,
    and the absolute output dir is logged."""
    import mindsight.GUI.workers as workers_mod
    monkeypatch.setattr(workers_mod, "GazeWorker", FakeGazeWorker)
    FakeGazeWorker.instances = []
    proj = Path(tab._project_path)
    tab._run_single_run(_one_off_dlg(proj, output_dir=None))
    assert len(FakeGazeWorker.instances) == 1
    ns = FakeGazeWorker.instances[0].ns
    assert ns.log == str(proj / "Outputs" / "a_Events.csv")
    assert ns.summary == str(proj / "Outputs" / "a_summary.csv")
    log_text = tab._log_box.toPlainText()
    assert f"One-off outputs -> {proj / 'Outputs'}" in log_text
    tab._poll_timer.stop()


def test_run_single_run_refuses_while_worker_alive(tab, monkeypatch):
    """B1 F2: the one-off path refuses to start over a live batch worker."""
    import mindsight.GUI.workers as workers_mod
    monkeypatch.setattr(workers_mod, "GazeWorker", FakeGazeWorker)
    FakeGazeWorker.instances = []
    tab._worker = _AliveStub()
    proj = Path(tab._project_path)
    tab._run_single_run(_one_off_dlg(proj))
    assert "still finishing" in tab._log_box.toPlainText()
    assert FakeGazeWorker.instances == []       # no second worker started
    assert tab._one_off_worker is None
    tab._worker = None


def test_start_refuses_while_one_off_worker_alive(tab):
    """B1 F2: the batch path refuses to start over a live one-off worker."""
    FakeProjectWorker.instances = []
    tab._one_off_worker = _AliveStub()
    tab._start()
    assert "still finishing" in tab._log_box.toPlainText()
    assert len(FakeProjectWorker.instances) == 0
    tab._one_off_worker = None
