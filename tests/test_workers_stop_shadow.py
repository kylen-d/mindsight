"""Regression guard: GUI worker threads must not shadow ``Thread._stop``.

The three background workers in :mod:`mindsight.GUI.workers`
(:class:`GazeWorker`, :class:`ProjectWorker`, :class:`VPInferenceWorker`) are
``threading.Thread`` subclasses.  A previous version stored the cancellation
flag as ``self._stop = threading.Event()``, which *shadows* CPython's private
``threading.Thread._stop()`` method.  On CPython < 3.14 (the shipped install
runs 3.12) ``Thread.is_alive()``/``Thread.join()`` reach the internal
``_wait_for_tstate_lock`` once a finished thread's tstate lock is released and
call ``self._stop()`` -- now an ``Event`` instead of a method -- raising
``TypeError: 'Event' object is not callable`` and aborting the process.  The
Gaze/Analyze tabs call ``worker.is_alive()`` on a just-finished worker before
launching the next run (gaze_tab ``_start``), so a second run crashed the app.

The fix renames the flag to ``_stop_event``.  These tests pin the invariant
(no worker rebinds ``_stop`` to an ``Event``) and exercise the crash path
(construct -> run a no-op body to completion -> ``is_alive()``/``join()`` must
not raise).  The invariant assertion fails before the fix on every CPython; the
behavioural guard additionally reproduces the abort on CPython < 3.14.
"""
import os
import queue
import threading
from argparse import Namespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from mindsight.GUI.workers import (  # noqa: E402
    GazeWorker,
    ProjectWorker,
    VPInferenceWorker,
)


def _make_gaze():
    return GazeWorker(Namespace(), queue.Queue(), queue.Queue())


def _make_project():
    return ProjectWorker("/nonexistent", Namespace(),
                         queue.Queue(), queue.Queue(), queue.Queue())


def _make_vp():
    return VPInferenceWorker("model.pt", [], queue.Queue(), queue.Queue())


worker_factories = pytest.mark.parametrize(
    "make_worker",
    [pytest.param(_make_gaze, id="GazeWorker"),
     pytest.param(_make_project, id="ProjectWorker"),
     pytest.param(_make_vp, id="VPInferenceWorker")],
)


@worker_factories
def test_worker_does_not_shadow_thread_stop(make_worker):
    """``_stop`` must never be rebound to an Event (the shadow that aborts)."""
    w = make_worker()
    shadow = getattr(w, "_stop", None)
    assert not isinstance(shadow, threading.Event), (
        "worker rebinds Thread._stop to an Event -- is_alive()/join() abort "
        "with TypeError on CPython < 3.14")
    # the cancellation flag lives under the safe name and stop() wires to it
    assert isinstance(w._stop_event, threading.Event)
    w.stop()
    assert w._stop_event.is_set()


@worker_factories
def test_is_alive_after_finish_does_not_raise(make_worker):
    """Repro of the crash path: a finished worker's ``is_alive()``/``join()``
    must not raise -- both reach ``Thread._stop`` on CPython < 3.14."""
    w = make_worker()
    w._main = lambda: None            # no-op body -> thread terminates at once
    w.start()
    w.join(timeout=5)                 # join() reaches _stop() on CPython < 3.14
    assert w.is_alive() is False      # is_alive() likewise; must not TypeError
    w.join(timeout=1)                 # idempotent second join, still no raise
