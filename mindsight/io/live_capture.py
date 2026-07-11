"""
mindsight.io.live_capture -- raw session recording for live studies (UP5).

A ``LiveRecorder`` thread copies the camera feed straight to disk at full
capture rate -- NO inference in the loop, so model speed never drops session
frames (the record side of record-then-analyze).  Every frame gets a
wall-clock stamp; on stop the recording is finalized at the MEASURED average
fps (so the offline pass's ``t_seconds = frame / fps`` tracks real time to
jitter level) and an exact per-frame sidecar CSV is written for provenance.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path


class LiveRecorder(threading.Thread):
    """Record ``source`` (camera index or path) to ``dest`` until stopped.

    ``latest_frame()`` hands the GUI a preview frame without touching the
    capture cadence.  ``finalize()`` (called by ``run`` on stop) closes the
    writer, re-times the container to the measured fps via ffmpeg when
    available, and writes ``<dest stem>_capture_timestamps.csv``.
    """

    def __init__(self, source, dest, *, provisional_fps: float = 30.0):
        super().__init__(daemon=True)
        self._source = source
        self.dest = Path(dest)
        self._provisional_fps = float(provisional_fps)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._stamps: list[float] = []
        self.started_at: float | None = None   # epoch, for the sidecar header
        self.frames_captured = 0
        self.measured_fps: float | None = None
        self.error: str | None = None
        self.sidecar: Path | None = None

    # -- control ---------------------------------------------------------

    def stop(self):
        self._stop_event.set()

    def latest_frame(self):
        with self._lock:
            return self._latest

    @property
    def elapsed(self) -> float:
        if not self._stamps:
            return 0.0
        return self._stamps[-1] - self._stamps[0]

    # -- capture loop ------------------------------------------------------

    def run(self):
        import cv2
        src = self._source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error = f"cannot open camera/source: {self._source}"
            return
        writer = None
        try:
            self.started_at = time.time()
            t0 = time.monotonic()
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # A file-backed source ends naturally; a camera hiccup
                    # also lands here -- either way, stop recording.
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    self.dest.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(self.dest), cv2.VideoWriter_fourcc(*"mp4v"),
                        self._provisional_fps, (w, h))
                writer.write(frame)
                self._stamps.append(time.monotonic() - t0)
                self.frames_captured += 1
                with self._lock:
                    self._latest = frame
        finally:
            if writer is not None:
                writer.release()
            cap.release()
        if writer is None:
            self.error = (self.error
                          or f"camera delivered no frames: {self._source}")
            return
        self._finalize()

    # -- finalization -------------------------------------------------------

    def _measure_fps(self) -> float:
        if len(self._stamps) >= 2 and self.elapsed > 0:
            return (len(self._stamps) - 1) / self.elapsed
        return self._provisional_fps

    def _finalize(self):
        self.measured_fps = self._measure_fps()
        self._write_sidecar()
        # Re-time the container to the measured rate so t_seconds of any
        # offline analysis tracks wall clock. Stream copy only -- no
        # re-encode; on failure the provisional-rate file stands and the
        # measured rate lives in the sidecar header.
        if abs(self.measured_fps - self._provisional_fps) < 0.05:
            return
        from .video_edit import ffmpeg_exe
        exe = ffmpeg_exe()
        if exe is None:
            return
        import subprocess
        tmp = self.dest.with_name(self.dest.stem + "._retime" + self.dest.suffix)
        try:
            subprocess.run(
                [exe, "-y", "-itsscale",
                 f"{self._provisional_fps / self.measured_fps:.6f}",
                 "-i", str(self.dest), "-c", "copy", str(tmp)],
                check=True, capture_output=True)
            tmp.replace(self.dest)
        except Exception:  # noqa: BLE001 -- provisional-rate file is still valid
            if tmp.exists():
                tmp.unlink()

    def _write_sidecar(self):
        self.sidecar = self.dest.with_name(
            self.dest.stem + "_capture_timestamps.csv")
        with open(self.sidecar, "w", newline="") as fh:
            fh.write(f"# capture_start_epoch={self.started_at:.3f} "
                     f"measured_fps={self.measured_fps:.4f}\n")
            w = csv.writer(fh)
            w.writerow(["frame", "t_wall"])
            for i, t in enumerate(self._stamps):
                w.writerow([i, f"{t - self._stamps[0]:.4f}"])
