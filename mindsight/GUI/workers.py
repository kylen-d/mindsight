"""
GUI/workers.py — Background worker threads for the MindSight GUI.

Workers
-------
GazeWorker          -- Namespace-driven: builds models via
                       ``mindsight.pipeline.build_from_namespace(ns)`` then consumes
                       ``Pipeline.run`` (FrameResult stream) with full CLI
                       feature parity (phenomena, plugins, heatmaps, flags).
VPInferenceWorker   -- Runs YOLOE inference on a batch of images using a
                       Visual Prompt file or text-class prompts.
ProjectWorker       -- Batch-processes all videos in a MindSight project
                       directory, reporting progress via queue.

All workers use ``threading.Event()`` for stop signalling, ``queue.Queue()``
for communication, run as daemon threads, and push a ``None`` sentinel when
they are finished.
"""
from __future__ import annotations

import queue
import threading
from argparse import Namespace
from pathlib import Path

from .widgets import load_vp_file, vp_to_yoloe_args


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: Weight downloads (manifest-driven, shared by tabs)
# ══════════════════════════════════════════════════════════════════════════════

class WeightsDownloadWorker(threading.Thread):
    """Download a list of manifest entries off the GUI thread.

    Drives :func:`mindsight.weights.download` for each *entry* and reports via
    *out_q* so the caller (Models tab, Analyze-Footage preflight) never blocks
    the UI.  Messages are ``(kind, entry, payload)`` tuples:

    * ``("log", entry, str)``   -- a progress line from the downloader.
    * ``("done", entry, Path)`` -- that entry finished and verified.
    * ``("error", entry, str)`` -- a plain-English failure (G-OFFLINE UX).
    * ``("finished", None, None)`` -- the whole batch is done (always last).

    *dest_for* optionally maps an entry to a destination path (tests point it at
    a tmp weights dir); when ``None`` the manifest's resolved path is used.
    """

    def __init__(self, entries, out_q: queue.Queue, *, dest_for=None):
        super().__init__(daemon=True)
        self._entries = list(entries)
        self._q = out_q
        self._dest_for = dest_for

    def run(self):
        from mindsight import weights
        for entry in self._entries:
            try:
                dest = self._dest_for(entry) if self._dest_for else None
                path = weights.download(
                    entry, dest=dest,
                    progress=lambda msg, e=entry: self._q.put(("log", e, msg)))
                self._q.put(("done", entry, path))
            except weights.WeightsError as exc:
                self._q.put(("error", entry, str(exc)))
        self._q.put(("finished", None, None))


class WeightsVerifyWorker(threading.Thread):
    """Checksum-verify manifest weight rows off the GUI thread (Models tab).

    Same worker pattern as :class:`WeightsDownloadWorker` (v1.1 W0.7 -- the
    tab previously spun raw ``threading.Thread``s for this).  Messages match
    the Models tab's queue pump:

    * ``("vstate", row, state)`` -- one row verified.
    * ``("vdone", None, None)``  -- the whole batch is done (always last).
    """

    def __init__(self, rows, row_info, out_q: queue.Queue):
        super().__init__(daemon=True)
        self._rows = list(rows)
        self._row_info = row_info
        self._q = out_q

    def run(self):
        from mindsight import weights
        for row in self._rows:
            info = self._row_info[row]
            state = weights.verify(info["dest"], info["entry"])
            self._q.put(("vstate", row, state))
        self._q.put(("vdone", None, None))

# ══════════════════════════════════════════════════════════════════════════════
# Background worker: Gaze Tracker (namespace-driven, full CLI parity)
# ══════════════════════════════════════════════════════════════════════════════

class GazeWorker(threading.Thread):
    """Builds models via ``build_from_namespace`` and drives ``Pipeline.run``.

    Accepts an ``argparse.Namespace`` with the same attribute names as the CLI
    produces, so every CLI feature (phenomena, plugins, heatmaps, etc.) works
    automatically without manual plumbing.

    Each :class:`~mindsight.pipeline.FrameResult`'s ``annotated`` frame is pushed to
    *frame_q* for the GUI to display; the worker's stop Event is translated into
    a per-frame :class:`~mindsight.pipeline.CancelToken` (no cv2 monkeypatching).
    """

    def __init__(self, ns: Namespace, frame_q: queue.Queue, log_q: queue.Queue,
                 dashboard_q: queue.Queue | None = None):
        super().__init__(daemon=True)
        self.ns           = ns
        self.frame_q      = frame_q
        self.log_q        = log_q
        self.dashboard_q  = dashboard_q
        self._stop_event  = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _log(self, msg):
        self.log_q.put(str(msg))

    def run(self):
        try:
            self._main()
        except Exception as exc:
            import traceback
            self._log(f"[ERROR] {exc}\n{traceback.format_exc()}")
        finally:
            self.frame_q.put(None)  # sentinel: worker is done

    def _main(self):
        from mindsight.constants import IMAGE_EXTS
        from mindsight.pipeline import (
            CancelToken,
            Pipeline,
            RunOptions,
            build_from_namespace,
        )

        self._log("Initializing models...")

        # Build all models, config objects, and plugins from the namespace —
        # same model wiring the CLI uses, giving automatic feature parity.
        (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
         output_cfg, active_plugins, phenomena_cfg,
         detection_plugins, depth_cfg, depth_backend,
         gazelle_provider, ray_cfg) = build_from_namespace(self.ns)

        self._log("Models loaded — starting pipeline...")

        # Inject the live dashboard bridge as a phenomena plugin so it is fed
        # INSIDE the loop (honoring the phenomena-skip / --fast throttle).
        _fast = getattr(self.ns, 'fast', False)
        if self.dashboard_q is not None:
            from .live_dashboard_bridge import LiveDashboardBridge
            bridge = LiveDashboardBridge(
                self.dashboard_q,
                throttle=6 if _fast else 0,
            )
            if active_plugins is None:
                active_plugins = []
            active_plugins.append(bridge)

        source = self.ns.source
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass

        pipeline = Pipeline(
            yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
            gaze_cfg=gaze_cfg, det_cfg=det_cfg, tracker_cfg=tracker_cfg,
            output_cfg=output_cfg, plugin_instances=active_plugins,
            detection_plugins=detection_plugins, phenomena_cfg=phenomena_cfg,
            depth_cfg=depth_cfg, depth_backend=depth_backend,
            gazelle_provider=gazelle_provider, ray_cfg=ray_cfg,
        )
        options = RunOptions(
            fast_mode=_fast,
            skip_phenomena=getattr(self.ns, 'skip_phenomena', 0),
            lite_overlay=getattr(self.ns, 'lite_overlay', False),
            no_dashboard=getattr(self.ns, 'no_dashboard', False),
            profile=getattr(self.ns, 'profile', False),
        )

        # Pump annotated frames to the GUI; translate the stop Event into a
        # per-frame CancelToken.  Never break -- iterate to StopIteration so the
        # pipeline's finally + post-run summaries finalize all outputs on cancel.
        from mindsight.outputs import provenance
        started = provenance.utcnow_iso()
        cancel = CancelToken()
        for result in pipeline.run(source, options=options, cancel=cancel):
            try:
                self.frame_q.put_nowait(result.annotated.copy())
            except queue.Full:
                pass
            if self._stop_event.is_set():
                cancel.cancel()

        # Per-run provenance manifest (D8): only when a file output is
        # configured; located next to the summary/log/saved-video (Q4).
        outputs = provenance.resolve_single_source_outputs(self.ns, source)
        manifest_path = provenance.manifest_path_for(outputs)
        if manifest_path:
            from mindsight.config import PipelineConfig
            provenance.write_run_manifest(
                manifest_path, ns=self.ns,
                config=PipelineConfig.from_namespace(self.ns),
                source=source, output_paths=outputs,
                started=started, finished=provenance.utcnow_iso(),
                status="completed")

        # Image sources yield a single frame; legacy GUI behavior kept the image
        # displayed until Stop -- preserve that (block until stopped).
        is_image = (isinstance(source, str)
                    and Path(source).suffix.lower() in IMAGE_EXTS)
        if is_image:
            self._stop_event.wait()


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: Project batch processing
# ══════════════════════════════════════════════════════════════════════════════

class ProjectWorker(threading.Thread):
    """Batch-processes all videos in a MindSight project directory.

    Thin consumer of ``project.runner.iter_project_runs`` (the single project-run
    implementation, SP3.1 D1): it owns NO discovery / ledger / manifest /
    global-CSV logic, only the translation of the ``ProjectEvent`` stream onto
    the GUI queues.  Each progress event is a dict with keys:
      - ``type``: "start", "progress", "done", "error"
      - ``current``, ``total``, ``source_name`` (for progress events)

    The worker's stop Event trips the iterator's batch ``CancelToken`` so the
    current video finalizes cleanly through the pipeline's post-run paths (T8).
    """

    def __init__(self, project_dir: str, ns: Namespace,
                 progress_q: queue.Queue, log_q: queue.Queue,
                 frame_q: queue.Queue, *,
                 project_cfg=None, dashboard_q: queue.Queue | None = None):
        super().__init__(daemon=True)
        self.project_dir = project_dir
        self.ns          = ns
        self.progress_q  = progress_q
        self.log_q       = log_q
        self.frame_q     = frame_q
        self.project_cfg = project_cfg
        self.dashboard_q = dashboard_q
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _log(self, msg):
        self.log_q.put(str(msg))

    def run(self):
        try:
            self._main()
        except Exception as exc:
            import traceback
            self._log(f"[ERROR] {exc}\n{traceback.format_exc()}")
            self.progress_q.put({"type": "error", "message": str(exc)})
        finally:
            self.progress_q.put(None)  # sentinel
            self.frame_q.put(None)

    def _main(self):
        from mindsight.pipeline import CancelToken
        from mindsight.project.events import (
            BatchDone,
            BatchStarted,
            VideoArchived,
            VideoDone,
            VideoError,
            VideoFrame,
            VideoSkipped,
            VideoStarted,
        )
        from mindsight.project.project import Project

        # Batch-level cancel: the stop Event trips it so the current video
        # finalizes through the pipeline's post-run paths (T8/T9).
        cancel = CancelToken()
        resume = not getattr(self.ns, "no_resume", False)

        # The GUI touches only the Project facade; it wraps iter_project_runs and
        # keeps the in-memory project_cfg (possibly-unsaved edits) authoritative.
        project = Project.open(self.project_dir)

        # GUI-only live dashboard bridge, fed the same way GazeWorker feeds it:
        # a phenomena plugin appended to EACH video's plugin set inside the run
        # loop. Passed as `gui_plugins` so the CLI path (which never sets a
        # dashboard_q) stays byte-identical -- nothing is injected there.
        gui_plugins = None
        if self.dashboard_q is not None:
            from .live_dashboard_bridge import LiveDashboardBridge
            _fast = getattr(self.ns, 'fast', False)
            bridge = LiveDashboardBridge(
                self.dashboard_q, throttle=6 if _fast else 0)
            gui_plugins = [bridge]

        # Output artifact toggles (Q7/A3): the RunSettings store rides its
        # save/heatmap/charts booleans on the namespace; map them so an unticked
        # toggle drops that per-video path.  Default (all produce) is byte-neutral
        # with today's project batch.
        from .run_settings import want_artifact
        output_toggles = {k: want_artifact(self.ns, k)
                          for k in ("save", "heatmap", "charts")}

        total = 0
        pos = 0  # 1-based source position, tracked for the "[i/N]" log prefix
        for event in project.run(self.ns, resume=resume, cancel=cancel,
                                 project_cfg=self.project_cfg,
                                 gui_plugins=gui_plugins,
                                 output_toggles=output_toggles):
            # Propagate a stop request to the iterator on every event.
            if self._stop_event.is_set():
                cancel.cancel()

            if isinstance(event, BatchStarted):
                total = event.total
                self.progress_q.put({"type": "start", "total": event.total})
                self._log(f"Output root: {event.out_root}")

            elif isinstance(event, VideoStarted):
                pos = event.index
                self.progress_q.put({
                    "type": "progress",
                    "current": event.index,
                    "total": event.total,
                    "source_name": event.run_id,
                })
                self._log(f"\n[{event.index}/{event.total}] "
                          f"Processing: {event.run_id}")

            elif isinstance(event, VideoFrame):
                try:
                    self.frame_q.put_nowait(event.result.annotated.copy())
                except queue.Full:
                    pass

            elif isinstance(event, VideoSkipped):
                pos += 1
                self.progress_q.put({"type": "skipped", "run_id": event.run_id,
                                     "reason": event.reason})
                self._log(f"[{pos}/{total}] Skipping "
                          f"{event.run_id} ({event.reason})")

            elif isinstance(event, VideoArchived):
                self.progress_q.put({"type": "archived", "run_id": event.run_id})

            elif isinstance(event, VideoDone):
                self.progress_q.put({"type": "video_done",
                                     "run_id": event.run_id})

            elif isinstance(event, VideoError):
                self.progress_q.put({"type": "video_error", "run_id": event.run_id,
                                     "error": str(event.error)})
                self._log(f"Error processing {event.run_id}: {event.error}")

            elif isinstance(event, BatchDone):
                self.progress_q.put({"type": "done"})
                self._log(f"\nProject complete. Outputs in: {event.out_root}")


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: VP Inference
# ══════════════════════════════════════════════════════════════════════════════

class VPInferenceWorker(threading.Thread):
    """
    Run YOLOE inference on a list of images using a .vp.json visual prompt
    file.  Supports both VP mode (refer_image) and traditional text-class mode
    (set_classes) depending on whether a *vp_file* is supplied.

    result_q items: {"path": Path, "dets": [...], "frame": ndarray|None}
    Sentinel: ``None`` pushed when done.
    """

    def __init__(self, model_path: str, image_paths: list,
                 result_q: queue.Queue, log_q: queue.Queue,
                 conf: float = 0.30,
                 vp_file: str | None = None,
                 text_classes: list | None = None):
        super().__init__(daemon=True)
        self.model_path  = model_path
        self.image_paths = image_paths
        self.result_q    = result_q
        self.log_q       = log_q
        self.conf        = conf
        self.vp_file     = vp_file          # None -> text-class mode
        self.text_classes = text_classes     # used when vp_file is None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _log(self, msg):
        self.log_q.put(str(msg))

    def run(self):
        try:
            self._main()
        except Exception as exc:
            import traceback
            self._log(f"[ERROR] {exc}\n{traceback.format_exc()}")
        finally:
            self.result_q.put(None)

    def _main(self):
        import cv2
        from ultralytics import YOLOE

        from mindsight.weights import resolve_weight
        resolved = str(resolve_weight("YOLO", self.model_path))
        self._log(f"Loading YOLOE: {resolved}")
        model = YOLOE(resolved)

        # ── Determine mode ───────────────────────────────────────────────────
        if self.vp_file:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            vp_data = load_vp_file(self.vp_file)
            refer_image, visual_prompts, class_names = vp_to_yoloe_args(vp_data)
            self._log(f"VP classes: {class_names}  (ref: {Path(refer_image).name})")
            classes_set = False

            def _predict(frame):
                nonlocal classes_set
                if not classes_set:
                    r = model.predict(frame,
                                      refer_image=refer_image,
                                      visual_prompts=visual_prompts,
                                      predictor=YOLOEVPSegPredictor,
                                      conf=self.conf, verbose=False)
                    classes_set = True
                    return r
                return model.predict(frame, conf=self.conf, verbose=False)

        else:
            # Text-class mode (traditional YOLOE)
            prompts = self.text_classes or []
            model.set_classes(prompts)
            class_names = prompts
            self._log(f"Text classes: {prompts}")

            def _predict(frame):
                return model.predict(frame, conf=self.conf, verbose=False)

        # ── Inference loop ───────────────────────────────────────────────────
        for i, img_path in enumerate(self.image_paths):
            if self._stop_event.is_set():
                break
            frame = cv2.imread(str(img_path))
            if frame is None:
                self._log(f"SKIP {img_path.name}")
                self.result_q.put({"path": img_path, "dets": [], "frame": None})
                continue

            try:
                results = _predict(frame)
            except Exception as e:
                self._log(f"[WARN] {img_path.name}: {e}")
                self.result_q.put({"path": img_path, "dets": [], "frame": frame.copy()})
                continue

            dets = []
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    c      = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if c < self.conf:
                        continue
                    cls_name = (class_names[cls_id]
                                if cls_id < len(class_names) else str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({"cls_name": cls_name, "cls_id": cls_id, "conf": c,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "selected": True})

            self._log(f"[{i+1}/{len(self.image_paths)}] {img_path.name} -> {len(dets)} det(s)")
            self.result_q.put({"path": img_path, "dets": dets, "frame": frame.copy()})
