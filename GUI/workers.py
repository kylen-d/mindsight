"""
GUI/workers.py — Background worker threads for the MindSight GUI.

Workers
-------
GazeWorker          -- Namespace-driven: calls ``_build_from_args(ns)`` then
                       ``run()`` with full CLI feature parity (phenomena,
                       plugins, heatmaps, and all config flags).
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
# Background worker: Gaze Tracker (namespace-driven, full CLI parity)
# ══════════════════════════════════════════════════════════════════════════════

class GazeWorker(threading.Thread):
    """Loads models via ``_build_from_args()`` and runs the full MindSight pipeline.

    Accepts an ``argparse.Namespace`` with the same attribute names as the CLI
    produces, so every CLI feature (phenomena, plugins, heatmaps, etc.) works
    automatically without manual plumbing.

    Frames are pushed to *frame_q* by monkey-patching ``cv2.imshow`` so the
    GUI can display them without touching core pipeline code.
    """

    def __init__(self, ns: Namespace, frame_q: queue.Queue, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.ns       = ns
        self.frame_q  = frame_q
        self.log_q    = log_q
        self._stop    = threading.Event()

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return super().is_alive()

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
        import cv2
        import time as _time
        from MindSight import run, _build_from_args

        self._log("Initializing models...")

        # Build all models, config objects, and plugins from the namespace —
        # this is the same code path the CLI uses, giving automatic feature parity.
        (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
         output_cfg, active_plugins, phenomena_cfg) = _build_from_args(self.ns)

        self._log("Models loaded — starting pipeline...")

        # ── Redirect cv2 display into the GUI ────────────────────────────────
        _orig_imshow      = cv2.imshow
        _orig_waitkey     = cv2.waitKey
        _orig_destroy_all = cv2.destroyAllWindows
        _orig_destroy_win = cv2.destroyWindow

        _frame_q = self.frame_q
        _stop_ev = self._stop

        def _gui_imshow(_, frame):
            try:
                _frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        def _gui_waitkey(delay):
            if delay == 0:
                while not _stop_ev.is_set():
                    _time.sleep(0.05)
                return ord('q')
            return ord('q') if _stop_ev.is_set() else 1

        cv2.imshow            = _gui_imshow
        cv2.waitKey           = _gui_waitkey
        cv2.destroyAllWindows = lambda: None
        cv2.destroyWindow     = lambda *_: None

        try:
            source = self.ns.source
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass

            run(source, yolo, face_det, gaze_eng,
                gaze_cfg, det_cfg, tracker_cfg, output_cfg,
                plugin_instances=active_plugins,
                phenomena_cfg=phenomena_cfg)
        finally:
            cv2.imshow            = _orig_imshow
            cv2.waitKey           = _orig_waitkey
            cv2.destroyAllWindows = _orig_destroy_all
            cv2.destroyWindow     = _orig_destroy_win


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: Project batch processing
# ══════════════════════════════════════════════════════════════════════════════

class ProjectWorker(threading.Thread):
    """Batch-processes all videos in a MindSight project directory.

    Wraps ``project_runner.run_project()`` with progress reporting via
    *progress_q*.  Each progress event is a dict with keys:
      - ``type``: "start", "progress", "done", "error"
      - ``current``, ``total``, ``source_name`` (for progress events)
    """

    def __init__(self, project_dir: str, ns: Namespace,
                 progress_q: queue.Queue, log_q: queue.Queue,
                 frame_q: queue.Queue):
        super().__init__(daemon=True)
        self.project_dir = project_dir
        self.ns          = ns
        self.progress_q  = progress_q
        self.log_q       = log_q
        self.frame_q     = frame_q
        self._stop       = threading.Event()

    def stop(self):
        self._stop.set()

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
        import cv2
        import time as _time
        from MindSight import run, _build_from_args
        from project_runner import (
            validate_project, discover_sources, discover_vp_file,
            project_output_paths,
        )
        from pipeline_loader import load_pipeline
        from pipeline_config import OutputConfig

        project = validate_project(self.project_dir)
        sources = discover_sources(project)
        if not sources:
            self._log("No media files found in project.")
            return

        # Load project pipeline config if it exists
        pipeline_yaml = project / "Pipeline" / "pipeline.yaml"
        if pipeline_yaml.exists():
            load_pipeline(pipeline_yaml, self.ns)
            self._log(f"Loaded project pipeline: {pipeline_yaml}")

        # Apply project VP file if no CLI override
        vp = discover_vp_file(project)
        if vp and not getattr(self.ns, 'vp_file', None):
            self.ns.vp_file = vp
            self._log(f"Using project VP file: {vp}")

        # Build models once for the whole project
        (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg,
         output_cfg, active_plugins, phenomena_cfg) = _build_from_args(self.ns)

        self._log(f"Project: {project.name}  —  {len(sources)} source(s)")
        self.progress_q.put({"type": "start", "total": len(sources)})

        # Redirect cv2 display
        _orig_imshow = cv2.imshow
        _orig_waitkey = cv2.waitKey
        _orig_destroy_all = cv2.destroyAllWindows
        _orig_destroy_win = cv2.destroyWindow
        _frame_q = self.frame_q
        _stop_ev = self._stop

        def _gui_imshow(_, frame):
            try:
                _frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass

        def _gui_waitkey(delay):
            if delay == 0:
                while not _stop_ev.is_set():
                    _time.sleep(0.05)
                return ord('q')
            return ord('q') if _stop_ev.is_set() else 1

        cv2.imshow = _gui_imshow
        cv2.waitKey = _gui_waitkey
        cv2.destroyAllWindows = lambda: None
        cv2.destroyWindow = lambda *_: None

        try:
            for i, source in enumerate(sources):
                if self._stop.is_set():
                    break
                self._log(f"\n[{i+1}/{len(sources)}] Processing: {source.name}")
                self.progress_q.put({
                    "type": "progress",
                    "current": i + 1,
                    "total": len(sources),
                    "source_name": source.name,
                })

                paths = project_output_paths(project, source)
                run_output = OutputConfig(
                    save=paths['save'],
                    log_path=paths['log'],
                    summary_path=paths['summary'],
                    heatmap_path=paths['heatmap'],
                )

                try:
                    run(str(source), yolo, face_det, gaze_eng,
                        gaze_cfg, det_cfg, tracker_cfg, run_output,
                        plugin_instances=active_plugins,
                        phenomena_cfg=phenomena_cfg)
                except Exception as exc:
                    self._log(f"Error processing {source.name}: {exc}")

            self.progress_q.put({"type": "done"})
            self._log(f"\nProject complete. Outputs in: {project / 'Outputs'}")
        finally:
            cv2.imshow = _orig_imshow
            cv2.waitKey = _orig_waitkey
            cv2.destroyAllWindows = _orig_destroy_all
            cv2.destroyWindow = _orig_destroy_win


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
        self._stop       = threading.Event()

    def stop(self):
        self._stop.set()

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

        self._log(f"Loading YOLOE: {self.model_path}")
        model = YOLOE(self.model_path)

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
            if self._stop.is_set():
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
