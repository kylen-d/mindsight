"""
outputs/provenance.py -- Per-run provenance manifest.

Collects the run environment, loaded-weight sha256s, source-file identity, and a
composite run-identity hash, then writes a per-run ``manifest.json`` ATOMICALLY.

Called only from the ORCHESTRATION layers -- ``cli.main`` single-source path,
``project.runner.run_project`` per-video loop, and the GUI ``GazeWorker`` /
``ProjectWorker`` workers -- after a run returns (or on per-video error). The
Pipeline generator stays provenance-free (a consumer may abandon it mid-run;
the manifest belongs to the layer that owns run completion -- SP2.1 T8/D8).

Design bindings:
- D6  run_identity: sha256 over the config model_dump (output/project sections
      removed) + model-wiring inputs (incl. device) + plugin flag values +
      sorted loaded-weight sha256s + ``__version__``.
- D7  weights: an explicit dest table resolved via ``mindsight.weights`` with a
      (path, size, mtime)-keyed sha256 cache so a batch hashes each weight once.
- D8  a manifest is written only when at least one file output is configured;
      location per the Q4 ruling (project -> Outputs/CSV Files/{stem}_manifest.json;
      single-source -> next to the summary CSV, else log CSV, else saved video).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from mindsight import __version__
from mindsight.constants import OUTPUTS_ROOT as _OUTPUTS_ROOT
from mindsight.weights import resolve_mgaze_family, resolve_weight, sha256_file

MANIFEST_SCHEMA_VERSION = 1

_CONFIG_NOTE = (
    "config.detection.class_ids / blacklist are null/empty here: the CLI "
    "resolves class names against the loaded model at build time, not in the "
    "namespace. The raw --classes / --blacklist inputs are captured in the "
    "run_identity wiring section instead."
)


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with offset."""
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

# Dependency modules whose __version__ is recorded (guarded by try/except).
_DEP_MODULES = ["torch", "ultralytics", "onnxruntime", "cv2", "numpy",
                "mediapipe"]


def collect_environment() -> dict:
    """Record mindsight/python/platform + each dependency's version.

    A dependency that is not importable is recorded as ``"absent"``; one that
    is importable but exposes no ``__version__`` is recorded as ``"unknown"``.
    """
    deps: dict = {}
    for mod in _DEP_MODULES:
        try:
            m = __import__(mod)
        except ImportError:
            deps[mod] = "absent"
            continue
        deps[mod] = getattr(m, "__version__", "unknown")
    return {
        "mindsight": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": deps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Weight enumeration + sha256 cache (D7)
# ══════════════════════════════════════════════════════════════════════════════

# (backend, ns_dest) -- backend None means the raw path is used verbatim.
_WEIGHT_TABLE = [
    ("YOLO", "model"),
    ("YOLO", "vp_model"),        # only when a vp_file is configured
    (None, "vp_file"),
    ("MGaze", "mgaze_model"),
    ("Gazelle", "rf_gazelle_model"),
    ("Gazelle", "gazelle_model"),
]

# (path, size, mtime_ns) -> sha256 hex. Module-level so a 30-video batch hashes
# each weight once.
_SHA_CACHE: dict = {}
_sha_compute_count = 0  # test hook: number of actual (uncached) hashings

# v1.1 W2.5: the in-process cache is also persisted to the per-user state dir
# so the FIRST preflight of an app launch stops re-hashing hundreds of MB of
# unchanged weights (the cold-start GUI freeze).  (path, size, mtime_ns) keys
# make staleness a non-issue; the Models tab's explicit Verify always does a
# full re-hash via weights.verify, independent of this cache.  Set
# MINDSIGHT_NO_HASH_CACHE=1 to disable persistence.
_SHA_CACHE_FILENAME = "weights_sha_cache.json"
_sha_disk_loaded = False


def _sha_cache_path() -> "Path | None":
    if os.environ.get("MINDSIGHT_NO_HASH_CACHE"):
        return None
    from mindsight.constants import state_dir
    return state_dir() / _SHA_CACHE_FILENAME


def _load_sha_disk_cache() -> None:
    global _sha_disk_loaded
    if _sha_disk_loaded:
        return
    _sha_disk_loaded = True
    path = _sha_cache_path()
    if path is None or not path.is_file():
        return
    try:
        for key, digest in json.loads(path.read_text()).items():
            fpath, size, mtime = key.rsplit("|", 2)
            _SHA_CACHE.setdefault((fpath, int(size), int(mtime)), digest)
    except Exception:
        pass    # corrupt/foreign cache: ignore, it will be rewritten


def _save_sha_disk_cache() -> None:
    path = _sha_cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in _SHA_CACHE.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    except Exception:
        pass    # best-effort persistence; never fail a run over the cache


def _sha256_file(path: Path) -> str:
    """sha256 of *path*, cached by (path, size, mtime_ns).

    The actual hashing loop lives once in :func:`mindsight.weights.sha256_file`;
    this wrapper adds a (path, size, mtime) cache -- in-process for the batch,
    persisted to the state dir across launches (W2.5) -- so unchanged weights
    hash once, ever.
    """
    global _sha_compute_count
    _load_sha_disk_cache()
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(path)
    _SHA_CACHE[key] = digest
    _sha_compute_count += 1
    _save_sha_disk_cache()
    return digest


def collect_weights(ns) -> dict:
    """Resolve + hash every configured weight (D7).

    Returns ``{dest: {backend, requested, resolved, sha256}}``; ``sha256`` is
    ``"missing"`` when the resolved file does not exist locally (auto-download
    default names never fetched on this machine).
    """
    out: dict = {}
    vp_file = getattr(ns, "vp_file", None)
    # LP2 --no-detector: the YOLO family is never loaded, so it must not
    # appear in preflight checks or the run-identity weight set.
    no_detector = getattr(ns, "no_detector", False)
    for backend, dest in _WEIGHT_TABLE:
        if no_detector and dest in ("model", "vp_model", "vp_file"):
            continue
        val = getattr(ns, dest, None)
        if not val:
            continue
        if dest == "vp_model" and not vp_file:
            continue
        name = str(val)
        if dest == "mgaze_model":
            # Extensionless family names pick their build per device at load
            # time (resnet50 -> resnet50_gaze.onnx off-CUDA).  Resolve the
            # same way here so preflight / manifest / run-identity all point
            # at the file the run actually loads (eyes-on A4: preflight
            # flagged "resnet50" missing while the run itself worked).
            name = resolve_mgaze_family(name, getattr(ns, "device", "auto"))
        resolved = resolve_weight(backend, name) if backend else Path(val)
        resolved = Path(resolved)
        entry = {
            "backend": backend,
            "requested": str(val),
            "resolved": str(resolved),
            "sha256": _sha256_file(resolved) if resolved.exists() else "missing",
        }
        out[dest] = entry
    return out


# ══════════════════════════════════════════════════════════════════════════════
# File identity
# ══════════════════════════════════════════════════════════════════════════════

def file_identity(path) -> dict:
    """Size, ISO mtime, and sha256 of *path* (webcam ints record no file)."""
    if isinstance(path, int):
        return {"kind": "webcam", "index": path}
    p = Path(str(path))
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "sha256": _sha256_file(p),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Run-identity hash (D6)
# ══════════════════════════════════════════════════════════════════════════════

# Model-wiring dests pulled from the namespace (device INCLUDED per Q3).
_WIRING_DESTS = ["model", "vp_file", "vp_model", "classes", "blacklist",
                 "device", "rf_gazelle_model", "rf_gazelle_name"]

# Static snapshot of plugin-contributed dests, welded to the live registries by
# tests/test_provenance.py::test_plugin_dests_match_registries.
_PLUGIN_DESTS = (
    # gaze backends
    "gazelle_model", "gazelle_name", "gazelle_inout_threshold",
    "gazelle_device", "gazelle_skip_frames", "gazelle_fp16", "gazelle_compile",
    "iris_refine", "iris_refine_weight", "iris_refine_upscale",
    "mpiifacegaze_model", "adas_gaze_model",
    # core backend (MGaze)
    "mgaze_model", "mgaze_arch", "mgaze_dataset",
    # object detection plugins
    "gaze_boost", "gaze_boost_factor", "gaze_boost_radius",
    "gaze_boost_min_conf", "gaze_boost_max_conf", "gaze_boost_classes",
    # phenomena plugins
    "eye_movement", "em_source", "em_saccade_thresh", "em_fixation_thresh",
    "em_min_fixation", "em_velocity_window",
    "novel_salience", "ns_speed_thresh", "ns_cooldown", "ns_history",
    "ns_flash",
    "pupillometry", "pupil_mode", "pupil_baseline", "pupil_upscale",
    "pupil_ir_thresh", "pupil_filter", "pupil_ema_alpha",
    "pupil_kalman_process_noise", "pupil_kalman_meas_noise",
    "pupil_ear_thresh", "pupil_blink_frames", "pupil_outlier_window",
    "pupil_per_eye",
)


def _canonical(obj):
    """JSON-safe canonical form: sets -> sorted lists; recurse dict/list."""
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    return obj


def run_identity(ns, *, config, weights) -> str:
    """Composite run-identity sha256 (D6).

    Payload = processing config (output/project sections removed) + model-wiring
    inputs (incl. device + plugin flag values) + sorted loaded-weight sha256s +
    ``__version__``. Output paths and project/video identity do NOT affect it, so
    changing --summary/--save (or the per-video output layout) does not force a
    reprocess; a --conf (or any processing) change does.
    """
    cfg = config.model_dump(mode="json")
    cfg.pop("output", None)
    cfg.pop("project", None)

    wiring: dict = {}
    vp_file = getattr(ns, "vp_file", None)
    for dest in _WIRING_DESTS:
        if dest == "vp_model" and not vp_file:
            continue
        wiring[dest] = getattr(ns, dest, None)
    for dest in _PLUGIN_DESTS:
        wiring[dest] = getattr(ns, dest, None)

    weight_shas = sorted(
        w["sha256"] for w in weights.values()
        if w.get("sha256") and w["sha256"] != "missing")

    payload = {
        "config": cfg,
        "wiring": _canonical(wiring),
        "weights": weight_shas,
        "version": __version__,
    }
    blob = json.dumps(_canonical(payload), sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Output-path resolution + manifest location (Q4 / D8)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_single_source_outputs(ns, source) -> dict:
    """Resolved file outputs for a single-source run ({name: path|None})."""
    from mindsight.outputs.csv_output import resolve_summary_path

    summary = resolve_summary_path(getattr(ns, "summary", None), source)
    save = getattr(ns, "save", None)
    save_path = None
    if save:
        if save is True:
            stem = (Path(str(source)).stem
                    if not isinstance(source, int) else "webcam")
            save_path = str(_OUTPUTS_ROOT / "Video" / f"{stem}_Video_Output.mp4")
        else:
            save_path = save
    return {
        "summary": summary,
        "log": getattr(ns, "log", None),
        "save": save_path,
        "heatmap": getattr(ns, "heatmap", None),
        "charts": getattr(ns, "charts", None),
    }


def manifest_path_for(output_paths) -> "str | None":
    """Q4 anchor: summary, else log, else save video, else heatmap/charts.

    Returns None when NO file output is configured (pure display run -> no
    manifest, per D8).
    """
    for key in ("summary", "log", "save", "heatmap", "charts"):
        anchor = output_paths.get(key)
        if anchor:
            p = Path(str(anchor))
            return str(p.parent / f"{p.stem}_manifest.json")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Manifest write (atomic)
# ══════════════════════════════════════════════════════════════════════════════

def _atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def write_run_manifest(path, *, ns, config, source, output_paths,
                       started, finished, status, error=None, meta=None) -> str:
    """Build and atomically write the per-run manifest to *path*.

    The manifest carries the full config dump + canonical hash, the composite
    run-identity, the environment, loaded-weight identities, source-file
    identity, the configured output paths, timestamps, and status/error.

    *meta* (SP3.1 Q2) is optional per-run staging provenance -- run.yaml's
    manifest-only ``date`` / ``session`` / ``notes`` / ``extra``.  When falsy it
    is omitted entirely, so flat-layout manifests are byte-unchanged.
    """
    weights = collect_weights(ns)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config": config.model_dump(mode="json"),
        "config_note": _CONFIG_NOTE,
        "config_canonical_hash": config.canonical_hash(),
        "run_identity": run_identity(ns, config=config, weights=weights),
        "environment": collect_environment(),
        "weights": weights,
        "source": file_identity(source),
        "outputs": {k: str(v) for k, v in output_paths.items() if v},
        "started": started,
        "finished": finished,
        "status": status,
        "error": error,
    }
    if meta:
        manifest["run_meta"] = meta
    _atomic_write_json(path, manifest)
    return str(path)
