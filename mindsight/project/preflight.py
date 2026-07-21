"""
project/preflight.py -- Structured project readiness checklist (SP3.1 D4).

``run_preflight`` inspects a project directory and returns a
:class:`PreflightReport` -- a list of :class:`CheckResult` covering pipeline
config validity, weight presence (with sha256 recorded for SP4), the VP file,
discovered runs, participant/condition coverage, device availability, disk
headroom, and plugin load errors (surfaced LOUDLY per D5).  Every check runs
under try/except: preflight NEVER raises, so the GUI checklist and the CLI
``--preflight`` report always render.

The report is READ-ONLY -- it builds no models and runs no videos.  It is
exposed via :meth:`mindsight.project.project.Project.preflight` and the CLI
``--preflight`` flag (SP3.1 D16).

Design bindings: D4 (checks + report shape), D5 (registry ``load_errors``), Q5
Option A (lists active DataCollection plugins + surfaces their load errors).
"""
from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path

from mindsight.project.runner import (
    discover_participant_ids,
    discover_sources,
    discover_vp_file,
    load_project_config,
)
from mindsight.project.staging import (
    _KNOWN_RUN_KEYS as _RUN_YAML_KEYS,
    AMBIGUOUS,
    RUN_FOLDER,
    detect_layout,
    discover_run_specs,
    inspect_run_folders,
)

# ══════════════════════════════════════════════════════════════════════════════
# Report data model (D4)
# ══════════════════════════════════════════════════════════════════════════════

_OK, _WARN, _FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class CheckResult:
    """One preflight check outcome.

    ``severity`` is ``ok`` / ``warn`` / ``fail``; ``fix_hint`` is a short,
    plain-English remedy shown beside a warn/fail line.
    """
    id: str
    label: str
    severity: str
    message: str
    fix_hint: str = ""


@dataclass(frozen=True)
class PreflightReport:
    """A full readiness checklist; ``ok`` is True when no check FAILED."""
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return not any(c.severity == _FAIL for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.severity == _FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.severity == _WARN)


# ══════════════════════════════════════════════════════════════════════════════
# Default injectable seams (device / disk / registries) -- overridable in tests
# ══════════════════════════════════════════════════════════════════════════════

def _default_device_check(requested: str) -> tuple[bool, str]:
    """Return ``(available, resolved)`` for a requested device string.

    ``auto`` always resolves (CUDA > MPS > CPU); an explicit ``cuda`` / ``mps``
    is available only when the backend reports it.
    """
    import torch

    from mindsight.utils.device import resolve_device
    if requested == "auto":
        return True, str(resolve_device("auto"))
    if requested == "cpu":
        return True, "cpu"
    if requested == "cuda":
        return bool(torch.cuda.is_available()), "cuda"
    if requested == "mps":
        avail = bool(getattr(torch.backends, "mps", None)
                     and torch.backends.mps.is_available())
        return avail, "mps"
    return False, requested


def _default_registries() -> list:
    """The four module-level plugin registries (D5 reads their load_errors)."""
    from Plugins import (
        data_collection_registry,
        gaze_registry,
        object_detection_registry,
        phenomena_registry,
    )
    return [gaze_registry, object_detection_registry,
            phenomena_registry, data_collection_registry]


# ══════════════════════════════════════════════════════════════════════════════
# Working-namespace assembly (weights + device resolution)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_pipeline_yaml(project: Path, project_cfg) -> Path:
    """Where the project's pipeline YAML lives (project.yaml override or default)."""
    if project_cfg and getattr(project_cfg, "pipeline_path", None):
        return project / project_cfg.pipeline_path
    return project / "Pipeline" / "pipeline.yaml"


def _build_work_ns(ns, pipeline_yaml: Path):
    """A namespace to resolve weights/device from -- the caller's ns (copied so
    it is not mutated) or a fresh default parse, with the project pipeline YAML
    merged in exactly as a real run would (T7 precedence preserved)."""
    if ns is None:
        from mindsight.cli_flags import parse_cli
        work = parse_cli([])
    else:
        work = copy.copy(ns)
    if pipeline_yaml and pipeline_yaml.exists():
        try:
            from mindsight.config_compat import load_pipeline
            load_pipeline(pipeline_yaml, work)
        except Exception:
            # A broken pipeline YAML is reported by the pipeline_config check;
            # weight/device checks fall back to whatever the ns already carries.
            pass
    return work


def _safe_sources(project: Path) -> list[Path]:
    try:
        return discover_sources(project)
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Individual checks (each returns exactly one CheckResult; never raises upward)
# ══════════════════════════════════════════════════════════════════════════════

def _check_structure(project: Path, layout: str) -> CheckResult:
    label = "Project structure"
    if not project.is_dir():
        return CheckResult("project_structure", label, _FAIL,
                           f"project directory not found: {project}",
                           "create the project directory")
    if layout == AMBIGUOUS:
        return CheckResult("project_structure", label, _FAIL,
                           "both Inputs/Runs/ and Inputs/Videos/ are populated "
                           "-- the layout is ambiguous",
                           "keep run folders (Inputs/Runs/) OR flat videos "
                           "(Inputs/Videos/), not both")
    if layout == RUN_FOLDER:
        return CheckResult("project_structure", label, _OK,
                           "Inputs/Runs/ present (run-folder layout)")
    if not (project / "Inputs" / "Videos").is_dir():
        return CheckResult("project_structure", label, _FAIL,
                           f"missing Inputs/Videos/ under {project.name}",
                           "create Inputs/Videos/ and add source videos")
    return CheckResult("project_structure", label, _OK,
                       "Inputs/Videos/ present")


def _check_pipeline_config(pipeline_yaml: Path) -> CheckResult:
    label = "Pipeline config"
    if pipeline_yaml is None or not pipeline_yaml.exists():
        return CheckResult("pipeline_config", label, _WARN,
                           "no pipeline config found; schema defaults will apply",
                           "add Pipeline/pipeline.yaml or set 'pipeline:' in project.yaml")
    try:
        from mindsight.config_compat import load_yaml
        load_yaml(pipeline_yaml)   # models are extra=forbid -> strict validation
    except Exception as exc:
        return CheckResult("pipeline_config", label, _FAIL,
                           f"{pipeline_yaml.name} invalid: {exc}",
                           "fix the pipeline YAML (see the error above)")
    return CheckResult("pipeline_config", label, _OK,
                       f"{pipeline_yaml.name} loads under the schema")


def _check_weights(work_ns) -> CheckResult:
    label = "Weights"
    from mindsight.outputs.provenance import collect_weights
    weights = collect_weights(work_ns)
    if not weights:
        return CheckResult("weights", label, _WARN,
                           "no weights configured (auto-download names not resolved)",
                           "set model/backend weight paths in the pipeline config")
    parts, missing = [], []
    for dest, w in weights.items():
        sha = w.get("sha256")
        resolved = w.get("resolved", "")
        name = Path(resolved).name
        if sha == "missing":
            missing.append(dest)
            # Print the resolved ABSOLUTE path, not just the basename -- weight
            # resolution is global (the shared Weights folder), so the user needs
            # to know exactly where the file is expected on disk.
            abs_path = str(Path(resolved).resolve()) if resolved else name
            parts.append(f"{dest}={abs_path} [MISSING]")
        else:
            parts.append(f"{dest}={name} [{sha[:12]}]")
    msg = "; ".join(parts)
    if missing:
        return CheckResult("weights", label, _FAIL,
                           f"weight file(s) not found -- {msg}",
                           "place the weight at the resolved absolute path above "
                           "(the shared Weights folder -- weight resolution is "
                           "global, not per-project), or fix the config path")
    return CheckResult("weights", label, _OK, msg)


def _check_vp_file(project: Path, work_ns) -> CheckResult:
    label = "Visual prompt"
    import json
    vp = discover_vp_file(project)
    # VP (YOLOE visual-prompt) detection is active only when a vp_file is
    # configured -- vp_model carries a non-None auto-download default and does
    # NOT by itself mean VP mode (mirrors provenance.collect_weights).
    vp_mode = bool(getattr(work_ns, "vp_file", None))
    if vp is None:
        if vp_mode:
            return CheckResult("vp_file", label, _WARN,
                               "VP-mode detector configured but no .vp.json in Inputs/Prompts/",
                               "add a visual prompt to Inputs/Prompts/ (VP Builder)")
        return CheckResult("vp_file", label, _OK, "no VP file (VP mode not active)")
    try:
        data = json.loads(Path(vp).read_text())
    except Exception as exc:
        return CheckResult("vp_file", label, _FAIL,
                           f"{Path(vp).name} is not valid JSON: {exc}",
                           "re-export the visual prompt from VP Builder")
    refs = data.get("references") or []
    has_annot = any(isinstance(r, dict) and r.get("annotations") for r in refs)
    if not refs or not has_annot:
        return CheckResult("vp_file", label, _FAIL,
                           f"{Path(vp).name} has no reference with annotations",
                           "add at least one annotated reference in VP Builder")
    return CheckResult("vp_file", label, _OK,
                       f"{Path(vp).name} parses ({len(refs)} reference(s))")


def _resolve_pid_maps(project: Path, project_cfg):
    if project_cfg and getattr(project_cfg, "participants", None):
        return project_cfg.participants
    try:
        return discover_participant_ids(project)
    except Exception:
        return None


def _collect_runs(project: Path, project_cfg, layout: str) -> list[dict]:
    """Uniform per-run coverage view for the runs/participants/conditions checks.

    Each entry: ``{run_id, has_pid, has_cond, info}`` where ``info`` is the
    :class:`RunFolderInfo <mindsight.project.staging.RunFolderInfo>` (run-folder)
    or ``None`` (legacy).  Never raises -- ``inspect_run_folders`` is non-raising
    and legacy discovery is wrapped by ``_safe_sources``.
    """
    runs: list[dict] = []
    if layout == RUN_FOLDER:
        try:
            csv_pid = discover_participant_ids(project)
        except Exception:
            csv_pid = None
        cfg_pid = (project_cfg.participants
                   if project_cfg and getattr(project_cfg, "participants", None)
                   else {})
        cfg_cond = (project_cfg.conditions
                    if project_cfg and getattr(project_cfg, "conditions", None)
                    else {})
        for info in inspect_run_folders(project):
            rid = info.run_id
            pid = info.meta.pid_map or cfg_pid.get(rid) or (
                csv_pid.get(rid) if csv_pid else None)
            tags = info.meta.conditions or cfg_cond.get(rid, [])
            runs.append({"run_id": rid, "has_pid": bool(pid),
                         "has_cond": bool(tags), "info": info})
    elif layout != AMBIGUOUS:
        pid_maps = _resolve_pid_maps(project, project_cfg)
        conditions = (project_cfg.conditions
                      if project_cfg and getattr(project_cfg, "conditions", None)
                      else {})
        for s in _safe_sources(project):
            runs.append({"run_id": s.name,
                         "has_pid": bool(pid_maps and pid_maps.get(s.name)),
                         "has_cond": bool(conditions.get(s.name)),
                         "info": None})
    return runs


def _check_runs(runs, project: Path, layout: str) -> CheckResult:
    label = "Runs discovered"
    if layout == AMBIGUOUS:
        return CheckResult("runs_discovered", label, _FAIL,
                           "ambiguous layout -- no runs to process",
                           "resolve the Inputs/Runs vs Inputs/Videos ambiguity")
    if layout == RUN_FOLDER:
        if not runs:
            return CheckResult("runs_discovered", label, _FAIL,
                               "no run folders found in Inputs/Runs/",
                               "add a run folder Inputs/Runs/<run_id>/ with one video")
        # UP5: run.yaml-without-video = a PLANNED live session, not an error.
        from mindsight.project.staging import is_planned
        planned = [r["run_id"] for r in runs if is_planned(r["info"])]
        bad = [r["run_id"] for r in runs
               if len(r["info"].videos) != 1 and r["run_id"] not in planned]
        if bad:
            return CheckResult("runs_discovered", label, _FAIL,
                               "run folder(s) not holding exactly one video: "
                               + ", ".join(bad),
                               "each Inputs/Runs/<run_id>/ needs exactly one primary video")
        recorded = len(runs) - len(planned)
        if planned and not recorded:
            return CheckResult("runs_discovered", label, _WARN,
                               f"all {len(planned)} session(s) awaiting "
                               "recording",
                               "record them live or attach their footage "
                               "from the runs table in Analyze Footage")
        if planned:
            return CheckResult("runs_discovered", label, _OK,
                               f"{recorded} run folder(s) discovered, "
                               f"{len(planned)} session(s) awaiting recording")
        return CheckResult("runs_discovered", label, _OK,
                           f"{len(runs)} run folder(s) discovered")
    if not (project / "Inputs" / "Videos").is_dir():
        return CheckResult("runs_discovered", label, _FAIL,
                           "Inputs/Videos/ is missing -- no runs to process",
                           "create Inputs/Videos/ and add source videos")
    if not runs:
        return CheckResult("runs_discovered", label, _FAIL,
                           "no video/image sources found in Inputs/Videos/",
                           "add at least one source to Inputs/Videos/")
    return CheckResult("runs_discovered", label, _OK,
                       f"{len(runs)} source(s) discovered")


def _check_output_collisions(project: Path, project_cfg, layout: str) -> CheckResult:
    """Two sources that stage the SAME output CSV clobber each other (B1 F3).

    ``a.mp4`` and ``a.mov`` in Inputs/Videos/ both stem to ``a_Events.csv`` --
    the second run truncates the first's Events/summary CSVs (writers open in
    mode ``"w"``).  Flag any duplicate staged ``log`` path as a FAIL so the user
    renames one source before the run silently loses data.
    """
    label = "Output collisions"
    if layout == AMBIGUOUS:
        return CheckResult("output_collisions", label, _OK,
                           "layout ambiguous -- collision check skipped")
    specs = discover_run_specs(project, project_cfg, layout=layout)
    by_log: dict[str, list] = {}
    for spec in specs:
        by_log.setdefault(spec.output_paths.get("log"), []).append(spec)
    collisions = []
    for log_path, group in by_log.items():
        if len(group) > 1:
            names = " and ".join(sorted(Path(s.source).name for s in group))
            collisions.append(f"{names} both write {Path(log_path).name}")
    if collisions:
        return CheckResult("output_collisions", label, _FAIL,
                           "output name collision: " + " | ".join(collisions),
                           "rename one of the colliding source files")
    return CheckResult("output_collisions", label, _OK,
                       "no output name collisions")


def _check_run_metadata(runs) -> CheckResult:
    """Run-folder only: run.yaml validity (FAIL) + unknown keys (WARN), Q2."""
    label = "Run metadata"
    errors, warns = [], []
    for r in runs:
        meta = r["info"].meta
        if meta.error:
            errors.append(f"{r['run_id']}: {meta.error}")
        if meta.unknown_keys:
            warns.append(f"{r['run_id']}: unknown key(s) "
                         f"{', '.join(meta.unknown_keys)}")
    if errors:
        return CheckResult("run_metadata", label, _FAIL,
                           "run.yaml problem(s): " + " | ".join(errors),
                           "fix the run.yaml metadata (see the message)")
    if warns:
        return CheckResult("run_metadata", label, _WARN,
                           "unknown run.yaml key(s): " + " | ".join(warns),
                           f"use only: {', '.join(sorted(_RUN_YAML_KEYS))}")
    return CheckResult("run_metadata", label, _OK,
                       f"run.yaml valid for {len(runs)} run(s)")


def _check_participants(runs) -> CheckResult:
    label = "Participant coverage"
    if not runs:
        return CheckResult("participants_coverage", label, _OK,
                           "no runs to check")
    uncovered = [r["run_id"] for r in runs if not r["has_pid"]]
    if uncovered:
        return CheckResult("participants_coverage", label, _WARN,
                           f"{len(uncovered)}/{len(runs)} run(s) without "
                           f"participant labels: {', '.join(uncovered)}",
                           "add participants in project.yaml / participant_ids.csv "
                           "(optional -- defaults P0, P1, ...)")
    return CheckResult("participants_coverage", label, _OK,
                       "all runs have participant labels")


def _check_conditions(runs) -> CheckResult:
    label = "Condition coverage"
    if not runs:
        return CheckResult("conditions_coverage", label, _OK, "no runs to check")
    uncovered = [r["run_id"] for r in runs if not r["has_cond"]]
    if uncovered:
        return CheckResult("conditions_coverage", label, _WARN,
                           f"{len(uncovered)}/{len(runs)} run(s) without "
                           f"conditions: {', '.join(uncovered)}",
                           "add conditions in project.yaml (optional)")
    return CheckResult("conditions_coverage", label, _OK,
                       "all runs have conditions")


def _check_device(work_ns, device_check) -> CheckResult:
    label = "Compute device"
    requested = getattr(work_ns, "device", "auto") or "auto"
    try:
        available, resolved = device_check(requested)
    except Exception as exc:
        return CheckResult("device", label, _FAIL,
                           f"device '{requested}' check failed: {exc}",
                           "use --device auto or cpu")
    # W3Y item 4: an NVIDIA GPU with a CPU-only torch wheel silently
    # degrades every device decision -- surface it LOUDLY as a warn (the
    # run still works, just slower/on the wrong 'optimal' weights).
    try:
        from mindsight.utils.device import cuda_support_note
        note = cuda_support_note()
    except Exception:
        note = None
    if note:
        return CheckResult("device", label, _WARN, note,
                           "install a CUDA-enabled torch build")
    if requested == "auto":
        return CheckResult("device", label, _OK, f"auto -> {resolved}")
    if not available:
        return CheckResult("device", label, _FAIL,
                           f"device '{requested}' is not available",
                           "choose an available device or use auto/cpu")
    return CheckResult("device", label, _OK, f"{requested} available")


def _check_disk(project, sources, work_ns, disk_usage) -> CheckResult:
    label = "Disk space"
    if not getattr(work_ns, "save", None):
        return CheckResult("disk_space", label, _OK,
                           "video output off -- extra headroom not required")
    total = 0
    for s in sources:
        try:
            total += s.stat().st_size
        except OSError:
            pass
    try:
        free = disk_usage(str(project)).free
    except Exception as exc:
        return CheckResult("disk_space", label, _WARN,
                           f"could not check free space: {exc}",
                           "verify the output volume manually")
    need = int(total * 1.5)
    if free < need:
        return CheckResult("disk_space", label, _WARN,
                           f"low disk: {free / 1e9:.1f} GB free < "
                           f"{need / 1e9:.1f} GB (1.5x inputs)",
                           "free up space or disable video output")
    return CheckResult("disk_space", label, _OK,
                       f"headroom ok ({free / 1e9:.1f} GB free)")


def _check_plugins(registries, work_ns) -> CheckResult:
    label = "Plugins"
    errors: list[str] = []
    for reg in registries:
        for path_str, exc_str in getattr(reg, "load_errors", []):
            errors.append(f"{Path(path_str).name}: {exc_str}")
    # Q5 (Option A): list active DataCollection plugins (built via from_args).
    try:
        from mindsight.factory import build_data_plugins
        active_dc = build_data_plugins(work_ns)
    except Exception:
        active_dc = []
    if active_dc:
        names = ", ".join(getattr(p, "name", type(p).__name__) for p in active_dc)
        dc_note = f"data-collection plugins active: {names}"
    else:
        dc_note = "no data-collection plugins active"
    if errors:
        return CheckResult("plugins", label, _FAIL,
                           "plugin load error(s): " + " | ".join(errors),
                           "fix or remove the failing plugin module")
    return CheckResult("plugins", label, _OK, f"all plugins loaded; {dc_note}")


def _check_weights_verify(work_ns) -> CheckResult:
    """Verify PRESENT configured weights against the checksummed manifest (D5).

    For every weight ``collect_weights`` found on disk, look it up in
    ``weights_manifest.json`` by filename: a checksum match is OK; a mismatch is
    a FAIL (the file differs from the published weight); a weight with no
    manifest entry is a WARN (a custom / user-supplied weight, allowed).  Missing
    files are left to the ``weights`` check (this one only judges present bytes).
    """
    label = "Weight verification"
    from mindsight import weights as weights_mod
    from mindsight.outputs.provenance import collect_weights

    present = {dest: w for dest, w in collect_weights(work_ns).items()
               if w.get("sha256") and w["sha256"] != "missing"}
    if not present:
        return CheckResult("weights_verify", label, _OK,
                           "no present weights to verify")
    try:
        weights_mod.load_manifest()
    except weights_mod.WeightsError as exc:
        return CheckResult("weights_verify", label, _WARN,
                           f"manifest unavailable: {exc}",
                           "reinstall or restore weights_manifest.json")

    ok, mismatches, unknown = [], [], []
    for dest, w in present.items():
        name = Path(w.get("resolved", "")).name
        entry = weights_mod.find_entry(name, backend=w.get("backend"))
        if entry is None:
            unknown.append(name)
        elif entry.get("sha256") == w["sha256"]:
            ok.append(f"{name} [{w['sha256'][:12]}]")
        else:
            mismatches.append((name, entry.get("label", name)))
    if mismatches:
        detail = "; ".join(f"{n} differs from the published '{lbl}' weight"
                           for n, lbl in mismatches)
        return CheckResult("weights_verify", label, _FAIL, detail,
                           "re-download the weight via the Models tab "
                           "(mindsight-weights --force)")
    if unknown:
        return CheckResult("weights_verify", label, _WARN,
                           "custom weight(s) not in the manifest: "
                           + ", ".join(unknown),
                           "verified against the manifest where possible; "
                           "custom weights are allowed")
    return CheckResult("weights_verify", label, _OK,
                       "all present weights match the manifest: " + "; ".join(ok))


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_preflight(project_dir, project_cfg=None, ns=None, *,
                  registries=None, device_check=None, disk_usage=None
                  ) -> PreflightReport:
    """Assemble the readiness checklist for *project_dir* (never raises).

    Parameters
    ----------
    project_dir : str or Path
        The project directory (need not be fully valid -- structural problems
        are reported as checks, not exceptions).
    project_cfg : ProjectConfig or None
        The loaded project config; loaded from ``project.yaml`` when ``None``.
    ns : Namespace or None
        A namespace to resolve weights/device from (the CLI ``args`` or a GUI
        namespace).  When ``None`` a default parse is used.  It is COPIED before
        the project pipeline YAML is merged, so the caller's ns is untouched.
    registries, device_check, disk_usage :
        Injectable seams (default to the live registries / torch / shutil) so
        fast tests can exercise every branch without real hardware.
    """
    project = Path(project_dir).resolve()
    if project_cfg is None:
        try:
            project_cfg = load_project_config(project)
        except Exception:
            project_cfg = None

    pipeline_yaml = _resolve_pipeline_yaml(project, project_cfg)
    work_ns = _build_work_ns(ns, pipeline_yaml)
    try:
        layout = detect_layout(project)
    except Exception:
        layout = "legacy"
    runs = _collect_runs(project, project_cfg, layout)
    # Disk sizing uses the concrete source files (legacy flat list); harmless
    # for run-folder projects (video output off / per-run sizes not summed here).
    sources = _safe_sources(project)
    registries = registries if registries is not None else _default_registries()
    device_check = device_check or _default_device_check
    disk_usage = disk_usage or shutil.disk_usage

    def _safe(check_id, label, fn) -> CheckResult:
        try:
            return fn()
        except Exception as exc:      # preflight NEVER raises (D4)
            return CheckResult(check_id, label, _FAIL,
                               f"check crashed: {exc}",
                               "report this -- a preflight check errored")

    checks = [
        _safe("project_structure", "Project structure",
              lambda: _check_structure(project, layout)),
        _safe("pipeline_config", "Pipeline config",
              lambda: _check_pipeline_config(pipeline_yaml)),
        _safe("weights", "Weights",
              lambda: _check_weights(work_ns)),
        _safe("weights_verify", "Weight verification",
              lambda: _check_weights_verify(work_ns)),
        _safe("vp_file", "Visual prompt",
              lambda: _check_vp_file(project, work_ns)),
        _safe("runs_discovered", "Runs discovered",
              lambda: _check_runs(runs, project, layout)),
    ]
    # Run-folder projects gain a run.yaml metadata check (Q2) right after the
    # runs check; the legacy checklist is unchanged (byte-for-byte).
    if layout == RUN_FOLDER:
        checks.append(_safe("run_metadata", "Run metadata",
                            lambda: _check_run_metadata(runs)))
    checks.append(_safe("output_collisions", "Output collisions",
                        lambda: _check_output_collisions(
                            project, project_cfg, layout)))
    checks += [
        _safe("participants_coverage", "Participant coverage",
              lambda: _check_participants(runs)),
        _safe("conditions_coverage", "Condition coverage",
              lambda: _check_conditions(runs)),
        _safe("device", "Compute device",
              lambda: _check_device(work_ns, device_check)),
        _safe("disk_space", "Disk space",
              lambda: _check_disk(project, sources, work_ns, disk_usage)),
        _safe("plugins", "Plugins",
              lambda: _check_plugins(registries, work_ns)),
    ]
    return PreflightReport(checks=checks)


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print (CLI --preflight; reusable by the GUI checklist)
# ══════════════════════════════════════════════════════════════════════════════

_SEV_TAG = {_OK: "OK  ", _WARN: "WARN", _FAIL: "FAIL"}


def format_report(report: PreflightReport, *, title: str = "") -> str:
    """Render *report* as an aligned OK/WARN/FAIL checklist with fix hints."""
    lines: list[str] = []
    if title:
        lines.append(f"Preflight: {title}")
        lines.append("=" * 60)
    for c in report.checks:
        tag = _SEV_TAG.get(c.severity, c.severity.upper())
        lines.append(f"  [{tag}] {c.label}: {c.message}")
        if c.severity != _OK and c.fix_hint:
            lines.append(f"         fix: {c.fix_hint}")
    lines.append("-" * 60)
    if report.ok:
        lines.append(f"Preflight PASSED ({report.n_warn} warning(s))")
    else:
        lines.append(f"Preflight FAILED "
                     f"({report.n_fail} failure(s), {report.n_warn} warning(s))")
    return "\n".join(lines)
