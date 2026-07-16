# Changelog

## [Unreleased]

### Added (W3X fire-decision knobs, all default-off)
- `--rf-reuse-eps`: skip a scheduled Gaze-LLE call when the scene is
  visually unchanged since the last real call (mean-abs 64x64 grayscale
  frame diff below the threshold, stable face boxes) and re-anchor the
  cached heatmaps instead -- saves full forward passes on static footage.
- `--rf-onset-samples`: let a newly appeared face reach fixation
  eligibility after N gaze samples instead of the default 5, cutting
  first-correction latency for fresh faces.
- `--rf-onset-gap`: when a face that never had a correction wants one,
  relax the global call gap to min(`--min-call-gap`, N) so a new
  participant is not stuck behind another face's recent call (up to a
  second at defaults).

### Fixed
- **Face identity is now the stable track ID in every output.** The internal
  `hits` set and the mutual-gaze / social-referencing / gaze-leadership(tip)
  trackers previously used list-position indices, so when face order changed
  mid-video, "P0" could mean different people in different files and custom
  participant labels could attach to the wrong faces. All outputs now share
  the track-ID convention `{stem}_Events.csv` always used.
- **`gaze_following` summary rows had the episode stream's columns
  swapped.** Both now use participant = follower, partner = leader.
- The Gaze-LLE in/out-of-frame head is now read on the blend path (it was
  loaded but discarded); gating lands with the v1.1 accuracy work.
- DataCollection plugins' `on_frame` / `on_run_complete` lifecycle hooks are
  now actually invoked, and their CLI flags register with the parser.
- Bare `--anonymize` now means `blur` instead of crashing; stale `--summary`
  help text corrected; heatmap backgrounds fall back to the first readable
  frame when the mid-frame seek fails; the GUI settings dir honors
  `MINDSIGHT_STATE_DIR` / `MINDSIGHT_HOME` so relocated installs stop sharing
  one `~/.mindsight`.

### Added
- **New per-frame gaze stream `{stem}_gaze.csv`** (and `Global_gaze.csv`):
  one row per face per frame, hits or not -- gaze angles, ray origin and
  endpoint, snap flags, blend telemetry (trust / accepted inference /
  in-out score), depth at endpoint, and objects hit.
- **8 additive `{stem}_Events.csv` columns** after `participant_label`:
  `gaze_conf`, `gaze_pitch`, `gaze_yaw` (degrees), `ray_end_x`, `ray_end_y`,
  `depth_at_hit`, `ray_snapped`, `ray_extended`. The original columns are
  unchanged.
- `--profile` now reports `detect` / `depth` / `gaze` / `gazelle` /
  `phenomena` / `draw` / `dashboard` separately (gaze and Gaze-LLE were
  previously hidden inside `detect`).
- The in-app About reader carries all ten guide pages and degrades
  unsupported syntax (collapsibles, diagrams, grid cards) gracefully.
- **Every annotated visual-prompt reference image is now used**: per-class
  embeddings are mean-pooled across references (1.0 silently used only the
  first). Single-reference files behave exactly as before; the VP Builder's
  Test runs the same pooled priming as real runs.
- **`--rf-inout-gate T`** activates the Gaze-LLE checkpoint's
  in/out-of-frame head on the blend path (auto-upgrading to the `_inout`
  architecture when the checkpoint carries the head): heatmap accepts below
  the gate are vetoed and blend trust scales with the in/out score.
  Default 0.0 keeps 1.0.0 behavior exactly.
- Performance opt-ins (all default-off): `--mgaze-reuse-eps` skips the
  per-face gaze model on visually-unchanged face crops; `--rf-gazelle-fp16`
  / `--rf-gazelle-compile` reach the blend path; `--face-conf` /
  `--face-input-size` expose the face detector; weight hashes persist
  across launches so preflight stops re-hashing unchanged weights
  (`MINDSIGHT_NO_HASH_CACHE=1` opts out).
- `yolo11n.pt` added to the weights manifest (candidate default pending
  eval); `scripts/eval_annotate.py` + `scripts/eval_gaze.py` give accuracy
  work ground-truth numbers.
- Note: config hashes changed with the new schema fields, so pre-v1.1
  resume ledgers report a config mismatch and reprocess once.

## [1.0.0] - 2026-07-12

First stable release. Everything between the v0.2.0 beta and here was a
ground-up rebuild: a public pipeline API under a typed configuration schema,
a reworked GUI organized around studies, research-grade outputs, one-click
installers, and a documentation site.

### Changed (breaking)
- **Package restructure.** All domain code lives under `mindsight/` (the old
  `ms/` package and its shims are gone). The CLI and GUI are thin frontends
  over a public pipeline API; `pipeline.yaml` configs are validated against a
  strict typed schema, with compatibility aliases for legacy keys.
- **Gazelle Blend redesigned around a fixation-aware scheduler.** The
  belief-map tuning knobs (`direction_blend`, `length_blend`, `length_only`,
  `direction_decay`, `length_decay`, `diffusion_sigma`, `blend_conf_scale`,
  `belief_min_peak`, `inout_threshold`, and the fixed `gazelle_interval`) are
  removed. Replaced with 3 default-visible knobs (`min_call_gap`, `dir_beta`,
  `len_beta`) and 4 advanced knobs (`fixation_v_threshold`,
  `fixation_d_threshold`, `dir_min_cutoff`, `len_min_cutoff`). Gaze-LLE
  inferences fire only when at least one participant is fixating -- detected
  per-face from smoothed pitch/yaw velocity and windowed dispersion -- which
  prevents the post-inference head-turn artifact at its source. Output is
  smoothed with a One Euro Filter (adaptive per-frame cutoff, calibrated to
  the video's real fps) instead of a fixed-alpha EMA. Legacy
  `rf_gazelle_interval` / `--rf-gazelle-interval` still work as aliases for
  `min_call_gap`; the other removed knobs have no 1:1 replacement.
- **Dependencies are declared once, in `pyproject.toml`,** with exact versions
  pinned in the committed `uv.lock`. `requirements.txt` is gone.

### Added
- **Projects workflow**: a Projects tab with a study-creation wizard (videos,
  conditions, participant tagging), per-run data pane, planned sessions that
  can be fulfilled by live recording or attached external footage, and live
  record-then-analyze sessions that keep the raw camera feed as the run's
  primary video.
- **Inference Settings dialog** (seven tabs) owning the run-settings store;
  Gaze Tuning (now "Inference Tuning") no longer affects runs outside it.
- **Quick analysis**: analyze a bare video file or camera without a project,
  with live charts in-pane.
- **Crop & Adjust tool** with YOLOE-assisted auto-crop, non-destructive by
  default; frame extraction into the Visual Prompt Builder; portable
  `.vp.zip` visual-prompt archives.
- **About tab with an in-app documentation reader.** The wheel bundles the
  docs tree, config presets, and weights manifest as package data, so
  installed apps read the shipped docs offline.
- **One-click installers** for macOS and Windows: managed Python, locked
  dependencies, verified weight downloads -- and a real `MindSight.app` in
  /Applications (Dock name + icon) on macOS, Start Menu shortcuts with the
  MindSight icon on Windows.
- **Known-good pipeline preset** shipped at `configs/pipeline_known_good.yaml`
  (the validated Gaze-LLE Blend operating point), plus a low-power variant.
- **`--no-detector` mode** -- face and gaze analysis without any YOLO model.
- **Documentation site** (tutorial with screenshots, concepts, reference)
  and a GitHub issue template.

### Fixed
- `mindsight --help` crashed with `KeyError: 'default'` (help rendered
  during the explicit-flag detection parse; the real parse now runs first).
- The committed `uv.lock` had drifted from `pyproject.toml` (it was missing
  `imageio-ffmpeg` and the macOS camera-enumeration dependency).

### Removed
- **UniGaze gaze backend** -- never loaded reliably (required non-commercial
  `unigaze` PyPI package pinning `timm==0.3.2`); superseded by the MobileGaze
  and Gaze-LLE backends
- **GazelleSnap plugin** -- superseded by the core Ray Forming + Gazelle
  Blend pipeline. Legacy `--gs-*` CLI flags are no longer recognized

## [0.4.0-beta] - 2026-04-05

### Changed
- **Package restructure** -- all domain code moved under `ms/` package; `pip install -e .` now required; `mindsight` and `mindsight-gui` console commands available

### Added
- **L2CS-Net gaze backend** — dual classification heads, 3.92° MAE on MPIIGaze (vs 11° for MGaze)
- **UniGaze gaze backend** — ViT + MAE pre-training, best cross-dataset accuracy (~9.4°, non-commercial license)
- **Backend registry** — automatic discovery of gaze backends from `ms/GazeTracking/Backends/`
- **Unified pitchyaw pipeline** — `pitchyaw_pipeline.py` shared by all pitch/yaw-based backends
- **Live matplotlib dashboard** — real-time per-tracker charts during processing (`ms/GUI/live_dashboard.py`)
- **Dashboard bridge** — thread-safe GUI-to-dashboard data flow (`ms/GUI/live_dashboard_bridge.py`)
- **Post-run chart generation** — time-series charts via `ms/DataCollection/chart_output.py` (`--charts` flag)
- **Global CSV** — cross-video summary and per-condition statistics for project mode (`ms/DataCollection/global_csv.py`)
- **Matplotlib dashboard renderer** — `ms/DataCollection/dashboard_matplotlib.py` replaces OpenCV drawing
- **Face anonymization** — `--anonymize blur|black` with configurable padding and temporal smoothing
- **Auxiliary video streams** — per-participant secondary cameras (eye-tracking, FPV) via `AuxStreamConfig`
- **Participant ID mapping** — custom labels via `pid_map` in `project.yaml` and `ms/participant_ids.py`
- **ProjectConfig / ProjectOutputConfig** — study metadata dataclasses for project mode
- **Example project template** — `Projects/ExampleProject/` with `project.yaml`
- **CollapsibleGroupBox widget** — expandable/collapsible GUI sections
- **GazelleSnap plugin** — snap-augmented Gazelle gaze backend
- **GazeBoost plugin** — gaze-informed object detection boost using pitchyaw pipeline
- **Device auto-detection** — `ms/utils/device.py` for CUDA/MPS/CPU hardware selection
- **Plugin protocol methods** — `dashboard_data()`, `latest_metric()`, `latest_metrics()`, `dashboard_widget()`, `generate_charts()`
- **Performance flags** — `--fast-mode`, `--skip-phenomena N`, `--lite-overlay`, `--no-dashboard`, `--profile`
- **Gaze convergence tips** — `--gaze-tips` + `--tip-radius` for multi-person gaze convergence visualization

### Changed
- **MGaze relocated** from `Plugins/GazeTracking/MGaze/` to `ms/GazeTracking/Backends/MGaze/`
- **CLI flags renamed** — `--gaze-model` → `--mgaze-model`, `--gaze-arch` → `--mgaze-arch`, `--gaze-dataset` → `--mgaze-dataset`
- **GazeConfig.adaptive_ray** — type changed from `bool` to `str` (`"off"` / `"extend"` / `"snap"`)
- **`ja_conf_gate` renamed to `hit_conf_gate`** — broader semantics beyond joint attention
- **Adaptive snap scoring** — new parameters: `snap_bbox_scale`, `snap_w_dist`, `snap_w_size`, `snap_w_intersect`
- **GazeConfig additions** — `detect_extend`, `detect_extend_scope`, `forward_gaze_threshold`
- **TrackerConfig** — added `reid_max_dist` (default 200, up from 120)
- **OutputConfig** — added `charts_path`, `pid_map`, `aux_streams`, `anonymize`, `anonymize_padding`, `video_name`, `conditions`
- **Plugin signatures** — `csv_rows()`, `console_summary()`, `dashboard_section()` now accept `pid_map` kwarg
- **Gaze processing** — global motion compensation for camera jitter, deterministic left-to-right track-ID assignment, improved re-ID with histogram-weighted matching
- **GUI gaze tab** — horizontal/vertical splitter layout, backend selection with per-backend config panels, device selector, settings reorganization, preset system
- **GUI project tab** — complete rebuild with pipeline YAML loader, participants table, metadata editor, conditions support
- **CSV output** — grouped tracker sections (Dyadic Interactions, Individual Gaze Behavior, Group Dynamics), project mode columns
- **Dashboard output** — `finalize_video()` method, lite overlay mode, configurable element visibility
- **Geometry utils** — `bbox_diagonal()`, trig caching in `ray_hits_cone()`, squared-distance optimizations

### Fixed
- Heatmap output filepath handling for project structure
- Snap hysteresis tracker consistency
- Forward gaze dead zone producing errant rays near pitch/yaw zero
- Face re-ID resilience to camera movement (grace period + global motion compensation)
- Duplicate "face" label on video output

### Breaking Changes
- `--gaze-model` / `--gaze-arch` / `--gaze-dataset` CLI flags renamed to `--mgaze-*` prefix
- `GazeConfig.adaptive_ray` type changed from `bool` to `str`
- `GazeConfig.adaptive_snap_mode` removed (replaced by `snap_bbox_scale` and scoring weights)
- `GazeConfig.ja_conf_gate` renamed to `hit_conf_gate`
- MGaze plugin path changed from `Plugins/GazeTracking/MGaze/` to `ms/GazeTracking/Backends/MGaze/`
- `dashboard_section()`, `csv_rows()`, `console_summary()` signatures changed (added `pid_map` kwarg)
- Phenomena tracker `__init__` no longer returns separate `ja_tracker` — JA unified into tracker list

## [0.2.0-beta] - 2026-04-01

### Added
- Initial public beta release
- Multi-person gaze tracking pipeline (Detection -> Gaze -> Phenomena -> Data)
- Plugin architecture (Gaze backends, Object Detection, Phenomena, Data Collection)
- PyQt6 GUI with Gaze Tracker, Visual Prompt Builder, and Project tabs
- 8 built-in phenomena detectors (joint attention, mutual gaze, social referencing, gaze following, gaze aversion, scanpath analysis, gaze leadership, attention span)
- MGaze and Gazelle gaze estimation backends
- YOLOE-based object detection with visual prompts
- Project mode for batch video processing
- YAML pipeline configuration
- CSV, heatmap, and video overlay outputs
- AGPL-3.0 license
