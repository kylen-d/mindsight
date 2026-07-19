# Changelog

## [Unreleased]

### Added (W4A DINOv3-generation gaze-target weights in the manifest)
- **Five PINTO0309/gazelle-dinov3 ONNX exports are now managed weights**
  (checksummed manifest entries; rows appear in the GUI Models tab with
  license provenance, installable with `mindsight-weights` or one click):
  the 16 MB HGNetV2-pico model (640 px, in/out head, dynamic face count
  -- v1.1 eval 63.9 px mean / 71% hit rate vs 70.3 / 66% for the torch
  DINOv2 ViT-B/14 engine, on CPU at the same amortized cost), and two
  opt-in quality tiers: ViT tiny-plus (52 MB, 59.8 px / 78%) and
  ViT-S/16 (100 MB, 57.3 px / 80% -- the measured ceiling). The ViT
  tiers ship in both dynamic-face and single-face static exports; the
  static siblings are the shape Apple-GPU CoreML accepts, so
  `--device mps` accelerates them 2-3x over CPU. Use any of them via
  `--rf-gazelle-model <filename>.onnx`. Licensing: pico and tiny-plus
  are MIT releases over Apache-2.0 backbones distilled from DINOv3
  outputs (no Meta weights embedded); ViT-S/16 embeds the Meta DINOv3
  backbone and so additionally carries the DINOv3 License (commercial
  use permitted; passthrough on redistribution -- see
  THIRD_PARTY_LICENSES.md). All variants are trained on GazeFollow /
  VideoAttentionTarget, the same research-dataset provenance as the
  existing Gaze-LLE and MobileGaze weights, and carry it as a manifest
  `license_note`. Defaults unchanged in this step.

### Added (W3Z ONNX gaze-target engine)
- **`--rf-gazelle-model` now accepts `.onnx` checkpoints** and runs them
  on onnxruntime (CPU) as the blend-path gaze-target engine -- built for
  the PINTO0309 *gazelle-dinov3* exports (Gaze-LLE successors on DINOv3
  and distilled HGNetV2 backbones, MIT). Same heatmap/in-out contract as
  the torch engine, so the scheduler, blender, and length channel are
  untouched; sub-64 heatmaps are upsampled to the blend grid. Measured
  here: the 12 MB atto variant runs ~11 ms per call for two faces on CPU
  vs ~88 ms for the torch DINOv2-vitb14 engine on Apple MPS. The global
  `--device` selects the execution provider (cuda -> CUDA EP on NVIDIA
  machines with onnxruntime-gpu installed, mps -> Apple-GPU CoreML for
  the ViT-backbone static single-face exports, which the engine loops
  per face transparently -- measured 2-3x over CPU; cpu -> CPU), always
  falling back to CPU with a plain note when a provider is unavailable
  or rejects the model. No default change and no weights ship yet --
  variant choice awaits eval results.

### Added (W3Z length accuracy knobs)
- `--rf-len-gain F`: scale the blend ray-length target. The v1.1 eval
  decomposition found a systematic under-reach -- 84% of rays measured
  too short (197 px predicted vs 233 px true along-ray), and a global
  1.10 gain recovered ~5 px mean / ~8 px median offline. Applied before
  smoothing so snap and hit detection see the corrected reach.
  **Default 1.0 (off)** pending in-pipeline eval.
- `--rf-endpoint-extract {centroid,topp}`: how a Gaze-LLE heatmap
  becomes the ray-length endpoint. `topp` takes the mass centroid of
  only the top-50%-mass cells, so diffuse or multi-modal heatmaps stop
  dragging the endpoint toward the origin (a suspected mechanism of the
  under-reach). **Default `centroid` (historical)**. Both knobs also
  appear in Inference Settings. Note: adds two `rayforming` schema
  fields, so pre-existing resume ledgers report a config-hash change
  and reprocess once (pre-release; same caveat as earlier adds).

### Added (W3Z overlay theme)
- `--overlay-theme {classic,mindsight}`: restyle the annotated-frame
  overlays. `mindsight` uses the brand palette sampled from the logo and
  app icon -- deep-indigo ink label tabs with coloured borders instead
  of solid saturated fills, the logo's magenta and jade as hero accents
  (participants, lock, convergence), warm gold for joint attention and
  dwell, and indigo dashboard panels. Geometry is identical in both
  themes; this is purely cosmetic and analysis outputs are unaffected.
  Also selectable as "Overlay theme" in Inference Settings and the
  tuning panel. **Default `classic`** (the historical look, byte-pinned
  by the regression goldens).

### Added (W3Z VP Builder Suggest mode)
- **Click, don't draw: the VP Builder can now propose object boxes.** A
  new "Suggest mode" toggle under the canvas segments the region under
  your click (FastSAM point-prompt) and shows up to four dashed box
  proposals, most specific first -- click one to accept it into the
  selected class, exactly as if hand-drawn. The `.vp.json` format is
  unchanged, so nothing downstream moves. Needs the new FastSAM-s
  weight (24 MB, AGPL-3.0 -- the same license class as the Ultralytics
  package), added to the weights manifest as an optional entry with a
  Models-tab row; the toggle explains in plain English when the weight
  is not yet downloaded. Suggestion inference runs off the GUI thread
  (~0.4 s per click on Apple silicon).

### Fixed (W3Z installers)
- **The one-click installers no longer need git.** The `clip` dependency
  was pinned as a `git+` URL, which made `uv sync` shell out to a git
  executable during install and fail cryptically on machines without git
  (fresh Windows lab PCs; Macs without the Xcode Command Line Tools). It
  is now pinned as the same commit's HTTPS tarball -- byte-identical
  package, no git involved, and the lockfile now carries a sha256 hash
  for it. Both installers also gained a plain-English preflight note: if
  git is missing they say so up front and name the platform remedy, in
  case a future dependency ever reintroduces the requirement.

### Added (W3Z)
- `--rf-len-slew N`: smooth ray-length transitions between length
  refreshes. When a Gaze-LLE pass re-latches an already-latched ray
  length, the latch now slews linearly toward the new value over N
  frames instead of snapping instantly (a refresh arriving mid-slew
  restarts from the current interpolated value, so the ray never
  jumps). First-ever latches still snap, and the length-hold decay
  clock resets at slew start. **Default 0 (off)** -- a same-day flip to
  5 was reverted after eyes-on review: slewing the latch while the
  length-hold decay pulls the target toward the baseline reads as
  BOUNCE rather than smoothing on real footage. The knob remains for
  experimentation; a rework that slews the effective target is
  planned. Also available as "Length slew (frames)" in Inference
  Settings. Note: adds the
  `rayforming.rf_len_slew` schema field, so pre-existing resume
  ledgers report a config-hash change and reprocess once (pre-release;
  same caveat as earlier v1.1 schema adds).

### Added (W3Y)
- `--rf-len-refresh-gap N`: a cheap Gaze-LLE length-refresh channel for
  the blend path. Every N frames one extra Gaze-LLE pass refreshes ray
  LENGTH only. On CUDA the pass runs on a persistent half-precision copy
  of the model (same checkpoint, ~179 MB extra for vitb14, genuinely
  faster there); on Apple/CPU the main engine is shared, since measured
  on MPS fp16 is no faster per call (87.1 vs 87.7 ms) -- the channel's
  value is the scheduling itself. The full-precision, fixation-gated
  corrections remain the sole authority over direction and the belief
  map, and a track that has never received a full-precision correction
  is never touched. Keeps ray reach fresh between corrections without
  extra fixation-gated fires; honors the `--rf-inout-gate` veto.
  **Default ON at N=10** (eval-validated on the 87 hand-labeled frames:
  mean gaze-endpoint error 71.3 -> 70.3 px, median 58.7 -> 57.9 px, hit
  rate 64% -> 66%, for ~+0.6 ms/frame in the Gaze-LLE bucket);
  `--rf-len-refresh-gap 0` restores the previous behavior. Blend
  regression goldens re-blessed accordingly. Note: adds the
  `rayforming.rf_len_refresh_gap` schema field, so pre-existing resume
  ledgers report a config-hash change and reprocess once (pre-release;
  same caveat as earlier v1.1 schema adds).

### Added (W3Y update notifications)
- **MindSight now notices new releases.** On launch a silent, non-blocking
  check against GitHub Releases compares the latest tag with the running
  version; when newer, a subtle status-bar chip and an About-hero line
  appear ("vX.Y available -- release notes") and a click opens the release
  page in the browser. Nothing is ever downloaded or executed
  automatically; any network failure is a silent no-op. Opt out with the
  "Check for updates on launch" toggle in About, or set
  MINDSIGHT_NO_UPDATE_CHECK=1 for frozen lab environments. A release you
  have opened is not announced again.

### Changed (W3Y study-setup redesign)
- **The Analyze Footage "Study setup" pane is gone.** Project-level setup
  (pipeline in use, project-wide participants, per-video conditions,
  output root, Save project.yaml) moved to the **Projects tab** overview,
  edited before running; saving there resyncs an open Analyze Footage
  view. Per-run metadata is edited from the runs table ("Edit run...", as
  before). Project batch runs now read the SAVED `project.yaml` -- what
  you saved is what runs.
- **The Inference Settings dialog is the single processing authority for
  every launch.** The pane's "Anonymize Footage" checkbox (which silently
  overrode the dialog, and forced anonymize OFF for project runs) is
  gone; anonymize comes from Inference Settings in all modes, like every
  other processing option.

### Changed (W3Y GUI)
- **File pickers now start where their files live and offer the known
  candidates.** Model pickers (YOLO / YOLOE / MobileGaze / Gaze-LLE, in
  the Gaze Tuning tab and the Inference Settings dialog) are editable
  dropdowns listing the weights already in `Weights/<backend>/`, with
  Browse... opening there; VP-file dialogs remember the last visual-prompt
  folder (VP Builder saves record it); project-folder dialogs open beside
  the most recent project.
- The Models tab has a **License** column: the license id from the weights
  manifest, plus an honest usage note where the id alone would mislead --
  MobileGaze weights are Gaze360-trained (research use only) and Gaze-LLE
  checkpoints carry their training-set provenance. THIRD_PARTY_LICENSES
  documents the same.
- The About tab hero now centres the MindSight app icon above the program
  name (the wordmark, which repeats the name, is only a fallback), and
  installed builds render it too.

### Fixed (W3Y)
- **"Optimal for this device" no longer lies silently on NVIDIA machines
  with a CPU-only torch install.** PyPI's Windows torch wheels carry no
  CUDA support, so a CUDA lab machine reported `cuda.is_available() ==
  False` and the Models tab marked the ONNX weights optimal. The device
  decision was correct given that torch; the install problem is now
  detected (CPU-only build + `nvidia-smi` reporting a GPU) and surfaced
  loudly in the Models tab and project preflight, with the CUDA-index
  reinstall command as the remedy.
- **Study-setup values no longer override the Inference Settings dialog in
  Video File / Camera mode.** The Analyze Footage study-setup anonymize
  checkbox (a project-pane control) silently overrode the dialog's
  per-run settings for quick runs; quick modes now launch from the
  Inference Settings store untouched. Project-mode launches keep the
  checkbox as their single anonymize control.

### Fixed (W3X MPIIGaze substrate)
- `--mgaze-dataset mpiigaze` now works with ONNX models: the ONNX decode
  previously hardcoded gaze360 bin geometry (90 x 4° - 180°), so an
  MPIIGaze-trained export (28 x 3° - 42°, trainable with the vendored
  gaze-estimation library) mis-decoded. The default remains bit-identical.
  Note: no MPIIGaze weights ship -- the public pretrained sets
  (hysts/pytorch_mpiigaze etc.) are trained on CC-BY-NC-SA / research-only
  data.

### Changed (v1.1 default flips -- eval-validated on 87 hand-labeled frames)
- **New defaults**: `yolo11n.pt` detector, RetinaFace `r34` face backbone,
  eye-midpoint ray origins (`--no-face-eye-origin` restores bbox centres),
  and earlier Gaze-LLE corrections for new faces (`rf_onset_samples 3`,
  `rf_onset_gap 5`). Together: mean gaze-endpoint error 74.5 -> 71.3 px,
  median 63.6 -> 58.7 px, hit rate 62% -> 64%, first corrections at frame 3
  (was 5/15), and ~42% faster per frame (the r34 face backbone runs on the
  CoreML path; the old mobile backbone silently ran on CPU).
- Regression baselines re-blessed accordingly (smoke hit counts 2/1025/711;
  new frozen blend SSIM reference). Concurrent first-launch face-weight
  downloads are now serialized (flock), fixing a race two simultaneous
  first runs could hit.
- The weights manifest now also carries `yolo11n.onnx` (official ONNX
  export, faster on CPU-bound installs) and the YOLOE-11 visual-prompt
  family (`yoloe-11s/m/l-seg`).

### Added (W3X face/tracking knobs, all default-off)
- **RetinaFace landmarks and detection scores now reach the pipeline.**
  uniface 1.1.0 returns them under keys the pipeline never read, so the
  documented eye-midpoint ray origin was silently dead (origins always the
  face-box centre -- the behavior all baselines were blessed against). A
  boundary adapter normalizes the dicts; `--face-eye-origin` opts rays
  into the true eye-midpoint origin (now the default, see the flips above).
- `--face-reid-sim`: embedding-verified track revival ("redetection") --
  a lost face can be re-identified anywhere in the frame by ArcFace cosine
  similarity, with positional revival as the fallback. Weights
  auto-download on first use; see THIRD_PARTY_LICENSES for the InsightFace
  research-use provenance note.
- `--face-model`: RetinaFace backbone selector (mnet025 ... r34); larger
  backbones detect small/distant faces better at a speed cost.

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
- `yolo11n.pt` added to the weights manifest (now the default, see the
  flips above); `scripts/eval_annotate.py` + `scripts/eval_gaze.py` give accuracy
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
