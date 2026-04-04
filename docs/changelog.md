# Changelog

## [0.3.0-beta] - 2026-04-04

### Added
- **L2CS-Net gaze backend** — dual classification heads, 3.92° MAE on MPIIGaze (vs 11° for MGaze)
- **UniGaze gaze backend** — ViT + MAE pre-training, best cross-dataset accuracy (~9.4°, non-commercial license)
- **Backend registry** — automatic discovery of gaze backends from `GazeTracking/Backends/`
- **Unified pitchyaw pipeline** — `pitchyaw_pipeline.py` shared by all pitch/yaw-based backends
- **Live matplotlib dashboard** — real-time per-tracker charts during processing (`GUI/live_dashboard.py`)
- **Dashboard bridge** — thread-safe GUI-to-dashboard data flow (`GUI/live_dashboard_bridge.py`)
- **Post-run chart generation** — time-series charts via `DataCollection/chart_output.py` (`--charts` flag)
- **Global CSV** — cross-video summary and per-condition statistics for project mode (`DataCollection/global_csv.py`)
- **Matplotlib dashboard renderer** — `DataCollection/dashboard_matplotlib.py` replaces OpenCV drawing
- **Face anonymization** — `--anonymize blur|black` with configurable padding and temporal smoothing
- **Auxiliary video streams** — per-participant secondary cameras (eye-tracking, FPV) via `AuxStreamConfig`
- **Participant ID mapping** — custom labels via `pid_map` in `project.yaml` and `participant_ids.py`
- **ProjectConfig / ProjectOutputConfig** — study metadata dataclasses for project mode
- **Example project template** — `Projects/ExampleProject/` with `project.yaml`
- **CollapsibleGroupBox widget** — expandable/collapsible GUI sections
- **GazelleSnap plugin** — snap-augmented Gazelle gaze backend
- **GazeBoost plugin** — gaze-informed object detection boost using pitchyaw pipeline
- **Device auto-detection** — `utils/device.py` for CUDA/MPS/CPU hardware selection
- **Plugin protocol methods** — `dashboard_data()`, `latest_metric()`, `latest_metrics()`, `dashboard_widget()`, `generate_charts()`
- **Performance flags** — `--fast-mode`, `--skip-phenomena N`, `--lite-overlay`, `--no-dashboard`, `--profile`
- **Gaze convergence tips** — `--gaze-tips` + `--tip-radius` for multi-person gaze convergence visualization

### Changed
- **MGaze relocated** from `Plugins/GazeTracking/MGaze/` to `GazeTracking/Backends/MGaze/`
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
- MGaze plugin path changed from `Plugins/GazeTracking/MGaze/` to `GazeTracking/Backends/MGaze/`
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
- GPLv3 license
