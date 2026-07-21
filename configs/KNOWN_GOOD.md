# Known-Good Parameters

Living source-of-truth for empirically validated parameter values. When tuning
work lands on a good value, record it here **with the footage it was validated
on and the date** -- this file is what pipeline presets and the docs should be
generated from, and it survives refactors that move the code around.

Conventions: GUI label first, then CLI flag / YAML key. Values validated on
classroom-style dyadic footage unless noted.

---

## Gaze-LLE Blend (primary operating mode)

Validated 2026-07-05 on `test_data/trimmed.mp4` (869 frames, 2 participants,
MobileGaze resnet50 ONNX + gazelle_dinov2_vitb14, interval 10). Result:
72 inferences fired, 405 hit events (was 9 before calibration), output ray
length tracks the Gaze-LLE reach (median ratio 1.03) with ~1-2 px/frame
length jitter.

**Gaze-target engine (updated 2026-07-18, W4A user ruling):**
`rf_gazelle_model: gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx`
-- the DINOv3-distilled pico ONNX (PINTO0309/gazelle-dinov3, 16 MB,
in/out head, onnxruntime CPU). Validated on the 87 hand-labeled frames of
the same footage: 63.9 px mean / 71% hit rate vs 70.3 / 66% for the torch
`gazelle_dinov2_vitb14` engine at the same amortized cost (~10 ms/frame
gazelle bucket at interval 10), with both participants near-balanced.
Combined with the `--rf-len-gain 1.10` default (flipped the same day):
58.8 px mean / 75% hit rate. Opt-in quality tiers (manifest weights):
ViT tiny-plus 59.8/78%, ViT-S/16 57.3/80% (56.3 with gain -- the measured
ceiling); their static single-face exports ride the Apple GPU via
`--device mps`. The torch checkpoint remains the fallback engine.

| Parameter | Value | Flag / notes |
|---|---|---|
| Inference interval / min call gap | 25 | user ruling 2026-07-09 (pre-rewrite recommendation; 2026-07-05 ruling was 10, widget default 30) |
| Direction responsiveness | 0.5 | `--dir-beta` (default) |
| Length responsiveness | 0.3 | `--len-beta` (default) |
| **Length hold** | **5.0 s** | `--len-hold-tau`. Length persists at the LLE-latched reach and decays to the pitch/yaw baseline on this timescale. Direction reverts fast on its own; do NOT try to make length "responsive" -- holding reach is the point. |
| Fixation v-threshold | 0.04 rad/frame | `--fixation-v-threshold` (code default since 2026-07-05) |
| Fixation d-threshold | 0.15 rad | `--fixation-d-threshold` (code default since 2026-07-05) |

### Internal constants (code, not flags) -- calibration rationale

- `PY_CONF_FLOOR = 0.05` (`inference_scheduler.py`). MobileGaze softmax-peak
  confidence spans ~0.08-0.29 (median ~0.14) on real footage -- a 90-bin
  softmax classifier structurally cannot peak near 1.0. The original 0.5
  floor rejected 100% of observations and Gaze-LLE never fired at all.
  If a new per-face backend is added, check its confidence scale against
  this floor FIRST.
- `P_ACCEPT = 0.6` (`inference_scheduler.py`). At 0.7 only ~38% of frames
  were inference-eligible; 0.6 admits softer fixations (user-validated on
  footage, 2026-07-05).

## Per-face gaze backend

- **MobileGaze resnet50, device-switching** (user ruling 2026-07-09, revising
  the resnet34 pick from the same day): the preset says `mgaze_model: resnet50`
  -- an extensionless family name that resolves to `resnet50.pt` on NVIDIA/CUDA
  systems and `resnet50_gaze.onnx` on Macs/CPU (cookbook, paper 7.3). Lighter
  builds live in `configs/pipeline_low_power.yaml` (mobileone_s0, unvalidated
  throughput profile for weak hardware). The Models tab tags weights that are
  optimal for the current machine. Confidence scale ~0.08-0.29; see floor
  note above.

## Object detection

From cookbook testing on the primary validation study footage (2026-05):

| Parameter | Value | Notes |
|---|---|---|
| Model (YOLO mode) | `yolov8n.pt` | user ruling 2026-07-09: fast small default; VP-prompted project runs use vp_model |
| VP model | `yoloe-26l-seg.pt` (or `yoloe-v8l/-v8m`) | v8 outperformed v26 variants on the validation footage |
| Conf | 0.20-0.30 | with Merge Overlaps on |
| Merge Overlaps | on, threshold 0.50-0.60 | strategy `filter` or `dynamic` |
| VP resolution | match video resolution | YOLOE encodes pixel-size, not semantics |

## Ray geometry

| Parameter | Value | Notes |
|---|---|---|
| Ray length | 1.3 | KG_Standard 2026-07-09 (validated range 1.25-1.50); blend corrects reach dynamically |
| Smooth snap | all, alpha 0.9 | KG_Standard 2026-07-09 |
| JA temporal window | 0 (off) | KG_Standard 2026-07-09 |
| Gaze cone | ~5.0 deg | study-dependent |
| ReID grace | 4-5 s | re-identification window |
| Gaze tips | on, radius 70 | tip convergence counts as JA; tracks phenomena in lieu of object detections (2026-07-09) |
| Detect-extend scope | both | applies when an extend distance is set (2026-07-09) |
| Forward-gaze threshold | 13.0 deg | schema default 5.0; validated 2026-07-09 |

---

## Tuning queue (experiments in flight -- record results above when done)

(empty)

## Future work

- **Saccade safeguard (user, 2026-07-05).** With the tolerant fixation
  thresholds (v 0.04 / d 0.15), quick-but-distinct head-direction/gaze shifts
  risk being absorbed as "still fixating" -- the belief anchor can briefly
  drag direction behind a genuine rapid retarget. Build an explicit safeguard:
  detect large fast PY excursions (saccade detector) and immediately release
  the belief anchor / drop trust for that track, independent of the tolerant
  fixation criterion. Candidates: velocity spike gate above a hard ceiling,
  or dispersion jump detection over a 2-3 frame window.
- **Direction escalation path** (if drift returns): `--dir-trust-floor` knob
  (direction weight = max(trust, floor)); further: per-track PY-vs-belief bias
  correction learned during high-trust periods (design work, not a patch).

## Performance

| Parameter | Value | Notes |
|---|---|---|
| Dashboard panels | off (no_dashboard) | throughput; saved videos show raw annotated frame (2026-07-09) |

## Changelog

- **2026-07-09 (later still)**: MobileGaze resnet34 -> resnet50 with
  device-switching family names (extensionless `mgaze_model` resolves .pt on
  CUDA, `_gaze.onnx` elsewhere); `configs/pipeline_low_power.yaml` added
  (UNVALIDATED throughput profile: mobileone_s0, detect_scale 0.75, fast,
  min_call_gap 40); weights manifest tags MGaze builds with optimal devices,
  surfaced in the Models tab.
- **2026-07-09 (later)**: KG_Standard -- the user's first full-fidelity GUI
  export -- becomes the preset, with four review rulings: merge overlaps stay
  on/dynamic/0.55; blend cadence 25; detector yolov8n (YOLO mode); ReID grace
  stays 4.5 s. New from the export: ray 1.3, smooth snap all/0.9, MobileGaze
  resnet34 onnx, JA window 0. The exporter previously wrote only non-default
  keys and silently omitted hand-widget settings (merge, device) -- both fixed
  the same day.
- **2026-07-09**: User-validated TESTGOOD export merged into the shipped
  preset (layered on top of existing values): gaze tips on (radius 70),
  detect-extend scope both, forward-gaze threshold 13.0, dashboard panels
  off, all phenomena on. Existing detection/geometry/blend values unchanged.
- **2026-07-05 (later)**: Fixation thresholds v 0.02->0.04, d 0.10->0.15 made
  code defaults after A/B test on trimmed.mp4 (user-validated visually).
  Steering eligibility 38%->70% of frames, median direction trust 0.52->0.81,
  hit events 953->1037, length stability unchanged. Saccade-safeguard future
  work noted.
- **2026-07-05**: Created. Gaze-LLE blend calibration after scheduler redesign:
  PY_CONF_FLOOR 0.5 -> 0.05 (bug: blend silently inert), len_hold_tau added
  (bug: length bounced with per-frame trust), P_ACCEPT 0.7 -> 0.6 (tuning).
  Cookbook values (2026-05) imported for detection/geometry sections.
