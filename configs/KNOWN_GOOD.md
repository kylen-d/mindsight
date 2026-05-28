# Known-Good Parameters

Living source-of-truth for empirically validated parameter values. When tuning
work lands on a good value, record it here **with the footage it was validated
on and the date** -- this file is what pipeline presets and the docs should be
generated from, and it survives refactors that move the code around.

Conventions: GUI label first, then CLI flag / YAML key. Values validated on
KITCO-style footage unless noted.

---

## Gaze-LLE Blend (primary operating mode)

Validated 2026-07-05 on `test_data/trimmed.mp4` (869 frames, 2 participants,
MobileGaze resnet50 ONNX + gazelle_dinov2_vitb14, interval 10). Result:
72 inferences fired, 405 hit events (was 9 before calibration), output ray
length tracks the Gaze-LLE reach (median ratio 1.03) with ~1-2 px/frame
length jitter.

| Parameter | Value | Flag / notes |
|---|---|---|
| Inference interval | 10 | `--rf-gazelle-interval`; cookbook range 10-15 |
| Min call gap | 30 frames | `--min-call-gap` (default) |
| Direction responsiveness | 0.5 | `--dir-beta` (default) |
| Length responsiveness | 0.3 | `--len-beta` (default) |
| **Length hold** | **5.0 s** | `--len-hold-tau`. Length persists at the LLE-latched reach and decays to the pitch/yaw baseline on this timescale. Direction reverts fast on its own; do NOT try to make length "responsive" -- holding reach is the point. |
| Fixation v-threshold | 0.02 rad/frame | `--fixation-v-threshold` (default) |
| Fixation d-threshold | 0.10 rad | `--fixation-d-threshold` (default) |

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

- **MobileGaze resnet50** -- `.pt` on CUDA, `.onnx` on Apple Silicon/CPU
  (cookbook, paper 7.3). Confidence scale ~0.08-0.29; see floor note above.

## Object detection (KITCO)

From cookbook testing on KITCO 3 footage (2026-05):

| Parameter | Value | Notes |
|---|---|---|
| Model | `yoloe-v8l-seg.pt` or `yoloe-v8m-seg.pt` | v8 outperformed v26 variants on KITCO |
| Conf | 0.20-0.30 | with Merge Overlaps on |
| Merge Overlaps | on, threshold 0.50-0.60 | strategy `filter` or `dynamic` |
| VP resolution | match video resolution | YOLOE encodes pixel-size, not semantics |

## Ray geometry (KITCO)

| Parameter | Value | Notes |
|---|---|---|
| Ray length | 1.25-1.50 | base; blend corrects reach dynamically |
| Gaze cone | ~5.0 deg | study-dependent |
| ReID grace | 4-5 s | re-identification window |

---

## Tuning queue (experiments in flight -- record results above when done)

- **Direction drift between inferences (one participant).** Symptom: direction
  accuracy slips back to the (biased) per-face vector between Gaze-LLE
  inferences; participant-specific. Experiment: raise fixation tolerance so
  slight shifting still counts as fixating and belief keeps steering --
  GUI: Gazelle Blend > Advanced > "Fixation v-threshold" 0.02 -> 0.03-0.04
  and "Fixation d-threshold" 0.10 -> 0.15 (flags `--fixation-v-threshold`,
  `--fixation-d-threshold`). Escalation if insufficient: `--dir-trust-floor`
  knob (direction weight = max(trust, floor)); further: per-track PY-vs-belief
  bias correction learned during high-trust periods (design work, not a patch).

## Changelog

- **2026-07-05**: Created. Gaze-LLE blend calibration after scheduler redesign:
  PY_CONF_FLOOR 0.5 -> 0.05 (bug: blend silently inert), len_hold_tau added
  (bug: length bounced with per-frame trust), P_ACCEPT 0.7 -> 0.6 (tuning).
  Cookbook values (2026-05) imported for detection/geometry sections.
