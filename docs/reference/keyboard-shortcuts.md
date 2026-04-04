# Keyboard Shortcuts

## CLI Mode Controls

These keys are active in the OpenCV display window while MindSight is running.

| Key | Action |
|-----|--------|
| **Q** | Quit video or webcam playback (closes the window and ends the run) |
| **Any key** | Close a single-image result window and exit |

---

## On-Screen Overlay Legend

The annotated output frame uses the following visual conventions:

| Visual Element | Meaning |
|----------------|---------|
| Coloured arrow (per person) | Gaze ray projected from the eye/face origin in the estimated gaze direction |
| Thin coloured box around person | Person detection bounding box |
| Thick box labelled **JOINT** | Object currently under joint attention (all/quorum persons looking at it) |
| Gold box labelled **LOCKED** | Object that a person's gaze has locked onto (dwell threshold met) |
| Green-tinted ray tip | Ray endpoint was snapped to the nearest object (`adaptive_ray: snap`) |
| Dwell arc around object | Partial progress toward gaze lock-on (arc fills as dwell frames accumulate) |
| Teal circle labelled **CONVERGE** | Gaze-tip convergence point where multiple persons' rays meet within `tip_radius` pixels |
| HUD panel (top-left) | Run-time info: FPS, detection count, active phenomena, JA status, and accuracy-feature summary |
| HUD panel (right side) | Per-person gaze state: track ID / participant label, current hit target, lock status |
| Extra HUD lines | Phenomena-specific status text (e.g. mutual gaze pairs, social referencing events, gaze leader) |
