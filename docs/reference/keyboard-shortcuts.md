# Keyboard Shortcuts

This page covers the **CLI viewer window** (the OpenCV display shown while
`python MindSight.py` runs). The desktop GUI application itself defines **no** keyboard
shortcuts.

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
| Dwell arc at gaze origin | Partial progress toward gaze lock-on. The arc is centred on the person's gaze-ray **origin** (the eye/face point), not the object, and fills as dwell frames accumulate |
| Teal circle labelled **CONVERGE** | Gaze-tip convergence point where multiple persons' rays meet within `tip_radius` pixels |

## Dashboard Composite

By default MindSight composes each annotated frame into a wide **dashboard** layout
`[ left panel | video | right panel ]`, rendered with matplotlib. It is not a corner HUD
overlaid on the video -- the panels sit beside the video and widen the output frame.

| Region | Contents |
|--------|----------|
| Left panel | Run-time info: FPS, detection count, active phenomena, JA status, and accuracy-feature summary |
| Centre | The annotated video frame (gaze rays, boxes, badges, convergence markers) |
| Right panel | Per-person gaze state: track ID / participant label, current hit target(s), lock status, plus phenomena-specific lines (mutual gaze pairs, social referencing, gaze leader, etc.) |

Pass `--no-dashboard` to skip dashboard composition entirely; the window (and any saved
video) then shows only the raw annotated frame.
