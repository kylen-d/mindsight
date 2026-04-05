# Gaze Aversion

## What It Is

Gaze aversion occurs when a participant consistently fails to look at a visible, salient object for an extended period. It captures sustained avoidance of specific objects or regions, distinguishing deliberate or involuntary avoidance from momentary inattention by requiring the absence to persist across a configurable frame window.

## Research Context

Gaze aversion is studied extensively in social anxiety research, autism spectrum disorder assessments, phobia studies, and avoidance behavior paradigms. Measuring what participants do not look at can be as informative as measuring what they do look at, revealing discomfort, fear, or disinterest toward specific stimuli.

## How MindSight Detects It

The algorithm tracks consecutive non-looking frames for each participant-object pair:

1. For each `(face_idx, object_class)` pair every frame:
   - If the participant **is** looking at the object, reset the consecutive no-look counter to 0.
   - If the participant is **not** looking, increment the counter.
2. When the counter reaches or exceeds `aversion_window` frames, flag the pair as an active aversion.
3. Only objects detected above `aversion_conf` confidence threshold are considered, filtering out low-confidence detections.
4. Object class names (not per-frame indices) are used to survive object-index churn across frames.

```mermaid
stateDiagram-v2
    [*] --> Looking
    Looking --> NotLooking : Participant stops looking
    NotLooking --> Looking : Participant looks at object\n(counter reset to 0)
    NotLooking --> NotLooking : Still not looking\n(counter increments)
    NotLooking --> AversionFlagged : counter >= aversion_window
    AversionFlagged --> Looking : Participant looks at object\n(counter reset to 0)
    AversionFlagged --> AversionFlagged : Still not looking
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--gaze-aversion` | flag | disabled | Enable gaze aversion detection |
| `--aversion-window` | int | 60 | Number of consecutive non-looking frames required to flag aversion |
| `--aversion-conf` | float | 0.5 | Minimum object detection confidence to consider the object as present |

## Output

**CSV** (`gaze_aversion`): Each row contains `participant`, `object`, and `frames_active`.

**Dashboard**: A "GAZE AVERSION" panel displays active aversions such as `P0 avoids knife`.

**Console**: Reports active aversion events inline.

**Time-series**: Plots the count of currently active aversions over time.

## Example

```bash
python MindSight.py --source video.mp4 --gaze-aversion --aversion-window 90
```

## Related Phenomena

- [Attention Span](attention-span.md) -- complementary measure of sustained looking
- [Scanpath](scanpath.md) -- aversion manifests as absence from the fixation sequence

## Source

`ms/Phenomena/Default/gaze_aversion.py`
