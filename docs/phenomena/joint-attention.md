# Joint Attention

**Source:** `ms/Phenomena/Default/joint_attention.py`

## What It Is

Joint attention occurs when multiple participants simultaneously fixate on the same object. It is one of the most widely studied gaze phenomena because it reflects shared intentionality -- the ability of two or more people to coordinate their visual attention on a common target. MindSight detects joint attention on a per-object, per-frame basis and reports running percentages over the session.

## Research Context

Joint attention is a foundational milestone in early cognitive development and is routinely assessed in developmental psychology and ASD (Autism Spectrum Disorder) screening. Researchers study it to understand how infants learn to share attention with caregivers, how social cognition emerges, and how breakdowns in joint attention relate to developmental differences. It is also relevant in collaborative work, education, and human-robot interaction research.

## How MindSight Detects It

The detection algorithm has three stages:

1. **Raw JA detection.** For every detected object in the current frame, count how many faces have a gaze ray that intersects that object's bounding region. If the count is greater than or equal to `ceil(quorum * n_faces)` and there are at least 2 faces in the scene, the object is flagged as under raw joint attention.

2. **Temporal filtering (optional).** When `--ja-window` is set to a value greater than 0, a sliding window of that many frames is maintained per object. An object's joint attention is confirmed only if it appeared in the raw JA set for at least `ja_window_thresh` fraction of the frames in the window. This smooths out momentary flickers.

3. **Running percentage.** For each object, MindSight tracks `confirmed_frames / total_frames` as a running JA percentage over the entire session.

```mermaid
stateDiagram-v2
    [*] --> NoJA
    NoJA --> RawJA : gaze count >= quorum threshold
    RawJA --> WindowFilling : ja_window > 0
    RawJA --> Confirmed : ja_window == 0
    WindowFilling --> Confirmed : fraction >= ja_window_thresh
    WindowFilling --> NoJA : fraction < ja_window_thresh
    Confirmed --> NoJA : drops below threshold
    Confirmed --> Confirmed : sustained
```

## Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--joint-attention` | bool | `False` | Enable joint attention tracking |
| `--ja-window` | int | `0` | Sliding window size in frames for temporal filtering. 0 disables the filter. |
| `--ja-window-thresh` | float | `0.70` | Fraction of window frames an object must appear in raw JA to be confirmed |
| `--ja-quorum` | float | `1.0` | Fraction of visible faces that must fixate on an object for raw JA (1.0 = all faces) |
| `--all-phenomena` | bool | `False` | Enable all phenomena including joint attention |

## Output

**CSV** (`joint_attention` section):
- `confirmed_frames` -- number of frames where the object had confirmed JA
- `total_frames` -- total frames processed
- `pct` -- confirmed_frames / total_frames as a percentage

**Dashboard:**
A "JOINT ATTENTION" panel listing each object currently under joint attention alongside its running JA percentage.

**Console:**
Raw frame counts and confirmed frame counts per object printed at the end of the session.

## Example

```bash
python MindSight.py --source video.mp4 --joint-attention --ja-window 10
```

This enables joint attention detection with a 10-frame sliding window. An object must appear in raw JA in at least 70% of the window (the default threshold) to be confirmed.

## Related Phenomena

- [Gaze Leadership](gaze-leadership.md) -- identifies who initiates joint attention episodes
- [Gaze Following](gaze-following.md) -- captures sequential (rather than simultaneous) shifts of attention toward a shared target
