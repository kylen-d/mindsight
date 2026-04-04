# Phenomena Tracking

## What is Phenomena Tracking

MindSight can detect various gaze-based behavioral phenomena beyond simple gaze-object intersection. Phenomena tracking analyzes patterns in gaze data over time to identify meaningful social and cognitive behaviors -- such as two people looking at the same object simultaneously, or one person's gaze consistently leading another's.

## Enabling Phenomena

Enable individual phenomena using their CLI flags:

```bash
python MindSight.py --source video.mp4 --joint-attention --mutual-gaze
```

Or enable all phenomena at once:

```bash
python MindSight.py --source video.mp4 --all-phenomena
```

In the GUI, each phenomenon has a checkbox in the Phenomena Panel on the Gaze Tracker tab. Configurable parameters appear when a phenomenon is enabled.

In Project Mode, enable phenomena in the `pipeline.yaml`:

```yaml
phenomena:
  joint_attention: true
  mutual_gaze: true
  gaze_following: true
```

## Phenomena Overview

| Phenomenon | CLI Flag | Description | Details |
|---|---|---|---|
| Joint Attention | `--joint-attention` | Two or more people looking at the same object simultaneously | [Joint Attention](../phenomena/joint-attention.md) |
| Mutual Gaze | `--mutual-gaze` | Two people looking directly at each other | [Mutual Gaze](../phenomena/mutual-gaze.md) |
| Gaze Following | `--gaze-following` | One person shifts gaze to match where another is looking | [Gaze Following](../phenomena/gaze-following.md) |
| Gaze Leadership | `--gaze-leadership` | One person's gaze consistently directs others' attention | [Gaze Leadership](../phenomena/gaze-leadership.md) |
| Social Referencing | `--social-referencing` | Looking at another person's face after encountering a novel object | [Social Referencing](../phenomena/social-referencing.md) |
| Gaze Aversion | `--gaze-aversion` | Deliberately looking away from a person or object | [Gaze Aversion](../phenomena/gaze-aversion.md) |
| Attention Span | `--attention-span` | Duration of sustained gaze on a single target | [Attention Span](../phenomena/attention-span.md) |
| Scanpath | `--scanpath` | Sequential pattern of gaze fixations across objects | [Scanpath](../phenomena/scanpath.md) |

## How Phenomena Work

Each phenomenon is implemented as a tracker that receives per-frame data from the main pipeline. On every frame, the tracker receives:

- Gaze positions and ray directions for each detected person
- Bounding boxes and class labels for detected objects
- Hit events (which gaze rays intersect which objects)
- Temporal context from previous frames

The tracker applies its specific detection logic and emits events when the phenomenon is observed. All built-in phenomena trackers now implement the `dashboard_data()` protocol, providing structured metric dictionaries for live visualization. Results appear in three places:

- **Dashboard** -- Real-time charts during tracking
- **CSV output** -- Per-frame and summary data in the output CSV files
- **Console** -- Log messages when events are detected

## Combining Phenomena

Multiple phenomena can run simultaneously with no conflicts. Each tracker operates independently on the shared per-frame data.

Some phenomena interact or benefit from each other:

- **Gaze Leadership** depends on Joint Attention data to determine who initiates shared attention. Enable Joint Attention alongside Gaze Leadership for best results.
- **Social Referencing** uses gaze-to-face detection, which overlaps with Mutual Gaze processing.
- Running `--all-phenomena` ensures all dependencies are satisfied automatically.

## Custom Phenomena

You can create your own phenomena trackers as plugins. See:

- [Plugin System](../developer/plugin-system.md) -- Architecture overview
- [Writing a Plugin](../developer/writing-a-plugin.md) -- Step-by-step guide to creating a custom phenomenon
