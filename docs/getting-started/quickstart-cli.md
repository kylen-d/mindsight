# Quickstart: CLI

This guide walks through MindSight's command-line interface with progressive examples, from basic usage to full-featured tracking runs.

---

## 1. Basic Usage

**Webcam (live):**

```bash
python MindSight.py --source 0
```

**Video file:**

```bash
python MindSight.py --source video.mp4
```

**Single image:**

```bash
python MindSight.py --source image.jpg
```

Press **Q** to quit webcam or video playback. For images, press any key to close the window.

---

## 2. Adding Object Detection Classes

Filter detections to specific classes:

```bash
python MindSight.py --source video.mp4 --classes person knife cup
```

Or blacklist classes you want to ignore:

```bash
python MindSight.py --source video.mp4 --blacklist chair
```

---

## 3. Enabling Phenomena

Enable a single phenomenon:

```bash
python MindSight.py --source video.mp4 --joint-attention
```

Enable all phenomena at once:

```bash
python MindSight.py --source video.mp4 --all-phenomena
```

Add temporal confirmation windows to reduce false positives:

```bash
python MindSight.py --source video.mp4 \
  --joint-attention --ja-window 10 --ja-window-thresh 0.7
```

---

## 4. Saving Output

Save annotated video:

```bash
python MindSight.py --source video.mp4 --save
```

Log per-frame events to CSV:

```bash
python MindSight.py --source video.mp4 --log events.csv
```

Post-run summary CSV:

```bash
python MindSight.py --source video.mp4 --summary results.csv
```

Generate gaze heatmaps:

```bash
python MindSight.py --source video.mp4 --heatmap
```

Anonymize faces in the output video:

```bash
python MindSight.py --source video.mp4 --save --anonymize
```

Generate charts:

```bash
python MindSight.py --source video.mp4 --charts
```

---

## 5. Choosing a Gaze Backend

The default backend is **MGaze**. To use an alternative:

**L2CS:**

```bash
python MindSight.py --source video.mp4 --l2cs-model weights.pkl
```

**UniGaze:**

```bash
python MindSight.py --source video.mp4 --unigaze-model unigaze_b16_joint
```

**Gazelle:**

```bash
python MindSight.py --source video.mp4 --gazelle-model ckpt.pt
```

---

## 6. Using Visual Prompts

Supply a visual prompt file and a YOLOE model for open-vocabulary detection:

```bash
python MindSight.py --source video.mp4 \
  --vp-file prompt.vp.json --vp-model yoloe-26l-seg.pt
```

---

## 7. Tuning Gaze Parameters

**Ray length** (multiplier for the rendered gaze ray):

```bash
python MindSight.py --source video.mp4 --ray-length 1.5
```

**Adaptive snap** (snap the ray endpoint to the nearest object):

```bash
python MindSight.py --source video.mp4 --adaptive-ray snap --snap-dist 200
```

**Gaze lock** (hold gaze target for a dwell duration):

```bash
python MindSight.py --source video.mp4 --gaze-lock --dwell-frames 20
```

**Gaze cone** (widen the gaze hit-test angle, in degrees):

```bash
python MindSight.py --source video.mp4 --gaze-cone 5.0
```

---

## 8. Pipeline Configs

Load a full pipeline configuration from a YAML file:

```bash
python MindSight.py --pipeline my_pipeline.yaml
```

Pipeline configs let you define sources, detection settings, phenomena, gaze parameters, and outputs in a single file. See [Project Mode](../user-guide/project-mode.md) for details.

---

## 9. Full Example

A complete command combining several features:

```bash
python MindSight.py --source video.mp4 \
  --classes person knife cup \
  --joint-attention --ja-window 10 \
  --mutual-gaze --social-ref \
  --adaptive-ray snap --snap-dist 200 \
  --save --summary results.csv --heatmap
```

<!-- screenshot: terminal output during tracking -->
