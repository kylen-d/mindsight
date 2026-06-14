# Gaze Plugin Tutorial

> **See also:** [Phenomena Plugin Tutorial](phenomena-plugin-tutorial.md) | [Object Detection Plugin Tutorial](object-detection-plugin-tutorial.md) | [Data Collection Plugin Tutorial](data-collection-plugin-tutorial.md)

This tutorial covers all three gaze plugin patterns by walking through real backends that ship with MindSight:

- **[Part A — Per-face mode:](#part-a-per-face-backend-mgaze)** The MGaze backend (`mindsight/GazeTracking/Backends/MGaze/MGaze_Tracking.py`), which crops each face and estimates pitch/yaw angles independently.
- **[Part B — Scene-level mode:](#part-b-scene-level-backend-gazelle)** The Gazelle backend (`Plugins/GazeTracking/Gazelle/gazelle_backend.py`), which processes the full frame and all faces in a single DINOv2 forward pass.
- **[Part C — Composite / processing augmentation:](#part-c-composite-backend-temporarily-removed)** Previously demonstrated via the GazelleSnap plugin, which was removed in v0.8. A replacement composite-plugin example is TODO.

---

## Per-Face vs Scene-Level: When to Use Which

| Aspect | Per-face (`mode="per_face"`) | Scene-level (`mode="scene"`) |
|--------|-----|------|
| Core method | `estimate(face_bgr)` → `(pitch, yaw, conf)` | `estimate_frame(frame, bboxes)` → `[(xy, conf)]` |
| Input | Single cropped face image | Full frame + all face bounding boxes |
| Gaze format | Pitch/yaw angles (radians) | Pixel coordinates in the original frame |
| GPU passes | One per face | One for all faces |
| Ray construction | Handled by `run_pitchyaw_pipeline` | Handled by the gaze coordinator's default scene handler |
| Best for | Lightweight models, CPU, ONNX inference | Heavy models, GPU batch processing, heatmap outputs |

Choose your mode based on what your model produces. If it outputs pitch/yaw angles from a face crop, use per-face. If it takes the full scene and outputs gaze target coordinates, use scene-level.

---

# Part A: Per-Face Backend (MGaze)

The MGaze plugin is MindSight's default gaze backend. It supports both ONNX and PyTorch inference, wrapping the vendored `gaze-estimation` library. It demonstrates the per-face pattern where `estimate()` receives a single cropped face and returns pitch/yaw angles.

**Source:** `mindsight/GazeTracking/Backends/MGaze/MGaze_Tracking.py`

---

## A1. File Structure

```
mindsight/GazeTracking/Backends/MGaze/
├── __init__.py
├── MGaze_Tracking.py       # PLUGIN_CLASS = MGazePlugin
├── MGaze_Config.py         # DEFAULT_ONNX_MODEL, ARCH_CHOICES, DATA_CONFIG
└── gaze-estimation/        # Vendored gaze-estimation library
    ├── weights/
    │   └── mobileone_s0_gaze.onnx  # Default shipped model
    ├── models/
    │   ├── resnet.py
    │   ├── mobilenet.py
    │   └── mobileone.py
    ├── onnx_inference.py    # GazeEstimationONNX base class
    └── config.py
```

!!! note
    MGaze is a built-in core backend (resolved directly by `create_gaze_engine` since v1.0, not registered as a plugin); its code lives under `mindsight/GazeTracking/Backends/MGaze/`. The gaze registry scans `Plugins/GazeTracking/` only, for external plugins.

---

## A2. Configuration Module

`MGaze_Config.py` centralises model paths and dataset parameters:

```python
DEFAULT_ONNX_MODEL = str(
    Path(__file__).parent / "gaze-estimation" / "weights" / "mobileone_s0_gaze.onnx"
)

ARCH_CHOICES = [
    "resnet18", "resnet34", "resnet50", "mobilenetv2",
    "mobileone_s0", "mobileone_s1", "mobileone_s2",
    "mobileone_s3", "mobileone_s4",
]

DATA_CONFIG = {
    "gaze360":  {"bins": 90, "binwidth": 4, "angle": 180},
    "mpiigaze": {"bins": 28, "binwidth": 3, "angle": 42},
}
```

The `DATA_CONFIG` controls bin-based regression: gaze direction is predicted as a probability distribution over `bins` discrete bins, each `binwidth` degrees wide, spanning `±angle` degrees.

---

## A3. The Estimation Engines

MGaze wraps two interchangeable estimation engines behind the same `estimate(face_bgr)` interface.

### PyTorch Engine

```python
class GazeEstimationTorch:
    def __init__(self, weight_path, arch, dataset="gaze360", device="auto"):
        cfg = DATA_CONFIG[dataset]
        self._bins, self._binwidth, self._angle = cfg["bins"], cfg["binwidth"], cfg["angle"]
        self.device = resolve_device(device)
        self.idx_tensor = torch.arange(self._bins, dtype=torch.float32, device=self.device)

        model = utils_gaze.helpers.get_model(arch, self._bins, inference_mode=True)
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = model.to(self.device).eval()

        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
```

The `estimate()` method:

```python
def estimate(self, face_bgr):
    t = self._tf(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        pitch_logits, yaw_logits = self.model(t)
    pp = F.softmax(pitch_logits, 1)
    yp = F.softmax(yaw_logits, 1)

    to_rad = lambda p: float(np.radians(
        (torch.sum(p * self.idx_tensor) * self._binwidth - self._angle).item()))

    conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
    return to_rad(pp), to_rad(yp), conf
```

Step-by-step:

1. **Preprocess** — Convert BGR→RGB, resize to 448×448, normalize with ImageNet stats.
2. **Forward pass** — Model outputs two sets of logits: one for pitch bins, one for yaw bins.
3. **Softmax** — Convert logits to probability distributions.
4. **Expectation** — Compute the weighted sum `Σ(probability × bin_index)` to get the predicted bin, then convert to degrees and radians.
5. **Confidence** — The peak softmax probability indicates how "certain" the model is. The `_softmax_confidence` helper maps the average peak from `[1/n_bins, 1]` onto `[0, 1]`.

### ONNX Engine

```python
class _GazeONNXWithConf(GazeEstimationONNX):
    def estimate(self, face_bgr):
        out = self.session.run(self.output_names, {"input": self.preprocess(face_bgr)})
        pitch, yaw = self.decode(out[0], out[1])
        pp, yp = self.softmax(out[0]), self.softmax(out[1])
        conf = _softmax_confidence(float(pp.max()), float(yp.max()), self._bins)
        return pitch, yaw, conf
```

Extends the vendored `GazeEstimationONNX` class with confidence scoring using the same `_softmax_confidence` formula. The `preprocess`, `decode`, and `softmax` methods are inherited from the base class.

### Confidence Scoring

Both engines share this helper:

```python
def _softmax_confidence(pitch_probs_max, yaw_probs_max, n_bins):
    uniform = 1.0 / n_bins
    return float(np.clip(
        ((pitch_probs_max + yaw_probs_max) / 2 - uniform) / (1 - uniform),
        0, 1
    ))
```

A uniform distribution (maximum uncertainty) maps to 0.0; a perfect one-hot (maximum certainty) maps to 1.0.

---

## A4. The Plugin Class

```python
class MGazePlugin(GazePlugin):
    name = "mgaze"
    mode = "per_face"
    is_fallback = True

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        return self._engine.estimate(face_bgr)

    def run_pipeline(self, **kwargs):
        from mindsight.GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)
```

### Key decisions

- **`is_fallback = True`** — MGaze is tried last, only if no other gaze plugin was activated. This makes it the automatic default without blocking user-installed plugins.
- **Wrapper pattern** — The plugin wraps an interchangeable engine (`GazeEstimationTorch` or `_GazeONNXWithConf`). The plugin class itself is thin — it delegates `estimate()` directly.
- **`run_pipeline()` delegation** — Instead of letting the gaze coordinator's default handler crop faces and call `estimate()` individually, MGaze delegates to `run_pitchyaw_pipeline`. This shared pipeline handles face cropping, left-to-right sorting, temporal smoothing, ray construction, and adaptive snap for any per-face pitch/yaw backend.

### The `run_pitchyaw_pipeline` helper

Any per-face plugin that outputs `(pitch, yaw, confidence)` can use this shared pipeline:

```python
def run_pipeline(self, **kwargs):
    from mindsight.GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
    return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)
```

The pipeline handles:

1. **Face cropping** — Extracts face ROIs from the full frame using RetinaFace bounding boxes.
2. **Eye centre extraction** — Uses RetinaFace keypoints for accurate gaze origin (falls back to bbox centre).
3. **Left-to-right sorting** — Deterministic face ordering for consistent track ID assignment.
4. **Temporal smoothing** — Applies the `GazeSmootherReID` if one is available in context.
5. **Ray construction** — Converts pitch/yaw to 2D direction, scales by ray length and face width.
6. **Forward gaze dead zone** — Suppresses errant rays when both angles are near zero.
7. **Adaptive snap** — Extends or snaps ray tips to nearby objects with hysteresis.

Returns the standard 7-tuple: `(persons_gaze, face_confs, face_bboxes, face_track_ids, face_objs, ray_snapped, ray_extended)`.

---

## A5. CLI Activation

```python
@classmethod
def add_arguments(cls, parser):
    g = parser.add_argument_group("MGaze backend")
    g.add_argument("--mgaze-model", default=DEFAULT_ONNX_MODEL,
                    help="Path to MGaze model weights (.onnx or .pt)")
    g.add_argument("--mgaze-arch", default=None, choices=ARCH_CHOICES,
                    help="Architecture name (required for .pt models)")
    g.add_argument("--mgaze-dataset", default="gaze360",
                    help="Dataset config key (default: gaze360)")
```

The `from_args` method auto-selects between ONNX and PyTorch based on the file extension:

```python
@classmethod
def from_args(cls, args):
    model = getattr(args, "mgaze_model", None)
    if not model:
        return None
    model = Path(model)
    if not model.exists():
        raise FileNotFoundError(f"MGaze model not found: {model}")

    if model.suffix.lower() == ".pt":
        if not arch:
            raise ValueError("--mgaze-arch is required for .pt models")
        engine = GazeEstimationTorch(str(model), arch, dataset, device=device)
    else:
        # ONNX path: auto-select execution provider
        prov = [p for p in prefs if p in ort.get_available_providers()]
        engine = _GazeONNXWithConf(model_path=None,
            session=ort.InferenceSession(str(model), providers=prov))

    return cls(engine)
```

### ONNX provider selection

The ONNX path tries providers in priority order: CoreML (Apple Silicon) → CUDA → DirectML → CPU. This gives automatic hardware acceleration without user configuration.

---

## A6. Running MGaze

```bash
# Default ONNX (auto-selected, shipped with MindSight)
python MindSight.py --source video.mp4

# Explicit ONNX model
python MindSight.py --source video.mp4 --mgaze-model weights/resnet18_gaze.onnx

# PyTorch model (requires architecture specification)
python MindSight.py --source video.mp4 \
    --mgaze-model weights/resnet50_gaze360.pt \
    --mgaze-arch resnet50 \
    --mgaze-dataset gaze360
```

---

## A7. Writing Your Own Per-Face Plugin

To create a new per-face gaze backend as a plugin:

```python
# Plugins/GazeTracking/MyBackend/my_backend.py

from __future__ import annotations

from Plugins import GazePlugin


class MyGazeBackend(GazePlugin):
    name = "my_gaze"
    mode = "per_face"

    def __init__(self, model_path: str):
        # Load your model here
        self._model = self._load_model(model_path)

    def estimate(self, face_bgr):
        """
        Receive a cropped face image (BGR numpy array).
        Return (pitch_radians, yaw_radians, confidence).
        """
        # Your inference here — preprocess, forward pass, postprocess
        pitch, yaw = self._model.predict(face_bgr)
        confidence = 0.8  # your confidence metric
        return float(pitch), float(yaw), confidence

    def run_pipeline(self, **kwargs):
        """Delegate to the shared per-face pipeline."""
        from mindsight.GazeTracking.pitchyaw_pipeline import run_pitchyaw_pipeline
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("My Gaze Backend")
        g.add_argument("--my-gaze-model", default=None,
                       help="Path to model weights. Activates this backend.")

    @classmethod
    def from_args(cls, args):
        model = getattr(args, "my_gaze_model", None)
        if not model:
            return None
        return cls(model_path=model)


PLUGIN_CLASS = MyGazeBackend
```

### What you get for free

By implementing just `estimate()` and delegating `run_pipeline()` to `run_pitchyaw_pipeline`, your plugin automatically inherits:

- Face cropping from RetinaFace detections
- Eye-landmark gaze origin (with bbox-centre fallback)
- Temporal smoothing via `GazeSmootherReID`
- Left-to-right face sorting for deterministic track IDs
- Confidence-scaled ray length (`--conf-ray`)
- Adaptive ray snapping with hysteresis (`--adaptive-ray`)
- Forward gaze dead zone (`--forward-gaze-threshold`)
- All CLI gaze flags work without any extra code in your plugin

### If you need more control

Override `run_pipeline()` entirely to handle smoothing, ray construction, or multi-face batching yourself. Your method receives:

| Kwarg | Type | Description |
|-------|------|-------------|
| `frame` | ndarray | Full BGR frame |
| `faces` | list[dict] | RetinaFace face detections |
| `objects` | list[Detection] | Non-person detections |
| `gaze_cfg` | GazeConfig | Ray and snap parameters |
| `smoother` | GazeSmootherReID \| None | Temporal smoothing tracker |
| `snap_hysteresis` | SnapHysteresisTracker \| None | Snap hysteresis tracker |
| `aux_frames` | dict | Auxiliary per-participant video streams |

Must return the 7-tuple: `(persons_gaze, face_confs, face_bboxes, face_track_ids, face_objs, ray_snapped, ray_extended)`.

---

# Part B: Scene-Level Backend (Gazelle)

Gazelle is a scene-level gaze estimator built on DINOv2. It processes the entire scene image together with face bounding boxes in a single forward pass, producing per-face gaze-point heatmaps.

**Source:** `Plugins/GazeTracking/Gazelle/gazelle_backend.py`

---

## B1. File Structure

```
Plugins/GazeTracking/Gazelle/
├── __init__.py
├── gazelle_backend.py          # PLUGIN_CLASS = GazeEstimationGazelle
└── gazelle/                    # Vendored Gazelle library
    └── gazelle/
        ├── model.py            # get_gazelle_model(), load_gazelle_state_dict()
        ├── backbone.py         # DINOv2 backbone
        ├── dataloader.py
        └── utils.py
```

---

## B2. Class Definition

```python
class GazeEstimationGazelle(GazePlugin):
    name = "gazelle"
    mode = "scene"
```

`mode = "scene"` tells the gaze coordinator to call `estimate_frame()` (full frame + all bounding boxes) rather than `estimate()` (single cropped face).

### Model variants

```python
_VALID_MODELS = {
    "gazelle_dinov2_vitb14",
    "gazelle_dinov2_vitl14",
    "gazelle_dinov2_vitb14_inout",
    "gazelle_dinov2_vitl14_inout",
}
```

| Variant | Backbone | In/Out scoring |
|---------|----------|----------------|
| `gazelle_dinov2_vitb14` | ViT-B/14 | No |
| `gazelle_dinov2_vitb14_inout` | ViT-B/14 | Yes |
| `gazelle_dinov2_vitl14` | ViT-L/14 | No |
| `gazelle_dinov2_vitl14_inout` | ViT-L/14 | Yes |

The `_inout` variants add a head predicting whether each person is looking inside or outside the frame. When in-frame confidence falls below the threshold, gaze confidence is attenuated.

### Constructor

```python
def __init__(self, model_name, ckpt_path, inout_threshold=0.5,
             skip_frames=0, use_fp16=False, use_compile=False, device="auto"):
```

| Parameter | Purpose |
|-----------|---------|
| `model_name` | Which variant to load |
| `ckpt_path` | Path to `.pt` checkpoint |
| `inout_threshold` | Confidence cutoff for `_inout` models (default 0.5) |
| `skip_frames` | Reuse cached results for N frames between inference |
| `use_fp16` | Half-precision on CUDA/MPS |
| `use_compile` | `torch.compile()` wrapper (PyTorch 2.0+) |
| `device` | `"auto"`, `"cpu"`, `"cuda"`, or `"mps"` |

---

## B3. The `estimate_frame()` Method

The core method for scene-level backends:

```python
def estimate_frame(self, frame_bgr, face_bboxes_px: list) -> list:
```

### Data flow

1. **Early return** if no faces.
2. **Frame-skip check** — reuse cached result if skip is active and face count unchanged.
3. **BGR→RGB** — `frame_bgr[:, :, ::-1]` zero-copy view, then PIL wrap.
4. **Normalize bboxes** — `(x1/w, y1/h, x2/w, y2/h)` for Gazelle's `[0,1]` range.
5. **Transform** — Resize 448×448, ToTensor, ImageNet normalize, unsqueeze, to device.
6. **Forward pass** — `model({"images": tensor, "bboxes": [norm]})` with `torch.no_grad()`.
7. **Heatmap extraction** — `out["heatmap"][0]` gives `[N, 64, 64]` per-face heatmaps.
8. **Batched peak extraction**:

```python
hm_flat = heatmaps.flatten(start_dim=1)          # [N, 4096]
maxvals, argmaxes = hm_flat.max(dim=1)           # [N], [N]
```

All N heatmaps are processed in one batched operation, with a single `.cpu().numpy()` call.

9. **Pixel coordinate conversion**:

```python
idx = int(argmaxes_np[i])
xy  = np.array([idx % 64 / 64 * w, idx // 64 / 64 * h])
```

10. **Inout attenuation** — For `_inout` models, if confidence < threshold: `conf *= score`.
11. **Cache and return** — Store results for frame-skip reuse.

---

## B4. The `raw_heatmaps()` Method

```python
def raw_heatmaps(self, frame_bgr, face_bboxes_px) -> np.ndarray:
```

Returns the full `[N, 64, 64]` sigmoid-activated heatmaps. Useful for visualization or analysis beyond the peak point.

---

## B5. CLI Activation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gazelle-model PATH` | str | None | Checkpoint path. **Activates the backend.** |
| `--gazelle-name` | choice | `gazelle_dinov2_vitb14` | Model variant |
| `--gazelle-inout-threshold` | float | 0.5 | In/out confidence threshold |
| `--gazelle-device` | str | `auto` | Device override |
| `--gazelle-skip-frames` | int | 0 | Frames between inference |
| `--gazelle-fp16` | flag | False | Half-precision |
| `--gazelle-compile` | flag | False | `torch.compile()` |

---

## B6. Running Gazelle

```bash
# Standard usage
python MindSight.py --source video.mp4 \
    --gazelle-model checkpoints/gazelle_dinov2_vitb14_inout.pt \
    --gazelle-name gazelle_dinov2_vitb14_inout --gazelle-fp16

# With frame-skipping for slower hardware
python MindSight.py --source video.mp4 \
    --gazelle-model checkpoints/gazelle.pt --gazelle-skip-frames 2
```

---

# Key Design Patterns (Both Modes)

### Backend selection

The gaze coordinator tries plugins in registration order. The first `from_args` that returns a non-`None` instance wins. Plugins with `is_fallback = True` (like MGaze) are tried last, making them the automatic default.

### Lazy loading

All expensive operations (model loading, weight transfer to GPU) happen inside `from_args()`, not at import time. If the activation flag is not set, no resources are consumed.

### PLUGIN_CLASS sentinel

Both plugins expose `PLUGIN_CLASS = ClassName` at module level. This is what `PluginRegistry.discover()` looks for:

```python
# At the bottom of your module:
PLUGIN_CLASS = MGazePlugin       # per-face
PLUGIN_CLASS = GazeEstimationGazelle  # scene-level
```

---


## Part C: Composite Backend (temporarily removed)

> **Removed in v0.8.** The previous Part C used the GazelleSnap plugin as a worked example of a composite backend -- a plugin that combines per-face gaze estimation with post-processing features like ray forming and heatmap blending. GazelleSnap was deprecated (its features were folded into the core `mindsight/PostProcessing/RayForming/` pipeline) and deleted in v0.8.
>
> A replacement composite-plugin example (likely rewritten around one of the surviving plugins such as IrisRefinedGaze) is TODO. In the meantime, read Parts A and B for the per-face and scene-level patterns, and skim `mindsight/PostProcessing/RayForming/gazelle_provider.py` for the core integration pattern that replaced GazelleSnap.
