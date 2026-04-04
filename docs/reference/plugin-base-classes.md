# Plugin Base Classes

MindSight defines four plugin base classes. Every plugin must subclass exactly one of these and implement the required methods.

---

## GazePlugin

Base class for gaze estimation backends.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Unique identifier for the backend (e.g., `"gazelle"`). |
| `mode` | `str` | Processing mode: `"face"` (per-crop) or `"frame"` (whole-frame). |
| `is_fallback` | `bool` | If `True`, this backend is used when the preferred one fails. |

### Methods

```python
def estimate(self, face_bgr: np.ndarray) -> Tuple[float, float]:
```
Estimate gaze from a single face crop. Returns `(pitch, yaw)` in radians. Only called when `mode == "face"`.

---

```python
def estimate_frame(
    self,
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[float, float]]:
```
Estimate gaze for all faces in one forward pass. Returns a list of `(pitch, yaw)` tuples, one per bbox. Only called when `mode == "frame"`.

---

```python
def run_pipeline(self, **kwargs) -> Any:
```
Optional. Run a full custom pipeline. Receives the same keyword arguments as the main pipeline loop. Return value is backend-specific.

---

```python
@staticmethod
def add_arguments(parser: argparse.ArgumentParser) -> None:
```
Register backend-specific CLI arguments on the given parser.

---

```python
@classmethod
def from_args(cls, args: argparse.Namespace) -> "GazePlugin":
```
Construct an instance from parsed CLI arguments.

---

## ObjectDetectionPlugin

Base class for object detection backends.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Unique identifier for the detection backend. |

### Methods

```python
def detect(
    self,
    frame: np.ndarray,
    detection_frame: np.ndarray,
    all_dets: Any,
    det_cfg: dict,
) -> Any:
```
Run detection on a frame. `detection_frame` may be a resized or preprocessed copy of `frame`. `all_dets` carries forward detections from a previous stage (e.g., the built-in YOLOE pass). `det_cfg` contains detection parameters from the pipeline config. Returns updated detections in the same format as `all_dets`.

---

```python
@staticmethod
def add_arguments(parser: argparse.ArgumentParser) -> None:
```
Register detection-specific CLI arguments.

---

```python
@classmethod
def from_args(cls, args: argparse.Namespace) -> "ObjectDetectionPlugin":
```
Construct an instance from parsed CLI arguments.

---

## PhenomenaPlugin

Base class for phenomena tracker plugins.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Unique identifier for the tracker (e.g., `"joint_attention"`). |
| `dashboard_panel` | `Optional[str]` | Name of the dashboard section, or `None` to skip dashboard rendering. |

### Methods

```python
def update(self, **kwargs) -> dict:
```
Called once per frame. Receives keyword arguments including `frame_idx`, `person_gazes`, `detections`, and other pipeline state. Returns a dict of computed metrics for this frame.

---

```python
def draw_frame(self, frame: np.ndarray) -> np.ndarray:
```
Draw annotations onto the video frame. Called after `update()`. Returns the modified frame.

---

```python
def dashboard_section(
    self,
    panel: np.ndarray,
    y: int,
    line_h: int,
    pid_map: Dict[int, str],
) -> int:
```
Render a section on the dashboard image. `panel` is the image buffer, `y` is the starting vertical offset, `line_h` is the line height in pixels, and `pid_map` maps track IDs to participant labels. Returns the new `y` position after drawing.

---

```python
def dashboard_data(self, pid_map: Dict[int, str]) -> dict:
```
Return structured data for the GUI dashboard. `pid_map` maps track IDs to participant labels.

---

```python
def csv_rows(
    self,
    total_frames: int,
    pid_map: Dict[int, str],
) -> List[dict]:
```
Return a list of dicts suitable for CSV output. Each dict is one row.

---

```python
def console_summary(
    self,
    total_frames: int,
    pid_map: Dict[int, str],
) -> str:
```
Return a human-readable summary string printed to the console at the end of a run.

---

```python
def time_series_data(self) -> List[Any]:
```
Return time-series data for charting (e.g., per-frame metric values).

---

```python
def latest_metric(self) -> Any:
```
Return the most recent scalar metric value (used by live dashboard widgets).

---

```python
def latest_metrics(self) -> dict:
```
Return a dict of the most recent metric values (multi-metric trackers).

---

```python
def dashboard_widget(self) -> Optional[Any]:
```
Return a PyQt6 widget for the GUI dashboard, or `None` for the default rendering.

---

```python
def dashboard_widget_update(self, data: dict) -> None:
```
Update the custom dashboard widget with new data from `dashboard_data()`.

---

```python
@staticmethod
def add_arguments(parser: argparse.ArgumentParser) -> None:
```
Register tracker-specific CLI arguments.

---

```python
@classmethod
def from_args(cls, args: argparse.Namespace) -> "PhenomenaPlugin":
```
Construct an instance from parsed CLI arguments.

---

## DataCollectionPlugin

Base class for data collection and output plugins.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Unique identifier for the data collector. |

### Methods

```python
def on_frame(self, **kwargs) -> None:
```
Called once per processed frame. Receives keyword arguments including `frame_idx`, `frame`, `person_gazes`, `detections`, and phenomena results. Use this to accumulate per-frame data.

---

```python
def on_run_complete(self, **kwargs) -> None:
```
Called once after the pipeline finishes processing all frames. Receives summary keyword arguments including `total_frames`, `output_dir`, and `pid_map`. Use this to flush buffers, close files, or write final outputs.

---

```python
def generate_charts(
    self,
    output_dir: str,
    **kwargs,
) -> None:
```
Generate charts or visualizations and save them to `output_dir`. Called after `on_run_complete()`.

---

```python
@staticmethod
def add_arguments(parser: argparse.ArgumentParser) -> None:
```
Register output-specific CLI arguments.

---

```python
@classmethod
def from_args(cls, args: argparse.Namespace) -> "DataCollectionPlugin":
```
Construct an instance from parsed CLI arguments.
