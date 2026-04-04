# Third-Party Licenses

MindSight depends on several third-party libraries and embedded components.
This file documents their licenses for compliance purposes.

## Embedded Components

### MGaze / gaze-estimation
- **License:** MIT
- **Copyright:** (c) 2024 Yakhyokhuja Valikhujaev
- **Location:** `GazeTracking/Backends/MGaze/gaze-estimation/`
- **Full license:** `GazeTracking/Backends/MGaze/gaze-estimation/LICENSE`

### Gazelle (Gaze-LLE)
- **License:** MIT
- **Copyright:** (c) 2024 Fiona Ryan
- **Location:** `Plugins/GazeTracking/Gazelle/gazelle/`
- **Full license:** `Plugins/GazeTracking/Gazelle/gazelle/LICENSE`

## Python Dependencies

| Package | License | Notes |
|---------|---------|-------|
| ultralytics | AGPL-3.0 | YOLO/YOLOE detection engine. The combined work must comply with AGPL-3.0 network-use provisions. |
| PyQt6 | GPL-3.0 | GUI framework (GPL-licensed edition) |
| CLIP (ultralytics fork) | MIT | Vision-language model, installed via git |
| torch | BSD-3-Clause | Deep learning framework |
| torchvision | BSD-3-Clause | Vision utilities for PyTorch |
| onnxruntime | MIT | ONNX model inference engine |
| opencv-python | Apache-2.0 | Computer vision library |
| opencv-contrib-python | Apache-2.0 | Extended OpenCV modules |
| uniface | MIT | RetinaFace face detector |
| mediapipe | Apache-2.0 | MediaPipe inference |
| numpy | BSD-3-Clause | Numerical computing |
| pandas | BSD-3-Clause | Data manipulation |
| polars | MIT | DataFrame library |
| scipy | BSD-3-Clause | Scientific computing |
| scikit-image | BSD-3-Clause | Image processing |
| scikit-learn | BSD-3-Clause | Machine learning |
| matplotlib | PSF/BSD | Plotting |
| Pillow | HPND (Historical Permission Notice and Disclaimer) | Image handling |
| PyYAML | MIT | YAML parsing |
| requests | Apache-2.0 | HTTP client |
| rich | MIT | Terminal formatting |
| tqdm | MPL-2.0 | Progress bars |
| typer | MIT | CLI framework |
| click | BSD-3-Clause | CLI utilities |
| networkx | BSD-3-Clause | Graph algorithms |
| sympy | BSD-3-Clause | Symbolic math |
| huggingface_hub | Apache-2.0 | HuggingFace model hub integration |

## External Models (Loaded at Runtime)

| Model | License | Source |
|-------|---------|--------|
| DINOv2 | Apache-2.0 | Meta AI, loaded via `torch.hub` by the Gazelle backend |
| YOLO / YOLOE weights | AGPL-3.0 | Ultralytics, auto-downloaded on first use |

## AGPL-3.0 Notice

MindSight is licensed under AGPL-3.0, consistent with its dependency on
[ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). If you distribute
or provide network access to this software, you must make the complete corresponding
source code available under the same license. See the
[AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.html) for full details.
