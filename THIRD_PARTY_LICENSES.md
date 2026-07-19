# Third-Party Licenses

MindSight depends on several third-party libraries and embedded components.
This file documents their licenses for compliance purposes.

## Embedded Components

### MGaze / gaze-estimation
- **License:** MIT
- **Copyright:** (c) 2024 Yakhyokhuja Valikhujaev
- **Location:** `mindsight/GazeTracking/Backends/MGaze/gaze-estimation/`
- **Full license:** `mindsight/GazeTracking/Backends/MGaze/gaze-estimation/LICENSE`

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
| scikit-learn | BSD-3-Clause | Machine learning (used by vendored Gazelle utils) |
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
| MobileGaze weights (`*_gaze.onnx` / `*.pt`) | MIT (yakhyo/gaze-estimation release) | Trained on **Gaze360**, a non-commercial research dataset -- treat the weights as research use only. The manifest carries this as `license_note` and the Models tab surfaces it. |
| Gaze-LLE checkpoints | MIT (fkryan/gazelle release) | Trained on GazeFollow / VideoAttentionTarget (research datasets); DINOv2 backbone Apache-2.0. |
| Gaze-LLE DINOv3-distilled ONNX (pico / ViT tiny-plus) | MIT (PINTO0309/gazelle-dinov3 release) | Blend-path gaze-target models. Backbones are Apache-2.0 (D-FINE HGNetV2 / DEIMv2 ViT) trained on DINOv3 *outputs* via distillation -- no Meta weights embedded. Trained on GazeFollow / VideoAttentionTarget (research datasets). Downloaded from the upstream release; not redistributed by MindSight. |
| Gaze-LLE DINOv3 ONNX (ViT-S/16) | MIT (PINTO0309/gazelle-dinov3 release) + [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | Embeds the Meta DINOv3 ViT-S/16 backbone weights, so use is additionally subject to the DINOv3 License (Copyright (c) Meta Platforms, Inc.): commercial use is permitted; redistributing the weights or derivatives requires passing through the DINOv3 License text and attribution. MindSight does not redistribute these weights -- the manifest downloads them from the upstream release on demand. Trained on GazeFollow / VideoAttentionTarget (research datasets). |
| YOLO / YOLOE weights | AGPL-3.0 | Ultralytics, auto-downloaded on first use |
| FastSAM-s (`Weights/SAM/`) | AGPL-3.0 | CASIA-IVA-Lab FastSAM via the Ultralytics wrapper and asset release; powers the VP Builder's Suggest mode. Same license class as the ultralytics package itself. |
| RetinaFace weights | MIT (uniface release) | yakhyo/uniface, auto-downloaded to `~/.uniface` on first use; trained on WIDER FACE |
| ArcFace embeddings (`--face-reid-sim`) | MIT (uniface release) | yakhyo/uniface, auto-downloaded on first use. Upstream provenance: InsightFace model zoo, which marks its models for **non-commercial research use**; the training set (WebFace600K) is research-only. Off by default; enable only if that provenance is acceptable for your use. |

## AGPL-3.0 Notice

MindSight is licensed under AGPL-3.0, consistent with its dependency on
[ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). If you distribute
or provide network access to this software, you must make the complete corresponding
source code available under the same license. See the
[AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.html) for full details.
