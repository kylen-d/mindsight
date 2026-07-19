"""
Plugins/GazeTracking/MPIIFaceGaze/mpiifacegaze_backend.py — MPIIFaceGaze backend.

Head-pose-normalized per-face gaze estimation: hysts' MPIIFaceGaze
resnet_simple direct-regression model (pytorch_mpiigaze_demo release)
running on the vendored ptgaze normalization core.  Per face: MediaPipe
468-point landmarks on the RetinaFace crop -> solvePnP head pose against
the canonical face model -> normalizing warp to a 224x224 patch ->
model -> gaze vector denormalized into camera coordinates -> the
MindSight pitch/yaw ray convention.

Weights provenance: the checkpoint is MIT-licensed (hysts release) but
trained on MPIIFaceGaze (CC BY-NC-SA, research-only) — carried as a
manifest ``license_note`` like the Gaze360 MobileGaze rows.

Activation
----------
Pass ``--mpiifacegaze-model mpiifacegaze_resnet_simple.pth`` (bare
filenames resolve through ``Weights/MPIIFaceGaze/``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from Plugins import GazePlugin

# ptgaze transform: torchvision Normalize on a cv2 (BGR) patch, so the
# ImageNet stats are applied in BGR channel order.
_MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_STD_BGR = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def _build_model():
    """hysts resnet_simple: resnet18 stages 1-3 + attention conv + fc."""
    import torch
    import torchvision

    class _Backbone(torchvision.models.ResNet):
        def __init__(self):
            super().__init__(torchvision.models.resnet.BasicBlock,
                             [2, 2, 2, 1])
            del self.layer4, self.avgpool, self.fc

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            return self.layer3(self.layer2(self.layer1(x)))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = _Backbone()
            self.conv = torch.nn.Conv2d(256, 1, 1)
            self.fc = torch.nn.Linear(256 * 14 * 14, 2)

        def forward(self, x):
            x = self.feature_extractor(x)
            y = torch.nn.functional.relu(self.conv(x))
            x = (x * y).view(x.size(0), -1)
            return self.fc(x)

    return _Model()


class MPIIFaceGazeEngine:
    """(frame, bbox) -> (pitch_rad, yaw_rad, confidence) estimator."""

    def __init__(self, weight_path, device: str = "auto", landmarker=None):
        import torch

        from mindsight.utils.device import resolve_device
        self._torch = torch
        self.device = resolve_device(device)

        model = _build_model()
        sd = torch.load(str(weight_path), map_location="cpu",
                        weights_only=True)
        model.load_state_dict(sd.get("model", sd))       # strict
        self.model = model.to(self.device).eval()

        if landmarker is None:
            from mindsight.GazeTracking.normalized.landmarks import (
                CropFaceLandmarker,
            )
            landmarker = CropFaceLandmarker()
        self._landmarker = landmarker
        self._camera_matrix = None
        self._camera_shape = None

    def _camera_for(self, shape):
        """Dummy pinhole intrinsics for a frame shape: f = width,
        principal point at the frame center (the ptgaze dummy-camera
        convention the prototype validated)."""
        if self._camera_shape != shape:
            h, w = shape
            self._camera_matrix = np.array(
                [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
            self._camera_shape = shape
        return self._camera_matrix

    def estimate_in_frame(self, frame_bgr, bbox):
        """Estimate gaze for the face in *bbox* using full-frame context.

        Returns zero angles at zero confidence when the landmarker finds
        no face in the crop — the pipeline's forward-gaze dead zone turns
        that into a short stub ray instead of a stale long one.
        """
        from mindsight.GazeTracking.normalized import (
            compute_face_center,
            compute_normalizing_rotation,
            denormalize_gaze_vector,
            estimate_head_pose,
            gaze_angles_to_vector,
            normalize_image,
            vector_to_pipeline_pitchyaw,
        )

        landmarks = self._landmarker.detect(frame_bgr, bbox)
        if landmarks is None:
            return 0.0, 0.0, 0.0

        cam = self._camera_for(frame_bgr.shape[:2])
        rot, _tvec, model3d = estimate_head_pose(landmarks, cam)
        center = compute_face_center(model3d, mode="mpiifacegaze")
        nrot = compute_normalizing_rotation(center, rot)
        patch = normalize_image(frame_bgr, cam, nrot,
                                float(np.linalg.norm(center)))

        x = ((patch.astype(np.float32) / 255.0) - _MEAN_BGR) / _STD_BGR
        t = self._torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
        with self._torch.no_grad():
            angles = self.model(t)[0].float().cpu().numpy()

        gvec = denormalize_gaze_vector(gaze_angles_to_vector(angles), nrot)
        pitch, yaw = vector_to_pipeline_pitchyaw(gvec)
        return pitch, yaw, 1.0

    def estimate(self, face_bgr):
        """Crop-only fallback: treat the crop as the frame.  Degraded
        (crop-sized intrinsics) — the pipeline prefers estimate_in_frame."""
        h, w = face_bgr.shape[:2]
        return self.estimate_in_frame(face_bgr, (0, 0, w, h))


class MPIIFaceGazePlugin(GazePlugin):
    """MPIIFaceGaze head-pose-normalized per-face gaze plugin."""

    name = "mpiifacegaze"
    mode = "per_face"
    is_fallback = False

    def __init__(self, engine):
        self._engine = engine

    def estimate(self, face_bgr):
        return self._engine.estimate(face_bgr)

    def estimate_in_frame(self, frame_bgr, bbox):
        return self._engine.estimate_in_frame(frame_bgr, bbox)

    def run_pipeline(self, **kwargs):
        from mindsight.GazeTracking.pitchyaw_pipeline import (
            run_pitchyaw_pipeline,
        )
        return run_pitchyaw_pipeline(gaze_eng=self, **kwargs)

    # ── CLI protocol ─────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("MPIIFaceGaze backend")
        g.add_argument(
            "--mpiifacegaze-model", default=None, metavar="PATH",
            help=(
                "Path to the hysts MPIIFaceGaze resnet_simple checkpoint "
                "(.pth).  Activates the head-pose-normalized MPIIFaceGaze "
                "backend (requires the MediaPipe face_landmarker.task "
                "asset).  Weights are research-provenance -- trained on "
                "MPIIFaceGaze (CC BY-NC-SA); see THIRD_PARTY_LICENSES."
            ),
        )

    @classmethod
    def from_args(cls, args):
        from mindsight.weights import resolve_weight
        model = getattr(args, "mpiifacegaze_model", None)
        if not model:
            return None
        path = Path(resolve_weight("MPIIFaceGaze", str(model)))
        if not path.exists():
            raise FileNotFoundError(
                f"MPIIFaceGaze checkpoint not found: {path}\n"
                "Install it with: mindsight-weights --backend MPIIFaceGaze")
        device = getattr(args, "device", "auto")
        print(f"Backend: MPIIFaceGaze torch  resnet_simple/{device}")
        return cls(MPIIFaceGazeEngine(path, device=device))


PLUGIN_CLASS = MPIIFaceGazePlugin
