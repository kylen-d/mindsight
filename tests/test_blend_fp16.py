"""fp16 blend-path safety net (v1.1 W2.3).

--rf-gazelle-fp16 runs the blend-path Gaze-LLE model in half precision.  It
can never be byte-identical to fp32 (so it stays non-default and outside the
golden gate), but it must stay CLOSE: this standing slow test runs a short
segment of the blend smoke in both precisions and bounds the drift -- finalized
ray endpoints within a few pixels on average, hit totals within 2.
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO = REPO_ROOT / "test_data" / "trimmed.mp4"
SEGMENT_FRAMES = 120

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not VIDEO.exists(),
                       reason="test_data/trimmed.mp4 not available"),
]


def _blend_args(fp16: bool):
    args = ["--source", str(VIDEO),
            "--model", "Weights/YOLO/yolov8n.pt",
            "--mgaze-model", "Weights/MGaze/resnet50_gaze.onnx",
            "--rf-gazelle-model", "Weights/Gazelle/gazelle_dinov2_vitb14.pt",
            "--rf-gazelle-interval", "10",
            "--no-dashboard"]
    if fp16:
        args.append("--rf-gazelle-fp16")
    return args


def _run_segment(fp16: bool):
    from mindsight.cli import _args
    from mindsight.factory import build_from_namespace
    from mindsight.pipeline import CancelToken, Pipeline, RunOptions

    ns = _args(_blend_args(fp16))
    try:
        tup = build_from_namespace(ns)
    except Exception as exc:
        pytest.skip(f"could not build blend models: {exc}")

    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
     active_plugins, phenomena_cfg, detection_plugins, depth_cfg,
     depth_backend, gazelle_provider, ray_cfg) = tup
    pipeline = Pipeline(
        yolo=yolo, face_det=face_det, gaze_eng=gaze_eng, gaze_cfg=gaze_cfg,
        det_cfg=det_cfg, tracker_cfg=tracker_cfg, output_cfg=output_cfg,
        plugin_instances=active_plugins, detection_plugins=detection_plugins,
        phenomena_cfg=phenomena_cfg, depth_cfg=depth_cfg,
        depth_backend=depth_backend, gazelle_provider=gazelle_provider,
        ray_cfg=ray_cfg)

    cancel = CancelToken()
    endpoints, hits = [], 0
    for result in pipeline.run(str(VIDEO),
                               options=RunOptions(no_dashboard=True),
                               cancel=cancel):
        ctx = result.context
        frame_pts = [tuple(map(float, ray_end))
                     for _o, ray_end, _a in ctx.get('persons_gaze', [])]
        endpoints.append(frame_pts)
        hits += len(result.events)
        if result.frame_no + 1 >= SEGMENT_FRAMES:
            cancel.cancel()
    return endpoints, hits


def test_fp16_blend_stays_close_to_fp32(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available()
            or torch.backends.mps.is_available()):
        pytest.skip("fp16 requires CUDA or MPS")

    ep32, hits32 = _run_segment(fp16=False)
    ep16, hits16 = _run_segment(fp16=True)

    assert abs(hits16 - hits32) <= 2, (hits32, hits16)

    devs = []
    for f32, f16 in zip(ep32, ep16):
        for p32, p16 in zip(f32, f16):
            devs.append(np.hypot(p32[0] - p16[0], p32[1] - p16[1]))
    assert devs, "no endpoints collected"
    mean_dev = float(np.mean(devs))
    p95_dev = float(np.percentile(devs, 95))
    # One Euro smoothing keeps tiny numeric differences from accumulating;
    # anything beyond a few pixels means fp16 has genuinely diverged.
    assert mean_dev < 3.0, f"mean endpoint deviation {mean_dev:.2f}px"
    assert p95_dev < 10.0, f"p95 endpoint deviation {p95_dev:.2f}px"
