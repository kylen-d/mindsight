"""Integration test for the public Pipeline API (SP1.2 step 4).

Drives ``Pipeline.run`` -- the generator -- directly over the real sample
video, with no display and no cv2.imshow in the loop (that lives in
``run_to_completion``).  Pins three things:

* every frame of test_data/trimmed.mp4 is yielded as a FrameResult (869);
* the default-config hit-event count matches the golden (0 -- the default net
  scores no intersections on this clip; see localref/baselines/v1);
* a mid-run cancellation stops cleanly and finalizes outputs (summary CSV).

Requires the sample video and the model weights, so it self-skips when either
is unavailable (e.g. a checkout without the gitignored weights).
"""

import dataclasses
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO = REPO_ROOT / "test_data" / "trimmed.mp4"
EXPECTED_FRAMES = 869
EXPECTED_HITS = 0  # default config on trimmed.mp4 (golden_events.csv: header only)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not VIDEO.exists(), reason="test_data/trimmed.mp4 not available"),
]


@pytest.fixture(scope="module")
def built():
    """Build models + config dataclasses exactly as the CLI does (default cfg)."""
    from ms.cli import _args, _build_from_args
    ns = _args(["--source", str(VIDEO), "--no-dashboard"])
    try:
        tup = _build_from_args(ns)
    except Exception as exc:  # missing weights / backends in this environment
        pytest.skip(f"could not build pipeline models: {exc}")
    return ns, tup


def _make_pipeline(tup, **output_overrides):
    """Assemble a Pipeline from the _build_from_args tuple, optionally
    overriding OutputConfig fields (e.g. log/summary paths)."""
    from ms.pipeline import Pipeline
    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
     active_plugins, phenomena_cfg, detection_plugins, depth_cfg,
     depth_backend, gazelle_provider, ray_cfg) = tup
    if output_overrides:
        output_cfg = dataclasses.replace(output_cfg, **output_overrides)
    return Pipeline(
        yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
        gaze_cfg=gaze_cfg, det_cfg=det_cfg, tracker_cfg=tracker_cfg,
        output_cfg=output_cfg, plugin_instances=active_plugins,
        detection_plugins=detection_plugins, phenomena_cfg=phenomena_cfg,
        depth_cfg=depth_cfg, depth_backend=depth_backend,
        gazelle_provider=gazelle_provider, ray_cfg=ray_cfg,
    )


def test_pipeline_yields_every_frame(built):
    from ms.pipeline import FrameResult, RunOptions
    _ns, tup = built
    pipeline = _make_pipeline(tup)

    results = list(pipeline.run(str(VIDEO), options=RunOptions(no_dashboard=True)))

    assert len(results) == EXPECTED_FRAMES
    assert all(isinstance(r, FrameResult) for r in results)
    # frame numbers are the contiguous 0..N-1 sequence
    assert [r.frame_no for r in results] == list(range(EXPECTED_FRAMES))
    # hit-event total matches the default golden
    assert sum(len(r.events) for r in results) == EXPECTED_HITS
    # FrameResult field sanity on a representative frame
    r = results[100]
    assert r.total_frames == 101
    assert r.fps > 0
    assert r.t_seconds == pytest.approx(r.frame_no / r.fps)
    assert r.annotated is not None
    assert r.context is not None


def test_cancellation_stops_and_finalizes(built, tmp_path):
    from ms.pipeline import CancelToken, RunOptions
    _ns, tup = built
    log_path = tmp_path / "events.csv"
    summary_path = tmp_path / "summary.csv"
    pipeline = _make_pipeline(
        tup, log_path=str(log_path), summary_path=str(summary_path))

    cancel = CancelToken()
    results = []
    for i, r in enumerate(pipeline.run(str(VIDEO),
                                       options=RunOptions(no_dashboard=True),
                                       cancel=cancel)):
        results.append(r)
        if i == 9:            # cancel after the 10th frame
            cancel.cancel()

    # The loop checks the token at the next frame boundary, so exactly the
    # frames already in flight (0..9) come through -- not all 869.
    assert len(results) == 10
    assert [r.frame_no for r in results] == list(range(10))
    # Outputs finalized through the normal post-run path despite cancellation:
    # the event log was opened/closed and the summary CSV was written.
    assert log_path.exists()
    assert summary_path.exists() and summary_path.stat().st_size > 0


def test_image_source_yields_one_frameresult(built, tmp_path):
    """A still-image source yields exactly one FrameResult and needs no display
    (imshow/waitKey live in run_to_completion now, not the generator)."""
    import cv2

    from ms.pipeline import FrameResult
    _ns, tup = built
    # grab frame 0 of the sample clip as a .jpg
    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame = cap.read()
    cap.release()
    assert ok
    jpg = tmp_path / "frame.jpg"
    cv2.imwrite(str(jpg), frame)

    pipeline = _make_pipeline(tup)
    results = list(pipeline.run(str(jpg)))

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, FrameResult)
    assert r.frame_no == 0
    assert r.total_frames == 1
    assert r.annotated is not None
    assert r.context is not None


def test_from_config_matches_direct(built):
    """Pipeline.from_config (unified schema path) yields the same config
    dataclasses the direct dataclass constructor holds."""
    from ms.config import PipelineConfig
    from ms.pipeline import Pipeline
    ns, tup = built
    (yolo, face_det, gaze_eng, gaze_cfg, det_cfg, tracker_cfg, output_cfg,
     active_plugins, phenomena_cfg, detection_plugins, depth_cfg,
     depth_backend, gazelle_provider, ray_cfg) = tup

    config = PipelineConfig.from_namespace(
        ns, class_ids=det_cfg.class_ids, blacklist=det_cfg.blacklist)
    pipeline = Pipeline.from_config(
        config, yolo=yolo, face_det=face_det, gaze_eng=gaze_eng,
        plugin_instances=active_plugins, detection_plugins=detection_plugins,
        depth_backend=depth_backend, gazelle_provider=gazelle_provider)

    assert pipeline.gaze_cfg == gaze_cfg
    assert pipeline.det_cfg == det_cfg
    assert pipeline.tracker_cfg == tracker_cfg
    assert pipeline.output_cfg == output_cfg
    assert pipeline.phenomena_cfg == phenomena_cfg
    assert pipeline.depth_cfg == depth_cfg
    assert pipeline.ray_cfg == ray_cfg
    assert pipeline.yolo is yolo
    assert pipeline.gazelle_provider is gazelle_provider
