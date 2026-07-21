"""Profile-bucket attribution in process_frame (v1.1 W0.1).

The --profile accumulators historically charged the entire process_frame
call (detection + depth + gaze + Gaze-LLE + overlay) to the 'detect'
bucket, hiding the heaviest stages.  process_frame now takes an optional
``prof`` dict and attributes each stage separately; Gaze-LLE time is
measured inside the gaze step (ctx['_prof_gazelle']) and split out of
the 'gaze' bucket.  These tests drive process_frame with stubbed stages.
"""

import time

import numpy as np
import pytest

import mindsight.pipeline as pl
from mindsight.pipeline_config import DetectionConfig, FrameContext, GazeConfig

GAZELLE_SECONDS = 0.005


@pytest.fixture
def stubbed_stages(monkeypatch):
    def fake_detection(ctx, **kwargs):
        ctx['all_dets'] = []
        ctx['persons'] = []
        ctx['objects'] = []

    def fake_gaze(ctx, **kwargs):
        time.sleep(0.01)
        ctx['persons_gaze'] = []
        ctx['hits'] = set()
        ctx['hit_events'] = []
        ctx['face_bboxes'] = []
        ctx['face_track_ids'] = []
        ctx['_prof_gazelle'] = GAZELLE_SECONDS

    monkeypatch.setattr(pl, 'run_detection_step', fake_detection)
    monkeypatch.setattr(pl, 'run_gaze_step', fake_gaze)
    monkeypatch.setattr(pl, 'draw_overlay', lambda ctx, **kwargs: None)
    monkeypatch.setattr(pl, 'joint_attention', lambda *a, **kw: set())


def _run_process_frame(prof):
    ctx = FrameContext(frame=np.zeros((16, 16, 3), dtype=np.uint8), frame_no=0)
    pl.process_frame(ctx, yolo=None, face_det=None, gaze_eng=None,
                     gaze_cfg=GazeConfig(), det_cfg=DetectionConfig(),
                     prof=prof)
    return ctx


def test_prof_buckets_attributed_per_stage(stubbed_stages):
    prof = {'detect': 0.0, 'depth': 0.0, 'gaze': 0.0, 'gazelle': 0.0,
            'phenomena': 0.0, 'draw': 0.0, 'dashboard': 0.0, 'n': 0}
    _run_process_frame(prof)

    # Gaze-LLE time comes from the gaze step's ctx seam, verbatim.
    assert prof['gazelle'] == pytest.approx(GAZELLE_SECONDS)
    # The gaze bucket is the gaze step's wall time MINUS the Gaze-LLE share.
    assert prof['gaze'] >= 0.01 - GAZELLE_SECONDS
    assert prof['gaze'] < 0.5
    # Detection is no longer the catch-all: it must not contain the
    # 10 ms the stubbed gaze step slept.
    assert prof['detect'] < 0.01
    # No depth backend configured: bucket stays (near) zero but exists.
    assert prof['depth'] < 0.01
    assert prof['draw'] >= 0.0


def test_prof_none_is_default_and_harmless(stubbed_stages):
    ctx = _run_process_frame(None)
    assert '_prof_gazelle' in ctx  # seam recorded by the gaze step stub
    # and nothing blew up without a prof dict (the pre-v1.1 call shape).
