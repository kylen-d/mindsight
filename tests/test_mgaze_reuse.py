"""MGazeReuseCache -- perceptual no-change gate for per-face gaze (v1.1 W2.2).

The per-face gaze model historically ran every frame for every face
regardless of skip settings.  The cache reuses the previous frame's estimate
only when the face crop is visually unchanged (mean-abs 32x32 grayscale diff
<= eps) AND the bbox overlaps the previous entry (IoU >= 0.5) -- so it can
never lie to the blend scheduler's fixation history, and face-order churn
degrades to a miss, never a mixup.  Default eps = 0.0 keeps it fully off.
"""

import numpy as np

from mindsight.GazeTracking.gaze_processing import MGazeReuseCache


class _CountingEstimator:
    def __init__(self):
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        return (0.1, -0.2, 0.9)


def _crop(value=128, size=64):
    return np.full((size, size, 3), value, dtype=np.uint8)


BBOX = (10, 10, 74, 74)


def test_identical_crop_reuses_previous_estimate():
    cache, est = MGazeReuseCache(eps=2.0), _CountingEstimator()
    r1 = cache.estimate(BBOX, _crop(), est)
    cache.end_frame()
    r2 = cache.estimate(BBOX, _crop(), est)
    assert est.calls == 1          # second frame reused
    assert r1 == r2
    assert cache.hits == 1 and cache.misses == 1


def test_changed_crop_misses():
    cache, est = MGazeReuseCache(eps=2.0), _CountingEstimator()
    cache.estimate(BBOX, _crop(128), est)
    cache.end_frame()
    cache.estimate(BBOX, _crop(160), est)   # mean-abs diff = 32 > eps
    assert est.calls == 2
    assert cache.hits == 0


def test_moved_bbox_misses_even_with_identical_pixels():
    cache, est = MGazeReuseCache(eps=2.0), _CountingEstimator()
    cache.estimate(BBOX, _crop(), est)
    cache.end_frame()
    far = (200, 200, 264, 264)              # IoU 0 vs previous entry
    cache.estimate(far, _crop(), est)
    assert est.calls == 2
    assert cache.hits == 0


def test_eps_zero_disables_reuse_entirely():
    cache, est = MGazeReuseCache(eps=0.0), _CountingEstimator()
    for _ in range(3):
        cache.estimate(BBOX, _crop(), est)
        cache.end_frame()
    assert est.calls == 3
    assert cache.hits == 0 and cache.misses == 0   # gate never engaged


def test_reuse_only_matches_one_frame_back():
    cache, est = MGazeReuseCache(eps=2.0), _CountingEstimator()
    cache.estimate(BBOX, _crop(), est)
    cache.end_frame()
    cache.end_frame()                        # empty frame: entries expire
    cache.estimate(BBOX, _crop(), est)
    assert est.calls == 2


def test_flag_reaches_tracker_config():
    from mindsight.cli_flags import parse_cli
    from mindsight.pipeline_config import TrackerConfig

    ns = parse_cli([])
    assert TrackerConfig.from_namespace(ns).mgaze_reuse_eps == 0.0

    ns = parse_cli(["--mgaze-reuse-eps", "3.5"])
    assert TrackerConfig.from_namespace(ns).mgaze_reuse_eps == 3.5
