"""GazelleProvider + gazelle_backend inout wiring (v1.1 W0.2).

Historically ``GazeEstimationGazelle.raw_heatmaps`` never set
``_last_inout``, so ``GazelleProvider.step`` always cached
``inout_score=1.0`` -- the in/out-of-frame head was dead on the blend
path.  These tests pin the repaired wire end-to-end:

* ``raw_heatmaps`` stores per-face inout scores (or None for non-inout
  model variants / empty batches) as a side effect;
* ``GazelleProvider.step`` propagates those scores per track into the
  ``HeatmapCache`` (falling back to 1.0 when the engine reports None).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mindsight.PostProcessing.RayForming.gazelle_provider import (  # noqa: E402
    GazelleProvider,
)

FRAME = np.zeros((100, 200, 3), dtype=np.uint8)
BBOXES = [(10, 10, 40, 40), (60, 20, 90, 50)]


# ── Backend: raw_heatmaps side effect ──────────────────────────────────────

def _bare_engine(has_inout: bool, inout_row):
    """Build a GazeEstimationGazelle without loading real weights."""
    from PIL import Image  # noqa: F401  (import check, matches backend deps)
    from torchvision import transforms

    from Plugins.GazeTracking.Gazelle.gazelle_backend import (
        GazeEstimationGazelle,
    )

    eng = GazeEstimationGazelle.__new__(GazeEstimationGazelle)
    eng._PIL = Image
    eng._has_inout = has_inout
    eng._use_fp16 = False
    eng.device = torch.device("cpu")
    eng.transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    eng._last_inout = None

    def fake_model(inp):
        n = len(inp["bboxes"][0])
        return {
            "heatmap": torch.rand(1, n, 64, 64),
            "inout": (torch.tensor([inout_row[:n]])
                      if inout_row is not None else None),
        }

    eng.model = fake_model
    return eng


def test_raw_heatmaps_sets_last_inout_for_inout_variant():
    eng = _bare_engine(has_inout=True, inout_row=[0.9, 0.2])
    hm = eng.raw_heatmaps(FRAME, BBOXES)
    assert hm.shape == (2, 64, 64)
    assert eng._last_inout == pytest.approx([0.9, 0.2])


def test_raw_heatmaps_leaves_none_for_non_inout_variant():
    eng = _bare_engine(has_inout=False, inout_row=None)
    eng._last_inout = np.array([0.5])  # stale value from a previous pass
    eng.raw_heatmaps(FRAME, BBOXES)
    assert eng._last_inout is None


def test_raw_heatmaps_empty_batch_resets_last_inout():
    eng = _bare_engine(has_inout=True, inout_row=[0.9])
    eng._last_inout = np.array([0.9])
    hm = eng.raw_heatmaps(FRAME, [])
    assert hm.shape == (0, 64, 64)
    assert eng._last_inout is None


# ── Provider: propagation into HeatmapCache ────────────────────────────────

class _StubEngine:
    """Engine double: fixed heatmaps + a canned _last_inout side effect."""

    def __init__(self, inout_row):
        self._inout_row = inout_row
        self._last_inout = None

    def raw_heatmaps(self, frame, bboxes):
        n = len(bboxes)
        self._last_inout = (np.asarray(self._inout_row[:n], dtype=np.float32)
                            if self._inout_row is not None else None)
        return np.full((n, 64, 64), 0.5, dtype=np.float32)


class _AlwaysFireScheduler:
    """Scheduler double that fires every tick for all known tracks."""

    def __init__(self, tids):
        self.tracked_tids = set(tids)
        self._tids = set(tids)

    def tick(self):
        return True, set(self._tids)

    def record_accepted(self, tids):
        pass

    def forget(self, tids):
        pass

    def advance_frame(self):
        pass


def _provider_with(engine, tids):
    p = GazelleProvider(engine, v_threshold=0.04, d_threshold=0.15,
                        min_call_gap=30)
    p._scheduler = _AlwaysFireScheduler(tids)
    return p


def test_step_propagates_inout_scores_into_cache():
    p = _provider_with(_StubEngine([0.9, 0.2]), tids=[7, 8])
    p.step(FRAME, BBOXES, [7, 8])

    _, _, inout7, _ = p.heatmap_cache.get(7)
    _, _, inout8, _ = p.heatmap_cache.get(8)
    assert inout7 == pytest.approx(0.9)
    assert inout8 == pytest.approx(0.2)


def test_step_defaults_inout_to_one_when_engine_reports_none():
    p = _provider_with(_StubEngine(None), tids=[3])
    p.step(FRAME, [BBOXES[0]], [3])

    _, _, inout, _ = p.heatmap_cache.get(3)
    assert inout == pytest.approx(1.0)
