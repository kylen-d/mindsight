"""W3X uniface-feature knobs: detection-dict adapter, --face-eye-origin,
and embedding-verified track revival (--face-reid-sim).

uniface 1.1.0 emits {'bbox': [x1,y1,x2,y2], 'confidence': float,
'landmarks': [[x,y]*5]} while the pipeline convention is a 5-long scored
bbox plus 'kps' -- before the adapter, eye-centre origins and the
EYE_CONF_THRESH gate were silently dead.  The adapter normalizes
unconditionally; USING the eye midpoint as ray origin stays behind
--face-eye-origin because the blessed baselines anchor at bbox centres.
"""
from __future__ import annotations

import numpy as np

from mindsight.GazeTracking.gaze_pipeline import _estimate_pitchyaw
from mindsight.GazeTracking.gaze_processing import (
    GazeSmootherReID,
    _get_eye_center,
    normalize_face_dicts,
)

UNIFACE_FACE = {
    "bbox": [20.0, 20.0, 60.0, 60.0],
    "confidence": 0.9,
    "landmarks": [[30.0, 30.0], [50.0, 30.0], [40.0, 40.0],
                  [32.0, 50.0], [48.0, 50.0]],
}


# ── normalize_face_dicts ─────────────────────────────────────────────────────

def test_uniface_shape_is_normalized():
    out = normalize_face_dicts([dict(UNIFACE_FACE)])
    f = out[0]
    assert f["bbox"] == [20.0, 20.0, 60.0, 60.0, 0.9]
    assert f["kps"] == UNIFACE_FACE["landmarks"]
    assert _get_eye_center(f) is not None


def test_inverse_scale_applied_to_bbox_and_kps():
    out = normalize_face_dicts([dict(UNIFACE_FACE)], inv_scale=2.0)
    f = out[0]
    assert f["bbox"][:4] == [40.0, 40.0, 120.0, 120.0]
    assert f["bbox"][4] == 0.9                 # score is NOT scaled
    assert f["kps"][0] == [60.0, 60.0]


def test_already_normalized_dict_passes_through():
    pre = {"bbox": [1.0, 2.0, 3.0, 4.0, 0.5], "kps": [[1.0, 1.0]] * 5}
    f = normalize_face_dicts([pre])[0]
    assert f["bbox"] == [1.0, 2.0, 3.0, 4.0, 0.5]
    assert f["kps"] == [[1.0, 1.0]] * 5


def test_no_landmarks_no_score_stays_bare():
    f = normalize_face_dicts([{"bbox": [1, 2, 3, 4]}])[0]
    assert f["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert f["kps"] is None


# ── --face-eye-origin gating ─────────────────────────────────────────────────

class _StubGaze:
    def estimate(self, crop):
        return (0.1, 0.2, 0.9)


def _run_estimate(face, eye_origin):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    raw_faces, *_rest = _estimate_pitchyaw(
        frame, [face], _StubGaze(), smoother=None, eye_origin=eye_origin)
    return raw_faces[0][0]                     # origin


def test_origin_stays_bbox_center_when_flag_off():
    face = normalize_face_dicts([dict(UNIFACE_FACE)])[0]
    assert np.allclose(_run_estimate(face, eye_origin=False), [40.0, 40.0])


def test_origin_moves_to_eye_midpoint_when_flag_on():
    face = normalize_face_dicts([dict(UNIFACE_FACE)])[0]
    assert np.allclose(_run_estimate(face, eye_origin=True), [40.0, 30.0])


def test_low_detection_score_falls_back_to_bbox_center():
    weak = dict(UNIFACE_FACE, confidence=0.1)  # below EYE_CONF_THRESH (0.2)
    face = normalize_face_dicts([weak])[0]
    assert np.allclose(_run_estimate(face, eye_origin=True), [40.0, 40.0])


def test_kps_ride_along_for_the_smoother():
    face = normalize_face_dicts([dict(UNIFACE_FACE)])[0]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    raw_faces, *_ = _estimate_pitchyaw(
        frame, [face], _StubGaze(), smoother=None)
    kps_local = raw_faces[0][4]
    assert kps_local[0] == [10.0, 10.0]        # 30,30 minus crop origin 20,20


# ── Embedding-verified revival (--face-reid-sim) ─────────────────────────────

KPS = [[5.0, 5.0]] * 5


def _identity_embed(crop, kps):
    """Deterministic per-'person' unit embedding keyed off crop brightness."""
    if crop is None or kps is None:
        return None
    if int(crop[0, 0, 0]) < 128:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


def _crop(value):
    return np.full((20, 20, 3), value, dtype=np.uint8)


def _smoother(embed_sim, embed_fn=_identity_embed):
    return GazeSmootherReID(max_dist=50, grace_frames=10,
                            embed_fn=embed_fn, embed_sim=embed_sim)


def test_embedding_revives_lost_track_anywhere_in_frame():
    sm = _smoother(embed_sim=0.5)
    (_, _, tid0), = sm.update([(np.array([10.0, 10.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    sm.update([])                              # track goes dead
    (_, _, tid1), = sm.update([(np.array([300.0, 300.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    assert tid1 == tid0                        # far beyond max_dist, same face


def test_without_embeddings_far_reappearance_mints_new_id():
    sm = _smoother(embed_sim=0.0)              # feature off
    (_, _, tid0), = sm.update([(np.array([10.0, 10.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    sm.update([])
    (_, _, tid1), = sm.update([(np.array([300.0, 300.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    assert tid1 != tid0


def test_different_face_is_not_revived_by_embedding():
    sm = _smoother(embed_sim=0.5)
    (_, _, tid0), = sm.update([(np.array([10.0, 10.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    sm.update([])
    (_, _, tid1), = sm.update([(np.array([300.0, 300.0]), 0.0, 0.0,
                                _crop(200), KPS)])   # different identity
    assert tid1 != tid0


def test_positional_revival_still_works_as_fallback():
    sm = _smoother(embed_sim=0.5)
    (_, _, tid0), = sm.update([(np.array([10.0, 10.0]), 0.0, 0.0,
                                _crop(50), KPS)])
    sm.update([])
    # No landmarks -> no embedding for the detection; nearby position
    # still revives through the 1.0 positional path.
    (_, _, tid1), = sm.update([(np.array([12.0, 12.0]), 0.0, 0.0,
                                _crop(50))])
    assert tid1 == tid0


def test_legacy_four_tuple_entries_still_accepted():
    sm = _smoother(embed_sim=0.5)
    (_, _, tid0), = sm.update([(np.array([10.0, 10.0]), 0.0, 0.0, _crop(50))])
    assert tid0 == 0


# ── Flag plumbing ────────────────────────────────────────────────────────────

def test_flags_reach_runtime_configs():
    from mindsight.cli_flags import parse_cli
    from mindsight.pipeline_config import GazeConfig, TrackerConfig

    ns = parse_cli([])
    assert GazeConfig.from_namespace(ns).face_eye_origin is True   # 3.8 default
    assert TrackerConfig.from_namespace(ns).face_reid_sim == 0.0
    assert ns.face_model == 'r34'   # v1.1 3.8 default

    ns = parse_cli(["--face-eye-origin", "--face-reid-sim", "0.4",
                    "--face-model", "r34"])
    assert GazeConfig.from_namespace(ns).face_eye_origin is True
    assert TrackerConfig.from_namespace(ns).face_reid_sim == 0.4
    assert ns.face_model == "r34"
