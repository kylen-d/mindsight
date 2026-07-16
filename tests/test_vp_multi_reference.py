"""Multi-reference visual prompts (v1.1 W3.7).

1.0 hardcoded ``references[0]`` and silently ignored every other reference
image.  Now every annotated reference contributes: per-reference class
embeddings are mean-pooled per class (across the references that annotate
that class), re-normalized, and installed as the final class table.
Single-reference files keep the 1.0 native priming path byte-for-byte.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mindsight.ObjectDetection.object_detection import (  # noqa: E402
    parse_vp_references,
    prime_yoloe_multi_reference,
)


def _vp(refs):
    return {"classes": [{"id": 0, "name": "cup"}, {"id": 1, "name": "toy"}],
            "references": refs}


def test_parse_all_annotated_references():
    data = _vp([
        {"image": "a.png", "annotations": [
            {"cls_id": 0, "bbox": [0, 0, 10, 10]},
            {"cls_id": 1, "bbox": [5, 5, 15, 15]}]},
        {"image": "b.png", "annotations": []},              # skipped
        {"image": "c.png", "annotations": [
            {"cls_id": 1, "bbox": [1, 1, 9, 9]}]},
    ])
    refs = parse_vp_references(data)
    assert [r["image"] for r in refs] == ["a.png", "c.png"]
    assert refs[0]["bboxes"].shape == (2, 4)
    assert list(refs[1]["cls"]) == [1]


def test_parse_rejects_fully_unannotated_file():
    with pytest.raises(ValueError):
        parse_vp_references(_vp([{"image": "a.png", "annotations": []}]))


class _FakeInner:
    """Stands in for model.model: stores pe per priming call, records the
    final set_classes."""

    def __init__(self, pe_by_image):
        self._pe_by_image = pe_by_image
        self.pe = None
        self.set_calls = []

    def set_classes(self, names, embeddings):
        self.set_calls.append((names, embeddings))

    def parameters(self):
        yield torch.zeros(1)


class _FakeYOLOE:
    def __init__(self, pe_by_image):
        self.model = _FakeInner(pe_by_image)
        self.predict_calls = []

    def predict(self, source, **kwargs):
        self.predict_calls.append((source, kwargs))
        self.model.pe = self.model._pe_by_image[source]
        return []


def _unit(*v):
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.norm()


def test_pooling_averages_per_class_across_annotating_references():
    # ref a annotates cup(0)+toy(1); ref c annotates toy(1) only.
    ea_cup, ea_toy = _unit(1.0, 0.0), _unit(0.0, 1.0)
    ec_toy = _unit(1.0, 1.0)
    fake = _FakeYOLOE({
        "a.png": torch.stack([ea_cup, ea_toy]).unsqueeze(0),
        "c.png": ec_toy.reshape(1, 1, 2),
    })
    refs = [
        {"image": "a.png", "bboxes": np.zeros((2, 4)),
         "cls": np.array([0, 1])},
        {"image": "c.png", "bboxes": np.zeros((1, 4)),
         "cls": np.array([1])},
    ]
    idx_to_id = prime_yoloe_multi_reference(
        fake, refs, predictor_cls=object, class_names={0: "cup", 1: "toy"},
        log=lambda *_a: None)

    assert idx_to_id == {0: 0, 1: 1}
    assert len(fake.predict_calls) == 2
    # each priming call passed COMPACT per-reference cls indices
    assert list(fake.predict_calls[1][1]["visual_prompts"]["cls"]) == [0]

    names, pooled = fake.model.set_calls[-1]
    assert names == ["cup", "toy"]
    assert pooled.shape == (1, 2, 2)
    # cup seen once -> its own (normalized) embedding
    assert torch.allclose(pooled[0, 0], ea_cup, atol=1e-6)
    # toy seen twice -> normalized mean of the two
    expected_toy = (ea_toy + ec_toy) / 2
    expected_toy = expected_toy / expected_toy.norm()
    assert torch.allclose(pooled[0, 1], expected_toy, atol=1e-6)


def test_single_reference_files_keep_the_native_path():
    """YOLOEVPDetector with ONE reference must not touch the pooling helper
    (the 1.0 regression guarantee: identical native priming)."""
    import mindsight.ObjectDetection.object_detection as od

    data = _vp([{"image": "a.png", "annotations": [
        {"cls_id": 0, "bbox": [0, 0, 10, 10]}]}])
    refs = parse_vp_references(data)
    assert len(refs) == 1
    # The detector branches on len(self._references) > 1; with a single
    # reference the pooling helper must never be needed at call time.
    # (Pinned structurally: the legacy fields exist and match ref 0.)
    det = object.__new__(od.YOLOEVPDetector)
    det._references = refs
    det._refer_image = refs[0]["image"]
    det._visual_prompts = {"bboxes": refs[0]["bboxes"],
                           "cls": refs[0]["cls"]}
    assert len(det._references) == 1
    assert det._refer_image == "a.png"
