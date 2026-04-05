"""Tests for ObjectDetection/object_detection.py -- ObjectPersistenceCache."""

import pytest

from ms.ObjectDetection.detection import Detection
from ms.ObjectDetection.object_detection import ObjectPersistenceCache


def _det(cls="person", x1=0, y1=0, x2=100, y2=100, conf=0.9):
    return Detection(
        class_name=cls, cls_id=0, conf=conf,
        x1=x1, y1=y1, x2=x2, y2=y2,
    )


class TestObjectPersistenceCache:

    def test_fresh_detections_returned(self):
        cache = ObjectPersistenceCache(max_age=5)
        dets = [_det("person", 10, 10, 50, 50)]
        result = cache.update(dets)
        assert len(result) == 1
        assert result[0].ghost is False

    def test_ghost_appears_when_detection_disappears(self):
        cache = ObjectPersistenceCache(max_age=5)
        cache.update([_det("person", 10, 10, 50, 50)])
        # Object disappears
        result = cache.update([])
        assert len(result) == 1
        assert result[0].ghost is True

    def test_ghost_expires_after_max_age(self):
        cache = ObjectPersistenceCache(max_age=3)
        cache.update([_det("person", 10, 10, 50, 50)])
        for _ in range(4):
            result = cache.update([])
        assert len(result) == 0

    def test_matching_by_iou_and_class(self):
        cache = ObjectPersistenceCache(max_age=5, iou_threshold=0.3)
        cache.update([_det("person", 10, 10, 110, 110)])
        # Same class, overlapping bbox
        result = cache.update([_det("person", 15, 15, 115, 115)])
        assert len(result) == 1
        assert result[0].ghost is False
        assert result[0].x1 == 15  # updated to new position

    def test_different_class_not_matched(self):
        cache = ObjectPersistenceCache(max_age=5, iou_threshold=0.3)
        cache.update([_det("person", 10, 10, 110, 110)])
        # Different class at same position
        result = cache.update([_det("car", 10, 10, 110, 110)])
        # Should have ghost person + fresh car
        assert len(result) == 2

    def test_no_overlap_creates_new_slot(self):
        cache = ObjectPersistenceCache(max_age=5, iou_threshold=0.3)
        cache.update([_det("person", 10, 10, 50, 50)])
        # No overlap
        result = cache.update([_det("person", 500, 500, 600, 600)])
        # Ghost old + fresh new
        assert len(result) == 2

    def test_empty_input(self):
        cache = ObjectPersistenceCache()
        result = cache.update([])
        assert result == []

    def test_iou_computation(self):
        a = _det(x1=0, y1=0, x2=100, y2=100)
        b = _det(x1=50, y1=50, x2=150, y2=150)
        iou = ObjectPersistenceCache._iou(a, b)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert iou == pytest.approx(2500 / 17500, abs=0.01)

    def test_iou_no_overlap(self):
        a = _det(x1=0, y1=0, x2=50, y2=50)
        b = _det(x1=100, y1=100, x2=200, y2=200)
        assert ObjectPersistenceCache._iou(a, b) == 0.0

    def test_iou_perfect_overlap(self):
        a = _det(x1=10, y1=10, x2=50, y2=50)
        b = _det(x1=10, y1=10, x2=50, y2=50)
        assert ObjectPersistenceCache._iou(a, b) == pytest.approx(1.0)

    def test_reappearance_revives_slot(self):
        cache = ObjectPersistenceCache(max_age=5, iou_threshold=0.3)
        cache.update([_det("person", 10, 10, 110, 110)])
        # Disappear for 2 frames
        cache.update([])
        cache.update([])
        # Reappear at same location
        result = cache.update([_det("person", 10, 10, 110, 110)])
        # Should match the ghost slot and revive it
        assert len(result) == 1
        assert result[0].ghost is False
