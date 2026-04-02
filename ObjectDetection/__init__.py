"""
ObjectDetection — YOLO-based object detection pipeline stage.

Provides the detection pipeline step, object persistence cache, model
factories for YOLO and RetinaFace backends, and the Detection dataclass.
"""

from .detection_pipeline import run_detection_step
from .object_detection import YOLOEVPDetector, ObjectPersistenceCache
from .detection import Detection
from .model_factory import create_yolo_detector, create_face_detector

__all__ = [
    "run_detection_step",
    "YOLOEVPDetector",
    "ObjectPersistenceCache",
    "Detection",
    "create_yolo_detector",
    "create_face_detector",
]
