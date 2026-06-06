"""[SP1.5 shim] moved to mindsight.ObjectDetection.YOLO.yolo_tracking; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.YOLO.yolo_tracking
sys.modules[__name__] = mindsight.ObjectDetection.YOLO.yolo_tracking
