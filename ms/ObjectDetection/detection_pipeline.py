"""[SP1.5 shim] moved to mindsight.ObjectDetection.detection_pipeline; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.detection_pipeline
sys.modules[__name__] = mindsight.ObjectDetection.detection_pipeline
