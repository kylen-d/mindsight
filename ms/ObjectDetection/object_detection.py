"""[SP1.5 shim] moved to mindsight.ObjectDetection.object_detection; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.object_detection
sys.modules[__name__] = mindsight.ObjectDetection.object_detection
