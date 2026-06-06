"""[SP1.5 shim] moved to mindsight.ObjectDetection.detection; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.detection
sys.modules[__name__] = mindsight.ObjectDetection.detection
