"""[SP1.5 shim] moved to mindsight.ObjectDetection.YOLO.yoloe_folder; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.YOLO.yoloe_folder
sys.modules[__name__] = mindsight.ObjectDetection.YOLO.yoloe_folder
