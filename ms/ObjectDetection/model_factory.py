"""[SP1.5 shim] moved to mindsight.ObjectDetection.model_factory; delete in SP1.6."""
import sys
import mindsight.ObjectDetection.model_factory
sys.modules[__name__] = mindsight.ObjectDetection.model_factory
