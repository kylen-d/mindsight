"""[SP1.5 shim] moved to mindsight.GUI.pipeline_dialog; delete in SP1.6."""
import sys
import mindsight.GUI.pipeline_dialog
sys.modules[__name__] = mindsight.GUI.pipeline_dialog
