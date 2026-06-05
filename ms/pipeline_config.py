"""[SP1.5 shim] moved to mindsight.pipeline_config; delete in SP1.6."""
import sys
import mindsight.pipeline_config
sys.modules[__name__] = mindsight.pipeline_config
