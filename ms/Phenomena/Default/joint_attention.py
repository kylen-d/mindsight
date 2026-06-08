"""[SP1.5 shim] moved to mindsight.Phenomena.Default.joint_attention; delete in SP1.6."""
import sys
import mindsight.Phenomena.Default.joint_attention
sys.modules[__name__] = mindsight.Phenomena.Default.joint_attention
