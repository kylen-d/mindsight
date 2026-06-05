"""[SP1.5 shim] moved to mindsight.participant_ids; delete in SP1.6."""
import sys
import mindsight.participant_ids
sys.modules[__name__] = mindsight.participant_ids
