"""[SP1.5 shim] moved to mindsight.utils.mediapipe_face; delete in SP1.6."""
import sys
import mindsight.utils.mediapipe_face
sys.modules[__name__] = mindsight.utils.mediapipe_face
