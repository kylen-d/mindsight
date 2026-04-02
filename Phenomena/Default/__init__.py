"""
Phenomena/Default/ — Built-in phenomena tracker pack.

Exports all default tracker classes so they can be imported as::

    from Phenomena.Default import MutualGazeTracker, ScanpathTracker, ...
    from Phenomena.Default import JointAttentionTemporalTracker
"""

from .joint_attention import JointAttentionTemporalTracker
from .mutual_gaze import MutualGazeTracker
from .social_referencing import SocialReferenceTracker
from .gaze_following import GazeFollowingTracker
from .gaze_aversion import GazeAversionTracker
from .scanpath import ScanpathTracker
from .gaze_leadership import GazeLeadershipTracker
from .attention_span import AttentionSpanTracker

__all__ = [
    'JointAttentionTemporalTracker',
    'MutualGazeTracker',
    'SocialReferenceTracker',
    'GazeFollowingTracker',
    'GazeAversionTracker',
    'ScanpathTracker',
    'GazeLeadershipTracker',
    'AttentionSpanTracker',
]
