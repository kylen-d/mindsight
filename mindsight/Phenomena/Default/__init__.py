"""
Phenomena/Default/ — Built-in phenomena tracker pack.

Exports all default tracker classes so they can be imported as::

    from Phenomena.Default import MutualGazeTracker, ScanpathTracker, ...
"""

from .joint_attention import JointAttentionTracker
from .attention_span import AttentionSpanTracker
from .gaze_aversion import GazeAversionTracker
from .gaze_following import GazeFollowingTracker
from .gaze_leadership import GazeLeadershipTracker
from .mutual_gaze import MutualGazeTracker
from .scanpath import ScanpathTracker
from .social_referencing import SocialReferenceTracker

__all__ = [
    'JointAttentionTracker',
    'MutualGazeTracker',
    'SocialReferenceTracker',
    'GazeFollowingTracker',
    'GazeAversionTracker',
    'ScanpathTracker',
    'GazeLeadershipTracker',
    'AttentionSpanTracker',
]
