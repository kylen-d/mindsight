"""Phenomena/Default/joint_attention.py — Temporal joint-attention confirmation."""
import collections


class JointAttentionTemporalTracker:
    """
    Sliding-window filter for joint-attention events.

    Wraps the raw per-frame joint_objs set and returns only those object indices
    that have appeared in joint attention in >= threshold fraction of the last
    `window` frames.  This eliminates single-frame false positives caused by
    momentary gaze jitter.

    Usage
    -----
    tracker = JointAttentionTemporalTracker(window=20, threshold=0.70)
    confirmed = tracker.update(raw_joint_objs_this_frame)  # set of obj indices
    """

    def __init__(self, window: int, threshold: float = 0.70):
        self.window    = max(1, window)
        self.threshold = threshold
        self._history: collections.deque = collections.deque(maxlen=self.window)

    def update(self, joint_objs_this_frame: set) -> set:
        """Push current frame; return the temporally-confirmed joint-attention set."""
        self._history.append(frozenset(joint_objs_this_frame))
        n = len(self._history)
        counts = collections.Counter(oi for s in self._history for oi in s)
        return {oi for oi, cnt in counts.items() if cnt / n >= self.threshold}

    @property
    def fill(self) -> float:
        """Fraction of the window that has been populated (0–1)."""
        return len(self._history) / self.window
