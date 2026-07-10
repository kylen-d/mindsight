"""
Phenomena/helpers.py — Shared helper functions for phenomena tracking.

Joint-attention computation (hard and soft quorum) and gaze-convergence
clustering.  Used by the pipeline and by individual phenomenon trackers.
"""

import numpy as np


def joint_attention(persons_gaze, hits, quorum: float = 1.0) -> set:
    """
    Compute joint attention with a soft quorum: an object is flagged as
    joint attention if >= ceil(quorum * n_faces) distinct faces look at it,
    with a minimum of 2.

    quorum=1.0 reproduces strict all-faces-must-look behaviour.
    quorum=0.75 with 4 people requires 3/4.
    """
    if len(persons_gaze) < 2:
        return set()
    n_faces      = len(persons_gaze)
    min_watchers = max(2, int(np.ceil(quorum * n_faces)))
    watchers     = {}
    for fi, oi in hits:
        watchers.setdefault(oi, set()).add(fi)
    return {oi for oi, w in watchers.items() if len(w) >= min_watchers}


class EpisodeLog:
    """Tiny append-only recorder of phenomenon episodes.

    An *episode* is a contiguous stretch during which a phenomenon holds
    (e.g. a mutual-gaze pair being active, or a single point event).  Each
    tracker owns one ``EpisodeLog``; the writer collects every tracker's
    closed episodes into the merged ``{stem}_phenomena_events.csv``.

    Episodes are keyed by any hashable ``key`` so a tracker can juggle many
    concurrent episodes (one per pair / object / face-set).  A closed episode
    is a dict ``{phenomenon, participant, partner, object, frame_start,
    frame_end}``.  Dependency-free by design.
    """

    def __init__(self) -> None:
        self.rows: list = []      # closed episodes, in close order
        self._open: dict = {}     # key -> partial episode dict

    def open(self, key, *, phenomenon, participant, partner, object,
             frame_start) -> None:
        """Start an episode for *key* (no-op if one is already open)."""
        if key in self._open:
            return
        self._open[key] = {
            "phenomenon": phenomenon, "participant": participant,
            "partner": partner, "object": object,
            "frame_start": frame_start, "frame_end": frame_start,
        }

    def close(self, key, frame_end) -> None:
        """Close the episode for *key*, stamping *frame_end* (no-op if none)."""
        ep = self._open.pop(key, None)
        if ep is None:
            return
        ep["frame_end"] = frame_end
        self.rows.append(ep)

    def close_all(self, frame_end) -> None:
        """Close every still-open episode at *frame_end* (run-end flush)."""
        for key in list(self._open):
            self.close(key, frame_end)

    def is_open(self, key) -> bool:
        return key in self._open


def gaze_convergence(persons_gaze, tip_radius):
    """Cluster gaze-ray tips that are within 2*tip_radius of each other.

    Returns a list of (frozenset_of_face_indices, centroid_array) for each
    cluster of 2+ converging tips.
    """
    if len(persons_gaze) < 2:
        return []
    tips = [np.asarray(re, float) for _, re, _ in persons_gaze]
    thr  = tip_radius * 2.0

    # Vectorized pairwise distance computation
    tips_arr = np.array(tips)                                   # (N, 2)
    diff = tips_arr[:, None, :] - tips_arr[None, :, :]          # (N, N, 2)
    dists = np.linalg.norm(diff, axis=2)                        # (N, N)
    np.fill_diagonal(dists, np.inf)                             # exclude self
    close = dists < thr                                         # (N, N) bool
    adj = [set(np.nonzero(close[i])[0]) for i in range(len(tips))]

    visited, clusters = set(), []
    for s in range(len(tips)):
        if s in visited:
            continue
        q, cl = [s], set()
        while q:
            n = q.pop(0)
            if n in visited:
                continue
            visited.add(n); cl.add(n); q.extend(adj[n] - visited)
        if len(cl) >= 2:
            clusters.append((frozenset(cl), np.mean(tips_arr[list(cl)], axis=0)))
    return clusters
