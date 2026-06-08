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
