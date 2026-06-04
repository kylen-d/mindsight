"""
RayForming/fixation.py — Fixation lock-on tracker.

Moved from ``GazeTracking/gaze_processing.py``.  Snaps gaze ray to an
object after sustained dwell (dwell_frames of attention within lock_dist).
"""
from __future__ import annotations

import numpy as np

from mindsight.utils.geometry import bbox_center


class GazeLockTracker:
    """Fixation lock-on: snaps gaze ray to object after dwell_frames of sustained attention."""

    def __init__(self, dwell_frames=15, release_frames=10, lock_dist=100, max_face_dist=120):
        self.dwell, self.release           = dwell_frames, release_frames
        self.lock_dist, self.max_face_dist = lock_dist, max_face_dist
        self._tracks, self._nid            = {}, 0

    @staticmethod
    def _ray_pt_dist(origin, udir, pt):
        v = pt - origin
        t = float(np.dot(v, udir))
        return float(np.linalg.norm(v if t < 0 else pt - (origin + t * udir)))

    def _find_track(self, center):
        if not self._tracks:
            return None
        bid, bd = min(
            ((tid, float(np.linalg.norm(center - t['c']))) for tid, t in self._tracks.items()),
            key=lambda x: x[1])
        return bid if bd < self.max_face_dist else None

    def update(self, persons_gaze, objects):
        used, results = set(), []
        for origin, ray_end, _ in persons_gaze:
            c    = np.asarray(origin, float)
            dv   = np.asarray(ray_end, float) - c
            dl   = np.linalg.norm(dv)
            udir = dv / dl if dl > 1e-6 else np.array([0., 1.])

            tid = self._find_track(c)
            if tid is None:
                tid = self._nid; self._nid += 1
                self._tracks[tid] = dict(c=c.copy(), dwell={}, locked=None, rc=0)
            t = self._tracks[tid]
            t['c'] = c.copy()
            used.add(tid)

            obj_ctrs = [bbox_center(o) for o in objects]
            near = {oi for oi, ctr in enumerate(obj_ctrs)
                    if self._ray_pt_dist(c, udir, ctr) < self.lock_dist}

            for oi in list(t['dwell']):
                t['dwell'][oi] = t['dwell'][oi] + 1 if oi in near else max(0, t['dwell'][oi] - 2)
                if t['dwell'][oi] == 0:
                    del t['dwell'][oi]
            for oi in near - set(t['dwell']):
                t['dwell'][oi] = 1

            locked = t['locked']
            if locked is not None:
                if locked in near:
                    t['rc'] = 0
                else:
                    t['rc'] += 1
                    if t['rc'] >= self.release:
                        t['locked'] = None; t['rc'] = 0; locked = None

            if locked is None:
                best = max(
                    ((oi, cnt) for oi, cnt in t['dwell'].items() if cnt >= self.dwell),
                    key=lambda x: x[1], default=(None, 0))[0]
                if best is not None:
                    t['locked'] = locked = best; t['rc'] = 0

            frac = min(max(t['dwell'].values(), default=0) / self.dwell, 1.0)
            if locked is not None and locked < len(objects):
                obj     = objects[locked]
                snapped = bbox_center(obj)
                results.append((snapped, locked, frac))
            else:
                results.append((np.asarray(ray_end, float), None, frac))

        for tid in list(self._tracks):
            if tid not in used:
                del self._tracks[tid]
        return results


def apply_lock_on(persons_gaze, locker, objects):
    """Apply fixation lock-on and return updated persons_gaze and lock_info.

    Parameters
    ----------
    persons_gaze : list of (origin, ray_end, angles)
    locker       : GazeLockTracker instance or None
    objects      : non-person detection list

    Returns
    -------
    persons_gaze : list of (origin, ray_end, angles) -- ray_end snapped when locked
    lock_info    : list of (obj_idx_or_None, frac)
    """
    lock_info = [(None, 0.0)] * len(persons_gaze)
    if locker and persons_gaze:
        lr = locker.update(persons_gaze, objects)
        persons_gaze = [(o, se, a) for (o, _, a), (se, _, _) in zip(persons_gaze, lr)]
        lock_info    = [(oi, frac) for (_, oi, frac) in lr]
    return persons_gaze, lock_info
