"""Phenomena/Default/gaze_following.py — Gaze following / attention cueing detection."""
import numpy as np

from ms.DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from ms.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


class GazeFollowingTracker(PhenomenaPlugin):
    """
    Detects gaze following / attention cueing: person A begins looking at an
    object and within `lag_frames` person B also begins looking at that object.

    Only NEW look acquisitions (not sustained co-attendance) are counted.
    """

    name = "gaze_follow"
    dashboard_panel = "left"
    _COLOUR = (80, 200, 255)

    def __init__(self, lag_frames: int = 30):
        self.lag           = lag_frames
        self._prev_targets: dict = {}   # face_idx -> set of obj_idx
        self._shifts:       list = []   # pending gaze-shift records
        self.event_log:     list = []
        self._current_events: list = []
        self._history:      list = []   # [(frame_no, cumulative_event_count)]

    def update(self, **kwargs):
        frame_no = kwargs['frame_no']
        hits_set = kwargs.get('hits') or set()

        current: dict = {}
        for fi, oi in hits_set:
            current.setdefault(fi, set()).add(oi)

        # Expire old pending shifts
        self._shifts = [s for s in self._shifts
                        if frame_no - s['frame'] <= self.lag]

        events = []
        new_acqs = []
        for fi, objs in current.items():
            for oi in objs - self._prev_targets.get(fi, set()):
                new_acqs.append((fi, oi))

        for fi, oi in new_acqs:
            for s in self._shifts:
                if s['leader'] != fi and s['obj'] == oi:
                    ev = {'leader': s['leader'], 'follower': fi,
                          'obj_idx': oi, 'lag_frames': frame_no - s['frame'],
                          'frame': frame_no}
                    events.append(ev)
                    self.event_log.append(ev)
            self._shifts.append({'leader': fi, 'obj': oi, 'frame': frame_no})

        self._prev_targets = current
        self._current_events = events
        self._history.append((frame_no, len(self.event_log)))
        return {'events': events}

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        for ev in self.event_log[-3:]:
            fl = resolve_display_pid(ev['follower'], pid_map)
            ld = resolve_display_pid(ev['leader'], pid_map)
            rows.append((f"{fl}\u2190{ld}  lag={ev['lag_frames']}f", self._COLOUR))
        if not rows:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "GAZE FOLLOWING", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for ev in self.event_log[-3:]:
            fl = resolve_display_pid(ev['follower'], pid_map)
            ld = resolve_display_pid(ev['leader'], pid_map)
            rows.append({"label": f"{fl}\u2190{ld}  lag={ev['lag_frames']}f"})
        return {
            "title": "GAZE FOLLOWING",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        if not self.event_log:
            return []
        rows = [["category", "leader", "follower",
                 "event_count", "total_frames", "avg_lag_frames"]]
        pair_evts: dict = {}
        for ev in self.event_log:
            k = (ev['leader'], ev['follower'])
            pair_evts.setdefault(k, []).append(ev['lag_frames'])
        for (leader, follower), lags in sorted(pair_evts.items()):
            rows.append(["gaze_following",
                         resolve_display_pid(leader, pid_map),
                         resolve_display_pid(follower, pid_map),
                         len(lags), total_frames, f"{np.mean(lags):.1f}"])
        return rows

    def console_summary(self, total_frames, *, pid_map=None):
        if not self.event_log:
            return None
        return f"Gaze following events: {len(self.event_log)}"

    def time_series_data(self):
        if not self._history:
            return {}
        return {'gaze_follow_events': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Cumulative follow events',
            'chart_type': 'step',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(len(self.event_log))

    def latest_metrics(self):
        return {'events': {
            'value': float(len(self.event_log)),
            'label': 'Cumulative follow events',
            'y_label': 'events',
        }}

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Gaze Following CLI flags to the argument parser."""
        g = parser.add_argument_group("Gaze Following")
        g.add_argument(
            "--gaze-follow",
            action="store_true",
            help="Enable gaze following / attention cueing detection.",
        )
        g.add_argument(
            "--gaze-follow-lag",
            type=int, default=30,
            help="Max lag in frames for a follower to match a leader's gaze shift (default: 30).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--gaze-follow`` was passed, else ``None``."""
        if not (getattr(args, "gaze_follow", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls(lag_frames=args.gaze_follow_lag)
