"""Phenomena/Default/gaze_leadership.py — Gaze leadership tracking."""
from collections import deque

import numpy as np

from ms.DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from ms.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


class GazeLeadershipTracker(PhenomenaPlugin):
    """
    Tracks gaze leadership: who first looks at objects that subsequently
    become joint-attention targets receives a leadership credit.

    After a joint-attention event fires, the first-looker record for that
    object class is cleared so credits don't accumulate indefinitely.
    """

    name = "gaze_leader"
    dashboard_panel = "right"
    _COLOUR = (255, 210, 60)

    def __init__(self, tip_mode: bool = False, tip_lag: int = 15):
        # Object-based leadership state
        self._first_look:   dict = {}   # obj_cls -> (face_idx, frame_no)
        self._prev_lookers: dict = {}   # obj_cls -> set of face_idx last frame
        self.lead_counts:   dict = {}   # face_idx -> int
        self._current_leadership: dict = {}
        self._history:      list = []   # [(frame_no, max_lead_credit)]

        # Tip-convergence leadership state
        self._tip_mode = tip_mode
        self._tip_lag  = tip_lag
        self._tip_buffer: dict = {}            # face_idx -> deque of (frame_no, tip_pos)
        self._prev_convergence_sets: list = [] # last frame's convergence frozensets

    def update(self, **kwargs):
        frame_no = kwargs['frame_no']
        hit_events = kwargs.get('hit_events', [])
        joint_objs = kwargs.get('joint_objs', set())
        dets = kwargs.get('dets', [])

        objects = dets
        current: dict = {}
        for ev in hit_events:
            current.setdefault(ev['object'], set()).add(ev['face_idx'])

        # Record first look at objects newly attended (nobody looking last frame)
        for cls, lookers in current.items():
            if not self._prev_lookers.get(cls) and cls not in self._first_look:
                self._first_look[cls] = (min(lookers), frame_no)

        # Award credit for joint-attention events
        awarded_this_frame: set = set()
        for oi in joint_objs:
            if oi < len(objects):
                cls = objects[oi]['class_name']
                if cls in self._first_look:
                    leader_fi, _ = self._first_look.pop(cls)
                    self.lead_counts[leader_fi] = self.lead_counts.get(leader_fi, 0) + 1
                    awarded_this_frame.add(leader_fi)

        self._prev_lookers = current

        # ── Tip-convergence leadership ───────────────────────────────────
        if self._tip_mode:
            persons_gaze = kwargs.get('persons_gaze', [])
            tip_convergences = kwargs.get('tip_convergences', [])
            tip_radius = kwargs.get('tip_radius', 50)
            proximity_thr = tip_radius * 2.0

            # 1. Update tip position buffer for each face
            for fi, (_, ray_end, _) in enumerate(persons_gaze):
                buf = self._tip_buffer.setdefault(fi, deque(maxlen=self._tip_lag))
                buf.append((frame_no, np.asarray(ray_end, float)))

            # 2. Detect NEW convergence clusters (not present last frame)
            current_conv_sets = [faces for faces, _ in tip_convergences]
            for faces, centroid in tip_convergences:
                if faces in self._prev_convergence_sets:
                    continue  # sustained convergence — already credited

                centroid = np.asarray(centroid, float)

                # 3. Find who arrived near the centroid earliest
                earliest_frame: dict = {}
                for fi in faces:
                    buf = self._tip_buffer.get(fi, deque())
                    for fn, tip_pos in buf:
                        if np.linalg.norm(tip_pos - centroid) < proximity_thr:
                            earliest_frame[fi] = fn
                            break  # first (oldest) match in the buffer

                if len(earliest_frame) >= 2:
                    leader_fi = min(earliest_frame, key=earliest_frame.get)
                    if leader_fi not in awarded_this_frame:
                        self.lead_counts[leader_fi] = (
                            self.lead_counts.get(leader_fi, 0) + 1)
                        awarded_this_frame.add(leader_fi)

            self._prev_convergence_sets = current_conv_sets

        self._current_leadership = dict(self.lead_counts)
        max_credit = max(self.lead_counts.values()) if self.lead_counts else 0
        self._history.append((frame_no, max_credit))
        return {'leadership': self._current_leadership}

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        if self._current_leadership:
            for fi, c in sorted(self._current_leadership.items(),
                                key=lambda x: -x[1]):
                rows.append((f"{resolve_display_pid(fi, pid_map)}: {c} events",
                             self._COLOUR))
        else:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "GAZE LEADERSHIP", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        if self._current_leadership:
            for fi, c in sorted(self._current_leadership.items(),
                                key=lambda x: -x[1]):
                rows.append({
                    "label": f"{resolve_display_pid(fi, pid_map)}",
                    "value": f"{c} events",
                })
        return {
            "title": "GAZE LEADERSHIP",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        if not self.lead_counts:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        for fi, cnt in sorted(self.lead_counts.items()):
            rows.append(["gaze_leadership", resolve_display_pid(fi, pid_map),
                         "", cnt, total_frames, ""])
        return rows

    def console_summary(self, total_frames, *, pid_map=None):
        if not self.lead_counts:
            return None
        formatted = dict(sorted(
            (resolve_display_pid(fi, pid_map), c)
            for fi, c in self.lead_counts.items()))
        return f"Gaze leadership counts: {formatted}"

    def time_series_data(self):
        if not self._history:
            return {}
        return {'gaze_leadership_max': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Max leadership credit',
            'chart_type': 'step',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(max(self.lead_counts.values())) if self.lead_counts else 0.0

    def latest_metrics(self):
        result = {}
        for fi, c in sorted(self.lead_counts.items(), key=lambda x: -x[1]):
            plbl = resolve_display_pid(fi)
            result[plbl] = {
                'value': float(c),
                'label': f'{plbl}',
                'y_label': 'events',
            }
        return result or None

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Gaze Leadership CLI flags to the argument parser."""
        g = parser.add_argument_group("Gaze Leadership")
        g.add_argument(
            "--gaze-leader",
            action="store_true",
            help="Enable gaze leadership tracking.",
        )
        g.add_argument(
            "--gaze-leader-tips",
            action="store_true",
            help="Also detect leadership via gaze-tip convergence "
                 "(requires --gaze-tips).",
        )
        g.add_argument(
            "--gaze-leader-tip-lag",
            type=int, default=15, metavar="N",
            help="Lookback frames for tip-arrival priority (default: 15).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--gaze-leader`` was passed, else ``None``."""
        if not (getattr(args, "gaze_leader", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls(
            tip_mode=getattr(args, "gaze_leader_tips", False),
            tip_lag=getattr(args, "gaze_leader_tip_lag", 15),
        )
