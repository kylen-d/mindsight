"""Phenomena/Default/social_referencing.py — Social referencing detection."""
import numpy as np

from DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin
from utils.geometry import extend_ray, ray_hits_box


class SocialReferenceTracker(PhenomenaPlugin):
    """
    Detects social referencing: a person looks at another person's face and
    then within `window_frames` redirects their gaze to an object.

    State machine per face: NEUTRAL -> FACE_LOOK (onset recorded) -> REF_COMPLETE
    """

    name = "social_ref"
    dashboard_panel = "left"
    _COLOUR = (100, 255, 130)

    def __init__(self, window_frames: int = 60):
        self.window    = window_frames
        self._state:   dict = {}   # face_idx -> {'frame': int, 'targets': set}
        self.event_log: list = []
        self._current_events: list = []
        self._history:  list = []   # [(frame_no, cumulative_event_count)]

    def update(self, **kwargs):
        frame_no = kwargs['frame_no']
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])
        dets = kwargs.get('dets', [])
        hits_set = kwargs.get('hits') or set()

        detect_extend = kwargs.get('detect_extend', 0.0)
        scope = kwargs.get('detect_extend_scope', 'objects')
        use_extend = detect_extend > 0 and scope in ('phenomena', 'both')

        n = min(len(persons_gaze), len(face_bboxes))

        # Pre-compute ray endpoints once per person (O(N) instead of O(N^2))
        origins = []
        endpoints = []
        for i in range(n):
            oi_pt, rei, _ = persons_gaze[i]
            origins.append(oi_pt)
            if use_extend:
                endpoints.append(extend_ray(oi_pt, rei, length=float(np.linalg.norm(np.asarray(rei) - np.asarray(oi_pt))) + detect_extend))
            else:
                endpoints.append(rei)

        face_lookers: dict = {}   # viewer_fi -> set of looked-at face indices
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x1, y1, x2, y2 = face_bboxes[j]
                if ray_hits_box(origins[i], endpoints[i], x1, y1, x2, y2):
                    face_lookers.setdefault(i, set()).add(j)

        obj_lookers = {fi for fi, _ in hits_set}
        events = []

        for fi in range(len(persons_gaze)):
            looking_at_face = fi in face_lookers
            looking_at_obj  = fi in obj_lookers
            state = self._state.get(fi)

            if looking_at_face:
                self._state[fi] = {'frame': frame_no, 'targets': face_lookers[fi]}
            elif looking_at_obj and state is not None:
                if frame_no - state['frame'] <= self.window:
                    obj_names = [dets[oi]['class_name']
                                 for fi2, oi in hits_set if fi2 == fi and oi < len(dets)]
                    ev = {'face_idx': fi,
                          'prior_face_targets': sorted(state['targets']),
                          'object_names': obj_names,
                          'frame': frame_no}
                    events.append(ev)
                    self.event_log.append(ev)
                self._state.pop(fi, None)
            elif not looking_at_face and state is not None:
                if frame_no - state['frame'] > self.window:
                    self._state.pop(fi, None)

        self._current_events = events
        self._history.append((frame_no, len(self.event_log)))
        return {'events': events}

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        for ev in self.event_log[-3:]:
            pf = "+".join(resolve_display_pid(x, pid_map)
                          for x in ev['prior_face_targets'])
            ob = ",".join(ev['object_names']) or "?"
            plbl = resolve_display_pid(ev['face_idx'], pid_map)
            rows.append((f"{plbl} [{pf}]\u2192{ob}", self._COLOUR))
        if not rows:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "SOCIAL REFERENCE", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for ev in self.event_log[-3:]:
            pf = "+".join(resolve_display_pid(x, pid_map)
                          for x in ev['prior_face_targets'])
            ob = ",".join(ev['object_names']) or "?"
            plbl = resolve_display_pid(ev['face_idx'], pid_map)
            rows.append({"label": f"{plbl} [{pf}]\u2192{ob}"})
        return {
            "title": "SOCIAL REFERENCE",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        if not self.event_log:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        counts: dict = {}
        for ev in self.event_log:
            counts[ev['face_idx']] = counts.get(ev['face_idx'], 0) + 1
        for fi, cnt in sorted(counts.items()):
            rows.append(["social_reference", resolve_display_pid(fi, pid_map),
                         "", cnt, total_frames, ""])
        return rows

    def console_summary(self, total_frames, *, pid_map=None):
        if not self.event_log:
            return None
        return f"Social reference events: {len(self.event_log)}"

    def time_series_data(self):
        if not self._history:
            return {}
        return {'social_ref_events': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Cumulative reference events',
            'chart_type': 'step',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(len(self.event_log))

    def latest_metrics(self):
        return {'events': {
            'value': float(len(self.event_log)),
            'label': 'Cumulative reference events',
            'y_label': 'events',
        }}

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Social Referencing CLI flags to the argument parser."""
        g = parser.add_argument_group("Social Referencing")
        g.add_argument(
            "--social-ref",
            action="store_true",
            help="Enable social referencing detection.",
        )
        g.add_argument(
            "--social-ref-window",
            type=int, default=60,
            help="Max frames between face-look and object-look to count as social referencing (default: 60).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--social-ref`` was passed, else ``None``."""
        if not (getattr(args, "social_ref", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls(window_frames=args.social_ref_window)
