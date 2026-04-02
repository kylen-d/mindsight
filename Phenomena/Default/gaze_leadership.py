"""Phenomena/Default/gaze_leadership.py — Gaze leadership tracking."""
from Plugins import PhenomenaPlugin
from DataCollection.dashboard_output import _draw_panel_section, _DASH_DIM


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

    def __init__(self):
        self._first_look:   dict = {}   # obj_cls -> (face_idx, frame_no)
        self._prev_lookers: dict = {}   # obj_cls -> set of face_idx last frame
        self.lead_counts:   dict = {}   # face_idx -> int
        self._current_leadership: dict = {}

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
        for oi in joint_objs:
            if oi < len(objects):
                cls = objects[oi]['class_name']
                if cls in self._first_look:
                    leader_fi, _ = self._first_look.pop(cls)
                    self.lead_counts[leader_fi] = self.lead_counts.get(leader_fi, 0) + 1

        self._prev_lookers = current
        self._current_leadership = dict(self.lead_counts)
        return {'leadership': self._current_leadership}

    def dashboard_section(self, panel, y, line_h):
        rows = []
        if self._current_leadership:
            for fi, c in sorted(self._current_leadership.items(),
                                key=lambda x: -x[1]):
                rows.append((f"P{fi}: {c} events", self._COLOUR))
        else:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "GAZE LEADERSHIP", self._COLOUR, rows, line_h)

    def csv_rows(self, total_frames):
        if not self.lead_counts:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        for fi, cnt in sorted(self.lead_counts.items()):
            rows.append(["gaze_leadership", f"P{fi}", "",
                         cnt, total_frames, ""])
        return rows

    def console_summary(self, total_frames):
        if not self.lead_counts:
            return None
        formatted = dict(sorted(
            (f"P{fi}", c) for fi, c in self.lead_counts.items()))
        return f"Gaze leadership counts: {formatted}"

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

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--gaze-leader`` was passed, else ``None``."""
        if not (getattr(args, "gaze_leader", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls()
