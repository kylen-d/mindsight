"""Phenomena/Default/gaze_aversion.py — Gaze aversion detection."""
from DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


class GazeAversionTracker(PhenomenaPlugin):
    """
    Detects gaze aversion: a person consistently fails to look at a visible,
    salient object for >= `window_frames` consecutive frames.

    Uses (face_idx, class_name) keys to survive per-frame object-index churn.
    """

    name = "gaze_aversion"
    dashboard_panel = "right"
    _COLOUR = (60, 60, 230)

    def __init__(self, window_frames: int = 60, min_obj_conf: float = 0.5):
        self.window   = window_frames
        self.min_conf = min_obj_conf
        self._no_look: dict = {}   # (face_idx, class_name) -> consec frames without look
        self._current_aversions: set = set()
        self._history: list = []   # [(frame_no, n_active_aversions)]

    def update(self, **kwargs):
        persons_gaze = kwargs.get('persons_gaze', [])
        dets = kwargs.get('dets', [])
        hits_set = kwargs.get('hits') or set()

        objects = dets
        n_faces_local = len(persons_gaze)
        looking: dict = {}   # face_idx -> set of class_names currently looked at
        for fi, oi in hits_set:
            if oi < len(objects):
                looking.setdefault(fi, set()).add(objects[oi]['class_name'])

        aversions = set()
        for obj in objects:
            if obj['conf'] < self.min_conf:
                continue
            cls = obj['class_name']
            for fi in range(n_faces_local):
                key = (fi, cls)
                if cls in looking.get(fi, set()):
                    self._no_look[key] = 0
                else:
                    self._no_look[key] = self._no_look.get(key, 0) + 1
                    if self._no_look[key] >= self.window:
                        aversions.add(key)
        self._current_aversions = aversions
        self._history.append((kwargs.get('frame_no', 0), len(aversions)))
        return {'aversions': aversions}

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        if self._current_aversions:
            for fi, cls in sorted(self._current_aversions):
                rows.append((f"{resolve_display_pid(fi, pid_map)} avoids {cls}",
                             self._COLOUR))
        else:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "GAZE AVERSION", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for fi, cls in sorted(self._current_aversions):
            rows.append({"label": f"{resolve_display_pid(fi, pid_map)} avoids {cls}"})
        return {
            "title": "GAZE AVERSION",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        active = [(k, cnt) for k, cnt in self._no_look.items()
                  if cnt >= self.window]
        if not active:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        for (fi, cls), cnt in sorted(active):
            rows.append(["gaze_aversion", resolve_display_pid(fi, pid_map),
                         cls, cnt, total_frames, ""])
        return rows

    def time_series_data(self):
        if not self._history:
            return {}
        return {'gaze_aversion_count': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Active aversions',
            'chart_type': 'area',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(len(self._current_aversions))

    def latest_metrics(self):
        result = {}
        for fi, cls in sorted(self._current_aversions):
            plbl = resolve_display_pid(fi)
            key = f'{plbl}_{cls}'
            result[key] = {
                'value': 1.0,
                'label': f'{plbl} avoids {cls}',
                'y_label': 'active',
            }
        if not result:
            result['aversions'] = {'value': 0.0, 'label': 'Active aversions',
                                    'y_label': 'count'}
        return result

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Gaze Aversion CLI flags to the argument parser."""
        g = parser.add_argument_group("Gaze Aversion")
        g.add_argument(
            "--gaze-aversion",
            action="store_true",
            help="Enable gaze aversion detection.",
        )
        g.add_argument(
            "--aversion-window",
            type=int, default=60,
            help="Consecutive frames without looking at an object to flag aversion (default: 60).",
        )
        g.add_argument(
            "--aversion-conf",
            type=float, default=0.5,
            help="Minimum object detection confidence to consider for aversion (default: 0.5).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--gaze-aversion`` was passed, else ``None``."""
        if not (getattr(args, "gaze_aversion", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls(window_frames=args.aversion_window,
                   min_obj_conf=args.aversion_conf)
