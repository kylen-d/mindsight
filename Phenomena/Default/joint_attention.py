"""Phenomena/Default/joint_attention.py — Joint attention tracking.

Consolidates ALL joint attention logic that was previously scattered across
MindSight.py (percentage counters, history), dashboard_matplotlib.py (card),
csv_output.py (summary section), chart_output.py (time-series), and
data_pipeline.py (console output) into a single PhenomenaPlugin.

The tracker handles:
1. Temporal sliding-window confirmation of raw JA events
2. Running percentage (confirmed frames / total frames)
3. Dashboard card rendering via dashboard_data()
4. CSV summary via csv_rows()
5. Post-run charts via time_series_data()
6. Live dashboard metrics via latest_metrics()
7. Console summary via console_summary()
"""

import collections
import math

from Plugins import PhenomenaPlugin


class JointAttentionTracker(PhenomenaPlugin):
    """Full-lifecycle joint attention tracker.

    Replaces the old JointAttentionTemporalTracker (which only handled
    temporal filtering) with a complete PhenomenaPlugin that owns all JA
    metrics, display, and output.

    Must be the **first** tracker in ``all_trackers`` so that
    ``confirmed_objs`` is available to subsequent trackers.
    """

    name = "joint_attention"
    dashboard_panel = "left"
    _COLOUR = (0, 200, 255)  # BGR cyan

    def __init__(self, window: int = 0, threshold: float = 0.70,
                 quorum: float = 1.0):
        # Temporal filter parameters
        self._window = max(1, window) if window > 0 else 0
        self._threshold = threshold
        self._quorum = quorum
        self._filter_history: collections.deque = (
            collections.deque(maxlen=self._window) if self._window > 0
            else None
        )

        # Per-frame state
        self._current_confirmed: set = set()
        self._current_raw: set = set()
        self._current_obj_names: list = []

        # Running counters
        self._joint_frames: int = 0
        self._confirmed_frames: int = 0
        self._total_frames: int = 0

        # Time-series history for charts
        self._history: list = []  # [(frame_no, joint_pct)]

        # HUD extra text (window fill %)
        self._extra_hud: str | None = None

        # JA mode description string (set externally after construction)
        self.ja_mode_str: str | None = None

    # ── Core JA computation ──────────────────────────────────────────────────

    @staticmethod
    def _compute_raw_ja(hits: set, n_faces: int, quorum: float) -> set:
        """Compute raw joint attention from gaze-object intersections."""
        if n_faces < 2:
            return set()
        min_watchers = max(2, math.ceil(quorum * n_faces))
        watchers: dict = {}
        for fi, oi in hits:
            watchers.setdefault(oi, set()).add(fi)
        return {oi for oi, w in watchers.items() if len(w) >= min_watchers}

    def _temporal_filter(self, raw_ja: set) -> set:
        """Apply sliding-window temporal confirmation."""
        if self._filter_history is None:
            return raw_ja
        self._filter_history.append(frozenset(raw_ja))
        n = len(self._filter_history)
        counts = collections.Counter(
            oi for s in self._filter_history for oi in s)
        return {oi for oi, cnt in counts.items()
                if cnt / n >= self._threshold}

    @property
    def fill(self) -> float:
        """Fraction of the temporal window that has been populated (0-1)."""
        if self._filter_history is None:
            return 1.0
        return len(self._filter_history) / self._window

    @property
    def joint_pct(self) -> float:
        """Running confirmed-JA percentage."""
        if self._total_frames == 0:
            return 0.0
        return self._confirmed_frames / self._total_frames * 100

    # ── PhenomenaPlugin lifecycle ────────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        frame_no = kwargs.get('frame_no', 0)
        hits = kwargs.get('hits') or set()
        n_faces = kwargs.get('n_faces', 0)
        persons_gaze = kwargs.get('persons_gaze', [])
        dets = kwargs.get('dets', [])
        tip_convergences = kwargs.get('tip_convergences', [])

        if not n_faces:
            n_faces = len(persons_gaze)

        # 1. Raw JA computation
        raw_ja = self._compute_raw_ja(hits, n_faces, self._quorum)
        self._current_raw = raw_ja

        # 2. Temporal filtering
        confirmed = self._temporal_filter(raw_ja)
        self._current_confirmed = confirmed

        # 3. Resolve object names for display
        self._current_obj_names = [
            dets[oi]['class_name']
            for oi in sorted(confirmed) if oi < len(dets)
        ]

        # 4. Update running counters
        self._total_frames += 1
        if raw_ja or tip_convergences:
            self._joint_frames += 1
        if confirmed or tip_convergences:
            self._confirmed_frames += 1

        # 5. Build HUD text
        if self._filter_history is not None:
            fill_pct = self.fill * 100
            self._extra_hud = (
                f"{self.ja_mode_str}  win:{fill_pct:.0f}%"
                if self.ja_mode_str else f"win:{fill_pct:.0f}%"
            )
        else:
            self._extra_hud = self.ja_mode_str

        # 6. History for charts
        self._history.append((frame_no, self.joint_pct))

        return {
            'confirmed_objs': confirmed,
            'extra_hud': self._extra_hud,
            'joint_pct': self.joint_pct,
        }

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        if self._current_obj_names:
            rows.append({'label': ', '.join(self._current_obj_names)})
        rows.append({
            'label': 'JA frames',
            'value': f'{self.joint_pct:.1f}%',
            'pct': self.joint_pct / 100.0,
        })
        if self._extra_hud:
            rows.append({'label': self._extra_hud})
        return {
            'title': 'JOINT ATTENTION',
            'colour': self._COLOUR,
            'rows': rows,
            'empty_text': 'No joint attention',
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        tf = total_frames or self._total_frames
        pct = self._confirmed_frames / tf * 100 if tf else 0.0
        return [
            ["category", "participant", "object",
             "frames_active", "total_frames", "value_pct"],
            ["joint_attention", "all", "",
             self._confirmed_frames, tf, f"{pct:.4f}"],
        ]

    def time_series_data(self):
        if not self._history:
            return {}
        return {'ja_pct': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'JA %',
            'chart_type': 'area',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return self.joint_pct

    def latest_metrics(self):
        return {'ja_pct': {
            'value': self.joint_pct,
            'label': 'JA %',
            'y_label': '%',
        }}

    def console_summary(self, total_frames, *, pid_map=None):
        tf = total_frames or self._total_frames
        raw_pct = self._joint_frames / tf * 100 if tf else 0.0
        conf_pct = self._confirmed_frames / tf * 100 if tf else 0.0
        lines = [
            f"Joint attention (raw):       "
            f"{self._joint_frames}/{tf} frames = {raw_pct:.1f}%",
        ]
        if self._filter_history is not None:
            lines.append(
                f"Joint attention (confirmed): "
                f"{self._confirmed_frames}/{tf} frames = {conf_pct:.1f}%"
            )
        return "\n".join(lines)

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        pass  # JA args are handled by phenomena_tracking.py

    @classmethod
    def from_args(cls, args):
        return None  # Instantiated directly by init_phenomena_trackers
