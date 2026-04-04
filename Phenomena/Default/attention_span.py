"""Phenomena/Default/attention_span.py — Attention span tracking."""
import numpy as np

from DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


class AttentionSpanTracker(PhenomenaPlugin):
    """
    Tracks per-participant per-object average attention span (mean glance duration).

    A *glance* is a contiguous run of frames in which a participant looks at a
    given object class.  Only *completed* glances (those that have ended before
    the current frame) are averaged, so the metric is always based on finished,
    observable fixation episodes.

    Uses (face_idx, class_name) keys to survive per-frame object-index churn.
    """

    name = "attn_span"
    dashboard_panel = "right"
    _COLOUR = (80, 255, 200)

    def __init__(self):
        self._active:    dict = {}   # (face_idx, cls) -> frame_no of glance start
        self._durations: dict = {}   # (face_idx, cls) -> list[int] of completed glance lengths
        self._history:   list = []   # [(frame_no, max_avg_glance)]

    def update(self, **kwargs):
        frame_no = kwargs['frame_no']
        dets = kwargs.get('dets', [])
        hits_set = kwargs.get('hits') or set()

        objects = dets
        looking: set = set()
        for fi, oi in hits_set:
            if oi < len(objects):
                looking.add((fi, objects[oi]['class_name']))

        # Close glances that ended this frame
        for key in list(self._active):
            if key not in looking:
                dur = frame_no - self._active.pop(key)
                if dur > 0:
                    self._durations.setdefault(key, []).append(dur)

        # Open new glances
        for key in looking:
            if key not in self._active:
                self._active[key] = frame_no

        # Track max average glance duration across all participants
        max_avg = 0.0
        for fi in self.all_participants():
            result = self.most_salient(fi)
            if result:
                max_avg = max(max_avg, result[1])
        self._history.append((frame_no, max_avg))
        return {}

    def avg_glance_duration(self, face_idx: int, cls: str) -> float:
        """Return mean completed-glance duration for (face_idx, cls) in frames."""
        durs = self._durations.get((face_idx, cls), [])
        return float(np.mean(durs)) if durs else 0.0

    def all_averages(self, face_idx: int) -> dict:
        """Return {cls: avg_frames} for all objects face_idx has completed glances on."""
        result = {}
        for (fi, cls), durs in self._durations.items():
            if fi == face_idx and durs:
                result[cls] = float(np.mean(durs))
        return result

    def most_salient(self, face_idx: int) -> tuple | None:
        """
        Return (cls, avg_frames) of the most salient object for face_idx —
        the one with the highest average completed-glance duration.
        Returns None if no glances have been completed yet.
        """
        avgs = self.all_averages(face_idx)
        if not avgs:
            return None
        best = max(avgs, key=avgs.__getitem__)
        return best, avgs[best]

    def all_participants(self) -> set:
        """Return all face_idx values for which at least one glance has completed."""
        return {fi for fi, _ in self._durations}

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        all_pids = sorted(self.all_participants())
        for fi in all_pids:
            result = self.most_salient(fi)
            if result:
                cls, avg = result
                plbl = resolve_display_pid(fi, pid_map)
                rows.append((f"{plbl}: {cls}  {avg:.1f}f", self._COLOUR))
        if not rows:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "ATTN SPAN (salient)", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for fi in sorted(self.all_participants()):
            result = self.most_salient(fi)
            if result:
                cls, avg = result
                plbl = resolve_display_pid(fi, pid_map)
                rows.append({
                    "label": f"{plbl}: {cls}",
                    "value": f"{avg:.1f}f",
                })
        return {
            "title": "ATTN SPAN (salient)",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def console_summary(self, total_frames, *, pid_map=None):
        participants = sorted(self.all_participants())
        if not participants:
            return None
        lines = ["Attention span \u2014 most salient object per participant:"]
        for fi in participants:
            plbl = resolve_display_pid(fi, pid_map)
            result = self.most_salient(fi)
            if result:
                cls, avg = result
                lines.append(f"  {plbl}: {cls}  (avg glance {avg:.1f} frames)")
            all_avgs = self.all_averages(fi)
            for obj_cls, obj_avg in sorted(all_avgs.items(), key=lambda x: -x[1]):
                marker = " *" if obj_cls == (result[0] if result else None) else ""
                lines.append(f"    {obj_cls}: {obj_avg:.1f}f{marker}")
        return "\n".join(lines)

    def time_series_data(self):
        if not self._history:
            return {}
        return {'attn_span_max_avg': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Max avg glance duration (frames)',
            'chart_type': 'line',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        max_avg = 0.0
        for fi in self.all_participants():
            result = self.most_salient(fi)
            if result:
                max_avg = max(max_avg, result[1])
        return max_avg

    def latest_metrics(self):
        result = {}
        for fi in sorted(self.all_participants()):
            plbl = resolve_display_pid(fi)
            sal = self.most_salient(fi)
            if sal:
                cls, avg = sal
                result[plbl] = {
                    'value': avg,
                    'label': f'{plbl}: {cls}',
                    'y_label': 'avg frames',
                }
        return result or None

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Attention Span CLI flags to the argument parser."""
        g = parser.add_argument_group("Attention Span")
        g.add_argument(
            "--attn-span",
            action="store_true",
            help=(
                "Track per-participant per-object average attention span "
                "(mean completed-glance duration). Most salient object shown in HUD."
            ),
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--attn-span`` was passed, else ``None``."""
        if not (getattr(args, "attn_span", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls()
