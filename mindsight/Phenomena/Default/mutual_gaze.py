"""Phenomena/Default/mutual_gaze.py — Dyadic mutual gaze (eye contact) detection."""
import numpy as np

from mindsight.outputs.dashboard_output import _DASH_DIM, _draw_panel_section
from mindsight.Phenomena.helpers import EpisodeLog
from mindsight.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin
from mindsight.utils.geometry import extend_ray, ray_hits_box


class MutualGazeTracker(PhenomenaPlugin):
    """
    Detects dyadic mutual gaze (eye contact): person A's gaze ray hits person
    B's face bbox AND person B's ray hits person A's face bbox simultaneously.

    pair_counts accumulates total mutual-gaze frames per pair for post-run CSV.
    """

    name = "mutual_gaze"

    #: Live-dashboard style; mirrors time_series_data chart_type.
    live_chart_type = "area"
    dashboard_panel = "left"
    _COLOUR = (255, 150, 50)

    def __init__(self):
        self.pair_counts: dict = {}   # (i,j) i<j -> total mutual-gaze frames
        self._current_pairs: set = set()
        self._history: list = []      # [(frame_no, n_active_pairs)]
        self._episodes = EpisodeLog()  # per-pair mutual-gaze spans

    def update(self, **kwargs):
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])
        detect_extend = kwargs.get('detect_extend', 0.0)
        scope = kwargs.get('detect_extend_scope', 'objects')
        use_extend = detect_extend > 0 and scope in ('phenomena', 'both')

        mutual = set()
        n = min(len(persons_gaze), len(face_bboxes))

        # Pre-compute ray endpoints once per person (O(N) instead of O(N^2))
        endpoints = []
        origins = []
        for i in range(n):
            oi, rei, _ = persons_gaze[i]
            origins.append(oi)
            if use_extend:
                endpoints.append(extend_ray(oi, rei, length=float(np.linalg.norm(np.asarray(rei) - np.asarray(oi))) + detect_extend))
            else:
                endpoints.append(rei)

        for i in range(n):
            for j in range(i + 1, n):
                x1i, y1i, x2i, y2i = face_bboxes[i]
                x1j, y1j, x2j, y2j = face_bboxes[j]
                if (ray_hits_box(origins[i], endpoints[i], x1j, y1j, x2j, y2j) and
                        ray_hits_box(origins[j], endpoints[j], x1i, y1i, x2i, y2i)):
                    mutual.add((i, j))
                    self.pair_counts[(i, j)] = self.pair_counts.get((i, j), 0) + 1

        # Episode bookkeeping: open pairs that just became mutual, close pairs
        # that just stopped. pair_counts accumulation above is unchanged.
        frame_no = kwargs.get('frame_no', 0)
        for pair in mutual - self._current_pairs:
            i, j = pair
            self._episodes.open(pair, phenomenon="mutual_gaze",
                                participant=i, partner=j, object="",
                                frame_start=frame_no)
        for pair in self._current_pairs - mutual:
            self._episodes.close(pair, frame_no)

        self._current_pairs = mutual
        self._history.append((frame_no, len(mutual)))
        return {'pairs': mutual}

    def finalize(self, frame_no, **kwargs):
        """Close any mutual-gaze pairs still active at the final frame."""
        self._episodes.close_all(frame_no)

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        if self._current_pairs:
            for i, j in sorted(self._current_pairs):
                pi = resolve_display_pid(i, pid_map)
                pj = resolve_display_pid(j, pid_map)
                rows.append((f"{pi} \u2194 {pj}", self._COLOUR))
        else:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "MUTUAL GAZE", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for i, j in sorted(self._current_pairs):
            pi = resolve_display_pid(i, pid_map)
            pj = resolve_display_pid(j, pid_map)
            rows.append({"label": f"{pi} \u2194 {pj}"})
        return {
            "title": "MUTUAL GAZE",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def summary_metrics(self, total_frames, fps, *, pid_map=None):
        rows = []
        for (i, j), cnt in sorted(self.pair_counts.items()):
            pct = cnt / total_frames * 100 if total_frames else 0.0
            pi = resolve_display_pid(i, pid_map)
            pj = resolve_display_pid(j, pid_map)
            sec = f"{cnt / fps:.3f}" if fps else ""
            rows.append({"participant": pi, "partner": pj, "object": "",
                         "metric": "frames_active", "value": cnt})
            rows.append({"participant": pi, "partner": pj, "object": "",
                         "metric": "seconds_active", "value": sec})
            rows.append({"participant": pi, "partner": pj, "object": "",
                         "metric": "pct_of_video", "value": f"{pct:.4f}"})
        return rows

    def console_summary(self, total_frames, *, pid_map=None):
        if not self.pair_counts:
            return None
        lines = ["Mutual gaze frame counts:"]
        for (i, j), cnt in sorted(self.pair_counts.items()):
            pi = resolve_display_pid(i, pid_map)
            pj = resolve_display_pid(j, pid_map)
            lines.append(f"  {pi}\u2194{pj}: {cnt} frames "
                         f"({cnt / total_frames * 100:.1f}%)")
        return "\n".join(lines)

    def time_series_data(self):
        if not self._history:
            return {}
        return {'mutual_gaze_pairs': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Active mutual-gaze pairs',
            'chart_type': 'area',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(len(self._current_pairs))

    def latest_metrics(self):
        from mindsight.pipeline_config import resolve_display_pid
        if not self._current_pairs:
            return {'pairs': {'value': 0.0, 'label': 'Active pairs',
                              'y_label': 'pairs'}}
        result = {}
        for i, j in sorted(self._current_pairs):
            pi = resolve_display_pid(i)
            pj = resolve_display_pid(j)
            result[f'{pi}\u2194{pj}'] = {
                'value': 1.0, 'label': f'{pi} \u2194 {pj}', 'y_label': 'active'}
        return result

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Mutual Gaze CLI flags to the argument parser."""
        g = parser.add_argument_group("Mutual Gaze")
        g.add_argument(
            "--mutual-gaze",
            action="store_true",
            help="Enable mutual gaze (eye contact) detection.",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--mutual-gaze`` was passed, else ``None``."""
        if not (getattr(args, "mutual_gaze", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls()
