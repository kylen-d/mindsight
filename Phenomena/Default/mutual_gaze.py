"""Phenomena/Default/mutual_gaze.py — Dyadic mutual gaze (eye contact) detection."""
from Plugins import PhenomenaPlugin
from utils.geometry import ray_hits_box, extend_ray
from DataCollection.dashboard_output import _draw_panel_section, _DASH_DIM


class MutualGazeTracker(PhenomenaPlugin):
    """
    Detects dyadic mutual gaze (eye contact): person A's gaze ray hits person
    B's face bbox AND person B's ray hits person A's face bbox simultaneously.

    pair_counts accumulates total mutual-gaze frames per pair for post-run CSV.
    """

    name = "mutual_gaze"
    dashboard_panel = "left"
    _COLOUR = (255, 150, 50)

    def __init__(self):
        self.pair_counts: dict = {}   # (i,j) i<j -> total mutual-gaze frames
        self._current_pairs: set = set()

    def update(self, **kwargs):
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])

        mutual = set()
        n = min(len(persons_gaze), len(face_bboxes))
        for i in range(n):
            for j in range(i + 1, n):
                oi, rei, _ = persons_gaze[i]
                oj, rej, _ = persons_gaze[j]
                rei_ext = extend_ray(oi, rei)
                rej_ext = extend_ray(oj, rej)
                x1i, y1i, x2i, y2i = face_bboxes[i]
                x1j, y1j, x2j, y2j = face_bboxes[j]
                if (ray_hits_box(oi, rei_ext, x1j, y1j, x2j, y2j) and
                        ray_hits_box(oj, rej_ext, x1i, y1i, x2i, y2i)):
                    mutual.add((i, j))
                    self.pair_counts[(i, j)] = self.pair_counts.get((i, j), 0) + 1
        self._current_pairs = mutual
        return {'pairs': mutual}

    def dashboard_section(self, panel, y, line_h):
        rows = []
        if self._current_pairs:
            for i, j in sorted(self._current_pairs):
                rows.append((f"P{i} \u2194 P{j}", self._COLOUR))
        else:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "MUTUAL GAZE", self._COLOUR, rows, line_h)

    def csv_rows(self, total_frames):
        if not self.pair_counts:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        for (i, j), cnt in sorted(self.pair_counts.items()):
            pct = cnt / total_frames * 100 if total_frames else 0.0
            rows.append(["mutual_gaze", f"P{i}<->P{j}", "",
                         cnt, total_frames, f"{pct:.4f}"])
        return rows

    def console_summary(self, total_frames):
        if not self.pair_counts:
            return None
        lines = ["Mutual gaze frame counts:"]
        for (i, j), cnt in sorted(self.pair_counts.items()):
            lines.append(f"  P{i}\u2194P{j}: {cnt} frames "
                         f"({cnt / total_frames * 100:.1f}%)")
        return "\n".join(lines)

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
