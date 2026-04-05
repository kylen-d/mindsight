"""Phenomena/Default/scanpath.py — Fixation scanpath recording."""
from ms.DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
from ms.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


class ScanpathTracker(PhenomenaPlugin):
    """
    Records the ordered fixation sequence (scanpath) for each person.

    A fixation is confirmed when a person dwells on the same object class for
    >= `dwell_threshold` consecutive frames.  Only transitions are logged.
    """

    name = "scanpath"
    dashboard_panel = "right"
    _COLOUR = (200, 200, 60)

    def __init__(self, dwell_threshold: int = 8):
        self.dwell_thr  = dwell_threshold
        self._dwell:    dict = {}   # face_idx -> {obj_cls: consec_frames}
        self._current:  dict = {}   # face_idx -> currently fixated obj_cls
        self.scanpaths: dict = {}   # face_idx -> [(obj_cls, frame_no), ...]
        self._n_faces:  int  = 0
        self._history:  list = []   # [(frame_no, total_fixation_count)]

    def update(self, **kwargs):
        frame_no = kwargs['frame_no']
        dets = kwargs.get('dets', [])
        n_faces = kwargs.get('n_faces', 0)
        hits_set = kwargs.get('hits') or set()

        objects = dets
        self._n_faces = n_faces

        targets: dict = {}
        for fi, oi in hits_set:
            if oi < len(objects):
                targets.setdefault(fi, set()).add(objects[oi]['class_name'])

        all_faces = set(targets) | set(self._dwell)
        new_fixations: dict = {}

        for fi in all_faces:
            if fi not in self._dwell:
                self._dwell[fi] = {}
            current_cls = targets.get(fi, set())

            for cls in list(self._dwell[fi]):
                if cls in current_cls:
                    self._dwell[fi][cls] += 1
                else:
                    self._dwell[fi][cls] = max(0, self._dwell[fi][cls] - 1)
                    if self._dwell[fi][cls] == 0:
                        del self._dwell[fi][cls]
            for cls in current_cls - set(self._dwell[fi]):
                self._dwell[fi][cls] = 1

            for cls, cnt in self._dwell[fi].items():
                if cnt == self.dwell_thr and self._current.get(fi) != cls:
                    self._current[fi] = cls
                    self.scanpaths.setdefault(fi, []).append((cls, frame_no))
                    new_fixations.setdefault(fi, []).append(cls)

        total_fix = sum(len(v) for v in self.scanpaths.values())
        self._history.append((frame_no, total_fix))
        return {'new_fixations': new_fixations}

    def recent(self, face_idx: int, n: int = 5) -> list:
        """Return the last n fixation class names for a given face."""
        return [cls for cls, _ in self.scanpaths.get(face_idx, [])[-n:]]

    def dashboard_section(self, panel, y, line_h, *, pid_map=None):
        rows = []
        for fi in range(self._n_faces):
            recent = self.recent(fi, n=5)
            if recent:
                plbl = resolve_display_pid(fi, pid_map)
                rows.append((f"{plbl}: " + "\u2192".join(recent), self._COLOUR))
        if not rows:
            rows = [("--", _DASH_DIM)]
        return _draw_panel_section(panel, y, "SCANPATH", self._COLOUR, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        for fi in range(self._n_faces):
            recent = self.recent(fi, n=3)
            if recent:
                plbl = resolve_display_pid(fi, pid_map)
                # Show path as a single label so it wraps instead of overflowing
                rows.append({
                    "label": f"{plbl}: " + "\u2192".join(recent),
                })
        return {
            "title": "SCANPATH",
            "colour": self._COLOUR,
            "rows": rows,
            "empty_text": "--",
        }

    def csv_rows(self, total_frames, *, pid_map=None):
        if not self.scanpaths:
            return []
        rows = [["category", "participant", "object",
                 "frames_active", "total_frames", "value_pct"]]
        for fi, path_entries in sorted(self.scanpaths.items()):
            seq = ";".join(cls for cls, _ in path_entries)
            rows.append(["scanpath", resolve_display_pid(fi, pid_map), seq,
                         len(path_entries), total_frames, ""])
        return rows

    def time_series_data(self):
        if not self._history:
            return {}
        return {'scanpath_fixations': {
            'x': [f for f, _ in self._history],
            'y': [v for _, v in self._history],
            'label': 'Total fixation count',
            'chart_type': 'step',
            'color': self._COLOUR,
        }}

    def latest_metric(self):
        return float(sum(len(v) for v in self.scanpaths.values()))

    def latest_metrics(self):
        result = {}
        for fi in range(self._n_faces):
            plbl = resolve_display_pid(fi)
            count = len(self.scanpaths.get(fi, []))
            result[plbl] = {
                'value': float(count),
                'label': f'{plbl} fixations',
                'y_label': 'fixations',
            }
        return result or None

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add Scanpath CLI flags to the argument parser."""
        g = parser.add_argument_group("Scanpath")
        g.add_argument(
            "--scanpath",
            action="store_true",
            help="Enable fixation scanpath recording.",
        )
        g.add_argument(
            "--scanpath-dwell",
            type=int, default=8,
            help="Consecutive frames on the same object to confirm a fixation (default: 8).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--scanpath`` was passed, else ``None``."""
        if not (getattr(args, "scanpath", False)
                or getattr(args, "all_phenomena", False)):
            return None
        return cls(dwell_threshold=args.scanpath_dwell)
