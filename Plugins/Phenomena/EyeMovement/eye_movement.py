"""
Plugins/Phenomena/EyeMovement/eye_movement.py -- Eye Movement Classifier.

Continuous fixation/saccade/pursuit classification using I-VT (Identification
by Velocity Threshold).  Supports gaze-mode (uses persons_gaze ray_end) and
iris-mode (uses MediaPipe iris center displacement within the eye socket).

This plugin ONLY observes and classifies -- it never modifies persons_gaze,
ray_end, hit_events, or any other pipeline state.

Activation: ``--eye-movement``
"""

from __future__ import annotations

import cv2
import numpy as np

from ms.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin
from Plugins.Phenomena.EyeMovement.classifiers import EyeState, IVTClassifier


def _dash():
    from ms.DataCollection.dashboard_output import (
        _DASH_DIM,
        _dash_line_h,
        _draw_panel_section,
    )
    return _draw_panel_section, _dash_line_h, _DASH_DIM


_STATE_LABEL = {
    EyeState.FIXATION: "F",
    EyeState.SACCADE: "S",
    EyeState.SMOOTH_PURSUIT: "P",
}

_STATE_COL = {
    EyeState.FIXATION: (0, 255, 0),       # green
    EyeState.SACCADE: (0, 0, 255),         # red
    EyeState.SMOOTH_PURSUIT: (0, 200, 255), # yellow-orange
}


class EyeMovementTracker(PhenomenaPlugin):
    """
    Classifies each frame as fixation, saccade, or smooth pursuit per
    participant using the I-VT algorithm.
    """

    name = "eye_movement"
    dashboard_panel = "right"

    _COL = (255, 180, 50)  # teal-blue

    def __init__(
        self,
        source: str = "gaze",
        saccade_threshold: float = 30.0,
        fixation_threshold: float = 10.0,
        min_fixation_frames: int = 4,
        velocity_window: int = 3,
    ) -> None:
        self._source = source
        self._saccade_thresh = saccade_threshold
        self._fixation_thresh = fixation_threshold
        self._min_fix_frames = min_fixation_frames
        self._vel_window = velocity_window

        # Per-track classifiers
        self._classifiers: dict[int, IVTClassifier] = {}

        # Current-frame state cache
        self._current_states: dict[int, EyeState] = {}
        self._last_face_bboxes: list = []
        self._last_face_track_ids: list = []

    def _get_classifier(self, tid: int) -> IVTClassifier:
        if tid not in self._classifiers:
            self._classifiers[tid] = IVTClassifier(
                saccade_threshold=self._saccade_thresh,
                fixation_threshold=self._fixation_thresh,
                min_fixation_frames=self._min_fix_frames,
                velocity_window=self._vel_window,
            )
        return self._classifiers[tid]

    def _iris_position(self, frame, bbox, fi, aux_frames, pid_map):
        """Compute iris center position relative to eye socket (iris mode)."""
        from ms.utils.mediapipe_face import extract_iris_data

        # Get face crop with padding (mediapipe needs head context)
        if frame is None or bbox is None:
            return None

        import cv2
        bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 10 or bh < 10:
            return None
        pad_x, pad_y = bw // 2, bh // 2
        h, w = frame.shape[:2]
        x1 = max(0, bx1 - pad_x)
        y1 = max(0, by1 - pad_y)
        x2 = min(w, bx2 + pad_x)
        y2 = min(h, by2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if max(crop.shape[:2]) < 200:
            s = 200 / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                              interpolation=cv2.INTER_CUBIC)
        iris_data = extract_iris_data(crop)
        if iris_data is None:
            return None

        # Use average iris center relative to eye center for eye rotation signal
        positions = []
        for side in ('right', 'left'):
            if not getattr(iris_data, f'{side}_valid'):
                continue
            iris_c = getattr(iris_data, f'{side}_iris_center')
            eye_pts = getattr(iris_data, f'{side}_eye_contour')
            if iris_c is None or eye_pts is None:
                continue
            eye_center = np.mean(eye_pts, axis=0)
            # Relative position: iris displacement from eye center
            rel = iris_c - eye_center
            positions.append(rel)

        if not positions:
            return None

        return np.mean(positions, axis=0)

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        frame_no = kwargs['frame_no']
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])
        face_track_ids = kwargs.get('face_track_ids')
        ray_snapped = kwargs.get('ray_snapped', [])
        frame = kwargs.get('frame')
        aux_frames = kwargs.get('aux_frames', {})
        pid_map = kwargs.get('pid_map')

        tids = face_track_ids if face_track_ids is not None \
               else list(range(len(persons_gaze)))

        self._last_face_bboxes = face_bboxes
        self._last_face_track_ids = tids
        self._current_states.clear()

        for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
            tid = tids[fi] if fi < len(tids) else fi
            classifier = self._get_classifier(tid)

            # Skip snapped frames in gaze mode
            snapped = ray_snapped[fi] if fi < len(ray_snapped) else False

            if self._source == "iris":
                bbox = face_bboxes[fi] if fi < len(face_bboxes) else None
                pos = self._iris_position(frame, bbox, fi, aux_frames, pid_map)
                if pos is None:
                    continue
                state = classifier.classify(pos, frame_no)
            else:
                # Gaze mode: use ray_end displacement
                pos = np.array([float(ray_end[0]), float(ray_end[1])])
                state = classifier.classify(pos, frame_no, skip=snapped)

            self._current_states[tid] = state

        return {}

    # ── Frame overlay ─────────────────────────────────────────────────────────

    def draw_frame(self, frame) -> None:
        if not self._current_states:
            return

        for fi, bbox in enumerate(self._last_face_bboxes):
            tid = (self._last_face_track_ids[fi]
                   if fi < len(self._last_face_track_ids) else fi)
            if tid not in self._current_states:
                continue

            state = self._current_states[tid]
            label = _STATE_LABEL[state]
            col = _STATE_COL[state]

            x1, y1, x2, y2 = bbox
            # Draw label near top-right of face bbox
            lx = int(x2) + 4
            ly = int(y1) + 14
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def dashboard_section(self, panel, y: int, line_h: int, *,
                          pid_map=None) -> int:
        draw_section, _, DASH_DIM = _dash()
        rows = []

        if self._current_states:
            for tid in sorted(self._current_states):
                state = self._current_states[tid]
                plbl = resolve_display_pid(tid, pid_map)
                classifier = self._classifiers.get(tid)
                stats = classifier.summary_stats() if classifier else {}
                fix_pct = stats.get('fixation_pct', 0)
                sac_pct = stats.get('saccade_pct', 0)
                rows.append((
                    f"{plbl}: {state.value.upper()}"
                    f"  fix={fix_pct:.0f}%  sac={sac_pct:.0f}%",
                    _STATE_COL[state],
                ))
        else:
            rows = [("--", DASH_DIM)]

        return draw_section(panel, y, "EYE MOVEMENT", self._COL, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        if self._current_states:
            for tid in sorted(self._current_states):
                state = self._current_states[tid]
                plbl = resolve_display_pid(tid, pid_map)
                classifier = self._classifiers.get(tid)
                stats = classifier.summary_stats() if classifier else {}
                rows.append({
                    'label': f"{plbl}: {state.value.upper()}",
                    'value': f"fix={stats.get('fixation_pct', 0):.0f}%  sac={stats.get('saccade_pct', 0):.0f}%",
                })
        return {
            'title': 'EYE MOVEMENT',
            'colour': self._COL,
            'rows': rows,
            'empty_text': '--',
        }

    def latest_metrics(self) -> dict | None:
        if not self._current_states:
            return None
        metrics = {}
        for tid in self._current_states:
            classifier = self._classifiers.get(tid)
            if classifier:
                stats = classifier.summary_stats()
                metrics[f"em_fix_{tid}"] = {
                    'value': stats.get('fixation_pct', 0),
                    'label': f"P{tid} fixation%",
                    'y_label': '%',
                }
        return metrics

    # ── Time-series ───────────────────────────────────────────────────────────

    def time_series_data(self) -> dict:
        # Build per-participant fixation percentage over time from events
        series = {}
        for tid, classifier in self._classifiers.items():
            fix_events = [e for e in classifier.events if e['type'] == 'fixation']
            if not fix_events:
                continue
            frames = []
            durations = []
            for e in fix_events:
                frames.append(e['start_frame'])
                durations.append(e['duration_frames'])
            series[f"fixation_dur_{tid}"] = {
                'x': frames,
                'y': durations,
                'label': f"P{tid} fixation duration",
                'chart_type': 'step',
                'color': (0, 255, 0),
            }
        return series

    # ── CSV output ────────────────────────────────────────────────────────────

    def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
        has_events = any(c.events for c in self._classifiers.values())
        if not has_events:
            return []

        rows: list[list] = [
            [],
            ["eye_movement_events"],
            ["participant", "event_type", "start_frame", "end_frame",
             "duration_frames", "peak_velocity"],
        ]
        for tid in sorted(self._classifiers):
            plbl = resolve_display_pid(tid, pid_map)
            for ev in self._classifiers[tid].events:
                rows.append([
                    plbl, ev['type'], ev['start_frame'], ev['end_frame'],
                    ev['duration_frames'], f"{ev['peak_velocity']:.2f}",
                ])

        # Summary
        rows.append([])
        rows.append(["eye_movement_summary"])
        rows.append(["participant", "fixation_count", "saccade_count",
                      "mean_fixation_duration", "mean_saccade_velocity",
                      "fixation_pct", "saccade_pct"])
        for tid in sorted(self._classifiers):
            plbl = resolve_display_pid(tid, pid_map)
            # Finalize classifier to capture last segment
            self._classifiers[tid].finalize(total_frames)
            stats = self._classifiers[tid].summary_stats()
            rows.append([
                plbl,
                stats['fixation_count'],
                stats['saccade_count'],
                f"{stats['mean_fixation_duration']:.1f}",
                f"{stats['mean_saccade_velocity']:.1f}",
                f"{stats['fixation_pct']:.1f}",
                f"{stats['saccade_pct']:.1f}",
            ])

        return rows

    def console_summary(self, total_frames: int, *, pid_map=None) -> str | None:
        if not self._classifiers:
            return None

        lines = ["\n--- Eye Movement Summary ---"]
        for tid in sorted(self._classifiers):
            plbl = resolve_display_pid(tid, pid_map)
            self._classifiers[tid].finalize(total_frames)
            stats = self._classifiers[tid].summary_stats()
            lines.append(
                f"  {plbl}: fix={stats['fixation_count']}"
                f"  sac={stats['saccade_count']}"
                f"  fix%={stats['fixation_pct']:.1f}"
                f"  sac%={stats['saccade_pct']:.1f}"
            )
        return "\n".join(lines)

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        g = parser.add_argument_group("Eye Movement plugin")
        g.add_argument("--eye-movement", action="store_true",
                        help="Enable eye movement classification.")
        g.add_argument("--em-source", choices=["gaze", "iris"], default="gaze",
                        help="Velocity source (default: gaze).")
        g.add_argument("--em-saccade-thresh", type=float, default=30.0, metavar="F",
                        help="Saccade threshold px/frame (default: 30.0).")
        g.add_argument("--em-fixation-thresh", type=float, default=10.0, metavar="F",
                        help="Fixation threshold px/frame (default: 10.0).")
        g.add_argument("--em-min-fixation", type=int, default=4, metavar="N",
                        help="Min fixation duration frames (default: 4).")
        g.add_argument("--em-velocity-window", type=int, default=3, metavar="N",
                        help="Median filter window (default: 3).")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "eye_movement", False):
            return None
        inst = cls(
            source=getattr(args, "em_source", "gaze"),
            saccade_threshold=getattr(args, "em_saccade_thresh", 30.0),
            fixation_threshold=getattr(args, "em_fixation_thresh", 10.0),
            min_fixation_frames=getattr(args, "em_min_fixation", 4),
            velocity_window=getattr(args, "em_velocity_window", 3),
        )
        print(
            f"EyeMovement: source={inst._source}"
            f"  sac_thresh={inst._saccade_thresh}"
            f"  fix_thresh={inst._fixation_thresh}"
        )
        return inst


PLUGIN_CLASS = EyeMovementTracker
