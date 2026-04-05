"""
Plugins/Phenomena/NovelSalience/novel_salience.py — Novel Salience detector.

Tracks rapid shifts in gaze direction (saccades) as a proxy for detecting
highly salient or unexpected stimuli.  When a participant's gaze endpoint
moves faster than a configurable speed threshold, a "novel salience" event
is recorded.  The direction of the shift (LEFT / RIGHT / UP / DOWN) indicates
where the novel stimulus likely appeared.

Works with both per-face (pitch/yaw) and scene-level (Gazelle) backends:
  • Per-face mode  : tracks gaze-ray endpoint velocity in pixels/frame.
  • Gazelle mode   : same — gaze target position displacement in pixels/frame.

Activation
----------
    python MindSight.py --novel-salience [options …]

CLI flags
---------
    --novel-salience          Enable the plugin (required to activate).
    --ns-speed-thresh PX      Min endpoint speed (px/frame) to flag an event
                              (default 40).
    --ns-cooldown N           Frames between consecutive events per face
                              (default 20).
    --ns-history N            History depth for velocity smoothing (default 2).
    --ns-flash N              Frames to show the visual saccade indicator
                              (default 12).

Dashboard
---------
Appears in the right panel as "NOVEL SALIENCE", showing the 3 most recent
saccade events with face ID, direction, speed, and frame number.

CSV output (--summary)
----------------------
Appended as a section with columns:
    category, frame_no, face_id, direction, speed_px, speed_deg, delta_x, delta_y

Frame overlay
-------------
When a saccade event fires, a fading cyan ring is drawn around the face
bounding box and a directional arrow is drawn at the gaze endpoint.

Sign conventions (for reference)
---------------------------------
In MindSight's screen-space coordinates (y increases downward):
    positive yaw   → gaze moved LEFT  (ray_end.x decreases)
    negative yaw   → gaze moved RIGHT (ray_end.x increases)
    positive pitch → gaze moved DOWN  (ray_end.y increases)
    negative pitch → gaze moved UP    (ray_end.y decreases)
Direction labels are derived from ray_end displacement, so they are always
correct regardless of gaze backend.
"""

from __future__ import annotations

import collections
import math

import cv2
import numpy as np

from ms.pipeline_config import resolve_display_pid
from Plugins import PhenomenaPlugin


# ── Dashboard drawing helpers ─────────────────────────────────────────────────
# Imported lazily inside methods that need them so the plugin loads even when
# the DataCollection package has not been fully initialized yet.
def _dash():
    from ms.DataCollection.dashboard_output import (
        _DASH_DIM,
        _dash_line_h,
        _draw_panel_section,
    )
    return _draw_panel_section, _dash_line_h, _DASH_DIM


# ══════════════════════════════════════════════════════════════════════════════
# Plugin
# ══════════════════════════════════════════════════════════════════════════════

class NovelSalienceTracker(PhenomenaPlugin):
    """
    Detects rapid gaze shifts (saccades) as a proxy for novel-stimulus
    salience.  An event fires when the gaze-ray endpoint moves faster than
    ``speed_thresh`` pixels per frame (after ``history``-frame smoothing)
    and at least ``cooldown`` frames have passed since the last event for
    that face track.
    """

    name           = "novel_salience"
    dashboard_panel = "right"

    #: BGR colour used for all NovelSalience overlays and HUD text.
    _NS_COL = (0, 215, 255)   # vivid amber-cyan

    def __init__(
        self,
        speed_thresh: float = 40.0,
        cooldown: int       = 20,
        history: int        = 2,
        flash: int          = 12,
    ) -> None:
        """
        Parameters
        ----------
        speed_thresh : Min gaze-endpoint speed (pixels/frame) to flag an event.
        cooldown     : Frames between consecutive events for the same face track.
        history      : Depth of the sliding window used to smooth velocity
                       (1 = instantaneous; 2–3 = light smoothing).
        flash        : How many frames the visual indicator persists after an event.
        """
        self._thresh          = speed_thresh
        self._cooldown_frames = cooldown
        self._hist_len        = max(1, history)
        self._flash_len       = max(1, flash)

        # Per-track-ID state
        self._pos_hist:   dict[int, collections.deque] = {}  # ray_end history
        self._angle_hist: dict[int, collections.deque] = {}  # (pitch, yaw) history
        self._cooldown:   dict[int, int]               = {}  # frames until next event
        self._flash:      dict[int, dict]              = {}  # active visual indicators

        # Full event log (persists across the whole run)
        self.events: list[dict] = []

        # Last-frame data cached for draw_frame (populated by update())
        self._last_persons_gaze:   list = []
        self._last_face_bboxes:    list = []
        self._last_face_track_ids: list = []

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        """
        Detect novel-salience events for the current frame.

        For each tracked face:
        1.  Compute the gaze-ray endpoint displacement vs the previous frame.
        2.  Average over ``history`` frames to smooth noise.
        3.  If speed > threshold AND cooldown has expired → fire an event.

        Returns a dict with key ``'events'``: list of events fired this frame.
        """
        frame_no = kwargs['frame_no']
        persons_gaze = kwargs.get('persons_gaze', [])
        face_bboxes = kwargs.get('face_bboxes', [])
        face_track_ids = kwargs.get('face_track_ids')

        tids = face_track_ids if face_track_ids is not None \
               else list(range(len(persons_gaze)))

        # Cache for draw_frame
        self._last_persons_gaze   = persons_gaze
        self._last_face_bboxes    = face_bboxes
        self._last_face_track_ids = tids

        current_events: list[dict] = []

        for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
            tid = tids[fi] if fi < len(tids) else fi

            # --- Tick down cooldown and flash ---
            if self._cooldown.get(tid, 0) > 0:
                self._cooldown[tid] -= 1
            if tid in self._flash:
                self._flash[tid]['frames'] -= 1
                if self._flash[tid]['frames'] <= 0:
                    del self._flash[tid]

            # --- Update position history ---
            pos = np.array([float(ray_end[0]), float(ray_end[1])])
            if tid not in self._pos_hist:
                self._pos_hist[tid] = collections.deque(maxlen=self._hist_len + 1)
            self._pos_hist[tid].append(pos)

            # --- Update angle history (per-face backends only) ---
            if angles is not None:
                if tid not in self._angle_hist:
                    self._angle_hist[tid] = collections.deque(maxlen=self._hist_len + 1)
                self._angle_hist[tid].append((float(angles[0]), float(angles[1])))

            # --- Need at least 2 position samples to compute velocity ---
            hist = self._pos_hist[tid]
            if len(hist) < 2:
                continue

            # Smoothed velocity: average displacement over the window
            deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
            mean_delta = np.mean(deltas, axis=0)
            speed_px   = float(np.linalg.norm(mean_delta))

            # Angular speed in degrees/frame (if pitch/yaw available)
            speed_deg: float | None = None
            if angles is not None and tid in self._angle_hist:
                ah = self._angle_hist[tid]
                if len(ah) >= 2:
                    dp = (ah[-1][0] - ah[-2][0]) * 180.0 / math.pi
                    dy = (ah[-1][1] - ah[-2][1]) * 180.0 / math.pi
                    speed_deg = math.sqrt(dp * dp + dy * dy)

            # Direction from mean screen-space displacement
            ax, ay = float(mean_delta[0]), float(mean_delta[1])
            if abs(ax) >= abs(ay):
                direction = "LEFT" if ax < 0 else "RIGHT"
            else:
                direction = "UP" if ay < 0 else "DOWN"

            # --- Check threshold + cooldown ---
            if speed_px >= self._thresh and self._cooldown.get(tid, 0) == 0:
                event = {
                    'frame_no':  frame_no,
                    'face_id':   tid,
                    'speed_px':  speed_px,
                    'speed_deg': speed_deg,
                    'direction': direction,
                    'delta_x':   ax,
                    'delta_y':   ay,
                }
                self.events.append(event)
                current_events.append(event)
                self._cooldown[tid] = self._cooldown_frames
                self._flash[tid]    = {
                    'frames':    self._flash_len,
                    'direction': direction,
                    'speed_px':  speed_px,
                }

        return {'events': current_events}

    # ── Frame overlay ─────────────────────────────────────────────────────────

    def draw_frame(self, frame) -> None:
        """
        Draw fading indicators on the video frame for each active saccade event.

        For each flashing face:
        • A coloured ring around the face bounding box (thickness fades out).
        • A directional arrow at the gaze-ray endpoint.
        • A text label "NS! <direction>" near the endpoint.
        """
        if not self._flash:
            return

        h, w = frame.shape[:2]

        for fi, (origin, ray_end, _) in enumerate(self._last_persons_gaze):
            tid = (self._last_face_track_ids[fi]
                   if fi < len(self._last_face_track_ids) else fi)
            if tid not in self._flash:
                continue

            fdata = self._flash[tid]
            frac  = fdata['frames'] / self._flash_len  # 1.0 → 0.0 as flash fades

            col = tuple(int(c * frac) for c in self._NS_COL)

            # Ring around face bbox
            if fi < len(self._last_face_bboxes):
                x1, y1, x2, y2 = self._last_face_bboxes[fi]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r  = max((x2 - x1), (y2 - y1)) // 2 + 12
                thick = max(1, int(5 * frac))
                cv2.circle(frame, (cx, cy), r, col, thick, cv2.LINE_AA)

            # Directional arrow from gaze endpoint
            ex = int(np.clip(ray_end[0], 0, w - 1))
            ey = int(np.clip(ray_end[1], 0, h - 1))
            arrow_len = max(8, int(35 * frac))
            d = fdata['direction']
            tip_map = {
                'LEFT':  (ex - arrow_len, ey),
                'RIGHT': (ex + arrow_len, ey),
                'UP':    (ex, ey - arrow_len),
                'DOWN':  (ex, ey + arrow_len),
            }
            tip = tip_map[d]
            thick_arrow = max(1, int(3 * frac))
            cv2.arrowedLine(frame, (ex, ey), tip, col,
                            thick_arrow, cv2.LINE_AA, tipLength=0.45)

            # Text label
            label = f"NS! {d}"
            lx = min(ex + 6, w - 80)
            ly = max(ey - 10, 12)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # ── Dashboard section ─────────────────────────────────────────────────────

    def dashboard_section(self, panel, y: int, line_h: int, *,
                          pid_map=None) -> int:
        """
        Draw a "NOVEL SALIENCE" section in the side panel showing the 3 most
        recent saccade events and a per-face tally.
        """
        draw_section, _, DASH_DIM = _dash()

        rows: list[tuple[str, tuple]] = []

        if self.events:
            # Show last 3 events
            for ev in self.events[-3:]:
                speed_str = (f"{ev['speed_deg']:.1f}\u00b0/f"
                             if ev['speed_deg'] is not None
                             else f"{ev['speed_px']:.0f}px/f")
                plbl = resolve_display_pid(ev['face_id'], pid_map)
                rows.append((
                    f"{plbl} \u2192{ev['direction']}  "
                    f"{speed_str}  @f{ev['frame_no']}",
                    self._NS_COL,
                ))
            # Per-face tally
            counts: dict[int, int] = {}
            for ev in self.events:
                counts[ev['face_id']] = counts.get(ev['face_id'], 0) + 1
            tally = "  ".join(
                f"{resolve_display_pid(k, pid_map)}:{v}"
                for k, v in sorted(counts.items()))
            rows.append((f"total: {tally}", (160, 160, 160)))
        else:
            rows = [("--", DASH_DIM)]

        return draw_section(panel, y, "NOVEL SALIENCE", self._NS_COL, rows, line_h)

    def dashboard_data(self, *, pid_map=None) -> dict:
        rows = []
        if self.events:
            for ev in self.events[-3:]:
                speed_str = (f"{ev['speed_deg']:.1f}\u00b0/f"
                             if ev['speed_deg'] is not None
                             else f"{ev['speed_px']:.0f}px/f")
                plbl = resolve_display_pid(ev['face_id'], pid_map)
                rows.append({
                    'label': f"{plbl} \u2192{ev['direction']}",
                    'value': f"{speed_str}  @f{ev['frame_no']}",
                })
            counts: dict[int, int] = {}
            for ev in self.events:
                counts[ev['face_id']] = counts.get(ev['face_id'], 0) + 1
            tally = "  ".join(
                f"{resolve_display_pid(k, pid_map)}:{v}"
                for k, v in sorted(counts.items()))
            rows.append({'label': f"total: {tally}"})
        return {
            'title': 'NOVEL SALIENCE',
            'colour': self._NS_COL,
            'rows': rows,
            'empty_text': '--',
        }

    # ── CSV output ────────────────────────────────────────────────────────────

    def csv_rows(self, total_frames: int, *, pid_map=None) -> list:
        """
        Return one CSV section with all recorded novel-salience events.

        Columns: category, frame_no, face_id, direction,
                 speed_px, speed_deg, delta_x, delta_y
        """
        if not self.events:
            return []

        rows: list[list] = [
            [],  # blank separator
            ["novel_salience_events"],
            ["category", "frame_no", "face_id", "direction",
             "speed_px", "speed_deg", "delta_x", "delta_y"],
        ]
        for ev in self.events:
            rows.append([
                "novel_salience",
                ev['frame_no'],
                resolve_display_pid(ev['face_id'], pid_map),
                ev['direction'],
                f"{ev['speed_px']:.2f}",
                f"{ev['speed_deg']:.2f}" if ev['speed_deg'] is not None else "",
                f"{ev['delta_x']:.2f}",
                f"{ev['delta_y']:.2f}",
            ])

        # Summary row: total events per face
        counts: dict[int, int] = {}
        for ev in self.events:
            counts[ev['face_id']] = counts.get(ev['face_id'], 0) + 1
        rows.append([])
        rows.append(["novel_salience_summary", "face_id", "event_count",
                     "total_frames", "rate_pct"])
        for fid, cnt in sorted(counts.items()):
            rate = cnt / total_frames * 100 if total_frames else 0.0
            rows.append(["novel_salience_summary",
                         resolve_display_pid(fid, pid_map), cnt,
                         total_frames, f"{rate:.4f}"])
        return rows

    # ── CLI protocol ──────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add NovelSalience CLI flags to the argument parser."""
        g = parser.add_argument_group("Novel Salience plugin")
        g.add_argument(
            "--novel-salience",
            action="store_true",
            help="Enable novel-salience detection (rapid gaze-shift tracking).",
        )
        g.add_argument(
            "--ns-speed-thresh",
            type=float, default=40.0, metavar="PX",
            help=(
                "Gaze-endpoint speed threshold in pixels/frame to flag a "
                "novel-salience event (default: 40).  Lower = more sensitive."
            ),
        )
        g.add_argument(
            "--ns-cooldown",
            type=int, default=20, metavar="N",
            help=(
                "Minimum frames between consecutive novel-salience events for "
                "the same face track (default: 20)."
            ),
        )
        g.add_argument(
            "--ns-history",
            type=int, default=2, metavar="N",
            help=(
                "Sliding-window depth for velocity smoothing (default: 2).  "
                "1 = instantaneous; 3 = heavier smoothing."
            ),
        )
        g.add_argument(
            "--ns-flash",
            type=int, default=12, metavar="N",
            help="Frames the visual saccade indicator persists after an event (default: 12).",
        )

    @classmethod
    def from_args(cls, args):
        """Return an instance if ``--novel-salience`` was passed, else ``None``."""
        if not getattr(args, "novel_salience", False):
            return None
        inst = cls(
            speed_thresh = getattr(args, "ns_speed_thresh", 40.0),
            cooldown     = getattr(args, "ns_cooldown",     20),
            history      = getattr(args, "ns_history",      2),
            flash        = getattr(args, "ns_flash",        12),
        )
        print(
            f"NovelSalience: thresh={inst._thresh}px/f  "
            f"cooldown={inst._cooldown_frames}f  "
            f"history={inst._hist_len}f  flash={inst._flash_len}f"
        )
        return inst


# ── Exported symbol consumed by PluginRegistry.discover() ─────────────────────
PLUGIN_CLASS = NovelSalienceTracker
