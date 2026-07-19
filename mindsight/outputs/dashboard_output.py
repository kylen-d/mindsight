"""
outputs/dashboard_output.py — Display dashboard and frame overlay.

Responsibilities
----------------
- draw_overlay: draws gaze rays, object bounding boxes, face markers, and
  convergence circles onto the annotated video frame.
- compose_dashboard: builds the wide composite frame with left and right
  side panels showing all live phenomena metrics.
- Supporting drawing helpers: _draw_labelled_box, _face_colour,
  _draw_panel_section, _dash_line_h.
"""

import cv2
import numpy as np

from mindsight.constants import (
    BOX_THICKNESS,
    DASH_FONT_SCALE,
    DASH_PADDING,
    DASH_WIDTH,
    DWELL_INDICATOR_RADIUS,
    DWELL_MIN_FRACTION,
    OVERLAY_BLEND_ALPHA,
    OVERLAY_BLEND_BETA,
    UI_ARROW_LEFT,
    UI_LABEL_CONVERGE,
    UI_LABEL_GHOST,
    UI_LABEL_JOINT,
    UI_LABEL_LOCKED,
    get_colour,
)
from mindsight.pipeline_config import resolve_display_pid

# ── Overlay themes (v1.1 W3Z item 6) ──────────────────────────────────────────
# "classic" is the byte-exact historical look -- the regression goldens pin
# it, so its values must never drift.  "mindsight" restyles the overlay to
# the brand family sampled from the logo + app icon: deep-indigo ink
# (#1f1b37 wordmark background / #332747 icon halo), the brain's magenta
# (#b5367a) and the eye's jade (#4ec083) as hero accents, and the icon's
# warm off-white (#dbe1d4).  Ink label tabs with a coloured border replace
# solid saturated fills.  Geometry (box shapes, positions, sizes) is
# identical in both themes by design.
_INK      = (55, 27, 31)       # #1f1b37 -- wordmark/GUI background indigo
_INK_SEP  = (71, 39, 51)       # #332747 -- icon halo indigo
_MAGENTA  = (137, 68, 201)     # #c94489 -- icon brain magenta (lifted a touch)
_JADE     = (131, 192, 78)     # #4ec083 -- icon eye jade (light tone)
_GOLD     = (94, 199, 240)     # #f0c75e -- warm gold accent (complement)
_OFFWHITE = (212, 225, 219)    # #dbe1d4 -- icon off-white

OVERLAY_THEMES: dict[str, dict] = {
    "classic": dict(
        face_cols=[(100, 100, 255), (100, 255, 100), (255, 100, 100),
                   (255, 220, 50), (255, 80, 255), (80, 255, 255)],
        joint=(0, 200, 255), lock=(0, 215, 255), conv=(0, 220, 180),
        multi_watch=(0, 255, 0),
        label_fill=None,               # None -> solid box-colour tab
        label_text=(255, 255, 255),    # None -> box-colour text
        badge_bg=(20, 20, 20),
        lock_rim=(255, 255, 255),
        snap_fill=(180, 255, 120), snap_rim=(60, 180, 60),
        extend=(255, 165, 0), dwell=(255, 200, 0),
        footer_bg=(0, 0, 0),
        dash_bg=(18, 18, 18), dash_sep=(55, 55, 55),
        dash_head=(210, 210, 210), dash_dim=(70, 70, 70),
    ),
    "mindsight": dict(
        # Participant palette led by the logo pair (magenta, jade), then
        # complementary pastels that hold up on real footage.
        face_cols=[_MAGENTA, _JADE, (248, 158, 108), _GOLD,
                   (240, 138, 180), (200, 211, 94)],
        joint=_GOLD, lock=_MAGENTA, conv=_JADE,
        multi_watch=_JADE,
        label_fill=_INK,               # indigo ink tab, coloured border
        label_text=None,               # box-colour text on the ink tab
        badge_bg=_INK,
        lock_rim=_OFFWHITE,
        snap_fill=_JADE, snap_rim=_INK,
        extend=_GOLD, dwell=_GOLD,
        footer_bg=_INK,
        dash_bg=_INK, dash_sep=_INK_SEP,
        dash_head=_OFFWHITE, dash_dim=(100, 70, 85),
    ),
}

# ── Shared colour / font constants (rebound by set_overlay_theme) ─────────────
_FACE_COLS  = OVERLAY_THEMES["classic"]["face_cols"]
_JOINT_COL  = OVERLAY_THEMES["classic"]["joint"]
_LOCK_COL   = OVERLAY_THEMES["classic"]["lock"]
_CONV_COL   = OVERLAY_THEMES["classic"]["conv"]
_MULTI_WATCH_COL = OVERLAY_THEMES["classic"]["multi_watch"]
_LABEL_FILL = OVERLAY_THEMES["classic"]["label_fill"]
_LABEL_TEXT = OVERLAY_THEMES["classic"]["label_text"]
_BADGE_BG   = OVERLAY_THEMES["classic"]["badge_bg"]
_LOCK_RIM   = OVERLAY_THEMES["classic"]["lock_rim"]
_SNAP_FILL  = OVERLAY_THEMES["classic"]["snap_fill"]
_SNAP_RIM   = OVERLAY_THEMES["classic"]["snap_rim"]
_EXTEND_COL = OVERLAY_THEMES["classic"]["extend"]
_DWELL_COL  = OVERLAY_THEMES["classic"]["dwell"]
_FOOTER_BG  = OVERLAY_THEMES["classic"]["footer_bg"]
_FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ── Dashboard panel constants ─────────────────────────────────────────────────
_DASH_BG   = OVERLAY_THEMES["classic"]["dash_bg"]
_DASH_SEPL = OVERLAY_THEMES["classic"]["dash_sep"]
_DASH_HEAD = OVERLAY_THEMES["classic"]["dash_head"]
_DASH_DIM  = OVERLAY_THEMES["classic"]["dash_dim"]
_DASH_FS   = DASH_FONT_SCALE
_DASH_PAD  = DASH_PADDING
_DASH_W    = DASH_WIDTH

_ACTIVE_THEME = "classic"


def set_overlay_theme(name: str) -> None:
    """Select the overlay/dashboard theme (unknown names -> classic).

    Drawing happens on the single pipeline thread, so rebinding the module
    colour globals per call is safe and keeps every draw helper untouched.
    """
    global _ACTIVE_THEME, _FACE_COLS, _JOINT_COL, _LOCK_COL, _CONV_COL, \
        _MULTI_WATCH_COL, _LABEL_FILL, _LABEL_TEXT, _BADGE_BG, _LOCK_RIM, \
        _SNAP_FILL, _SNAP_RIM, _EXTEND_COL, _DWELL_COL, _FOOTER_BG, \
        _DASH_BG, _DASH_SEPL, _DASH_HEAD, _DASH_DIM
    if name == _ACTIVE_THEME:
        return
    t = OVERLAY_THEMES.get(name) or OVERLAY_THEMES["classic"]
    _ACTIVE_THEME = name if name in OVERLAY_THEMES else "classic"
    _FACE_COLS  = t["face_cols"]
    _JOINT_COL  = t["joint"]
    _LOCK_COL   = t["lock"]
    _CONV_COL   = t["conv"]
    _MULTI_WATCH_COL = t["multi_watch"]
    _LABEL_FILL = t["label_fill"]
    _LABEL_TEXT = t["label_text"]
    _BADGE_BG   = t["badge_bg"]
    _LOCK_RIM   = t["lock_rim"]
    _SNAP_FILL  = t["snap_fill"]
    _SNAP_RIM   = t["snap_rim"]
    _EXTEND_COL = t["extend"]
    _DWELL_COL  = t["dwell"]
    _FOOTER_BG  = t["footer_bg"]
    _DASH_BG    = t["dash_bg"]
    _DASH_SEPL  = t["dash_sep"]
    _DASH_HEAD  = t["dash_head"]
    _DASH_DIM   = t["dash_dim"]


def _face_colour(face_index):
    """Return a BGR colour for the given face index (cycles through 6 colours)."""
    return _FACE_COLS[face_index % len(_FACE_COLS)]


def _draw_labelled_box(frame, x1, y1, x2, y2, colour, label, thick=2):
    """Draw a bounding box with a filled label tab on *frame* (in-place).

    Classic theme: solid box-colour tab with white text (the golden-pinned
    look).  Navy theme: navy-ink tab with a 1 px coloured border and
    box-colour text.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
    (tw, th), bl = cv2.getTextSize(label, _FONT, 0.55, 1)
    fill = colour if _LABEL_FILL is None else _LABEL_FILL
    text = colour if _LABEL_TEXT is None else _LABEL_TEXT
    cv2.rectangle(frame, (x1, y1-th-bl-4), (x1+tw+4, y1), fill, -1)
    if _LABEL_FILL is not None:
        cv2.rectangle(frame, (x1, y1-th-bl-4), (x1+tw+4, y1), colour, 1)
    cv2.putText(frame, label, (x1+2, y1-bl-2), _FONT, 0.55, text, 1, cv2.LINE_AA)


def _dash_line_h() -> int:
    """Return the standard line height (px) for dashboard panel text."""
    return cv2.getTextSize("Ag", _FONT, _DASH_FS, 1)[0][1] + _DASH_PAD * 2 + 1


def _truncate_cv2(text: str, font, scale: float, thick: int,
                   max_w: int) -> str:
    """Truncate *text* with ellipsis so it fits within *max_w* pixels."""
    tw = cv2.getTextSize(text, font, scale, thick)[0][0]
    if tw <= max_w:
        return text
    ellipsis_w = cv2.getTextSize('...', font, scale, thick)[0][0]
    budget = max_w - ellipsis_w
    while text and cv2.getTextSize(text, font, scale, thick)[0][0] > budget:
        text = text[:-1]
    return text + '...'


def _draw_panel_section(panel, y: int, title: str, title_col,
                        rows: list, line_h: int) -> int:
    """Draw a labelled section into *panel* (a numpy view, modified in-place).

    Each entry in *rows* is ``(text, bgr_colour)``.
    Returns the y-coordinate immediately after the section.
    """
    pw = panel.shape[1]
    cv2.line(panel, (_DASH_PAD, y + 3), (pw - _DASH_PAD, y + 3), _DASH_SEPL, 1)
    y += 8
    cv2.putText(panel, title, (_DASH_PAD + 2, y + line_h - _DASH_PAD - 1),
                _FONT, _DASH_FS, title_col, 1, cv2.LINE_AA)
    y += line_h
    max_w = pw - _DASH_PAD * 4
    for text, col in rows:
        text = _truncate_cv2(text, _FONT, _DASH_FS, 1, max_w)
        cv2.putText(panel, text, (_DASH_PAD + 8, y + line_h - _DASH_PAD - 1),
                    _FONT, _DASH_FS, col, 1, cv2.LINE_AA)
        y += line_h
    return y + _DASH_PAD


# ══════════════════════════════════════════════════════════════════════════════
# Frame overlay — sub-functions
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_pid(fi, face_track_ids, pid_map=None):
    """Map a per-frame face index to ``(display_label, numeric_tid)``.

    *display_label* is the custom string (or ``"P{tid}"`` fallback).
    *numeric_tid* is the integer used for colour indexing.
    """
    tid = face_track_ids[fi] if face_track_ids and fi < len(face_track_ids) else fi
    return resolve_display_pid(tid, pid_map), tid


def _draw_object_boxes(frame, dets, obj_watchers, joint_objs, locked_set,
                       face_track_ids, pid_map=None):
    """Draw labelled bounding boxes for all detected objects.

    Colour and style vary by attention state:
    - Joint-attention objects get a double-outline in gold.
    - Locked objects get a crosshair marker.
    - Watched objects inherit the watcher's face colour.
    - Unwatched objects use the class-palette colour.
    """
    for oi, det in enumerate(dets):
        if det.get('_face_idx') is not None:
            continue  # face targets are drawn by _draw_face_markers instead
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        watchers = sorted(obj_watchers.get(oi, []))

        # Choose colour and label based on attention state (highest priority first)
        if oi in joint_objs:
            col, thick = _JOINT_COL, 3
            cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                          _JOINT_COL, 1)  # outer highlight ring
            label = (f"{det['class_name']} {det['conf']:.2f}"
                     f"  {UI_ARROW_LEFT} {UI_LABEL_JOINT}")
        elif oi in locked_set:
            col, thick = _LOCK_COL, 3
            # Draw a crosshair at the object centre to indicate gaze lock
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), _LOCK_COL, 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), _LOCK_COL, 2)
            cv2.circle(frame, (cx, cy), 4, _LOCK_COL, -1)
            label = (f"{det['class_name']} {det['conf']:.2f}"
                     f"  {UI_ARROW_LEFT} {UI_LABEL_LOCKED}")
        elif watchers:
            # Single watcher → their colour; multi-watcher → green
            _wtid = _resolve_pid(watchers[0], face_track_ids, pid_map)[1]
            col = (_face_colour(_wtid) if len(watchers) == 1
                   else _MULTI_WATCH_COL)
            thick = 3
            who = " ".join(
                _resolve_pid(fi, face_track_ids, pid_map)[0] for fi in watchers)
            label = f"{det['class_name']} {det['conf']:.2f}  {UI_ARROW_LEFT} {who}"
        else:
            col, thick = get_colour(det['cls_id']), BOX_THICKNESS
            label = f"{det['class_name']} {det['conf']:.2f}"

        # Ghost detections (persisted past their last real detection) are dimmed
        if det.get('_ghost', False):
            thick = 1
            label += f" {UI_LABEL_GHOST}"
            col = tuple(max(0, c - 80) for c in col)

        _draw_labelled_box(frame, x1, y1, x2, y2, col, label, thick)


def _draw_face_markers(frame, face_bboxes, face_track_ids, pid_map=None):
    """Draw thin rectangles and ID labels around each detected face."""
    if not face_bboxes:
        return
    for fi, (fx1, fy1, fx2, fy2) in enumerate(face_bboxes):
        plbl, tid = _resolve_pid(fi, face_track_ids, pid_map)
        fc = _face_colour(tid)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fc, 2)
        cv2.putText(frame, plbl, (fx1 + 2, fy1 + 14),
                    _FONT, 0.4, fc, 1, cv2.LINE_AA)


def _draw_gaze_rays(frame, persons_gaze, lock_info, ray_snapped, ray_extended,
                    face_targets, face_track_ids, gaze_cone_angle, gaze_debug,
                    pid_map=None):
    """Draw gaze rays (arrows), cone fills, endpoint markers, and participant badges.

    Each ray is drawn as an arrowed line from the gaze origin (eye centre) to
    the ray endpoint.  Visual indicators distinguish snap, lock, and extension
    states so the operator can quickly assess gaze accuracy.
    """
    h, w = frame.shape[:2]

    # Pre-compute cone trig constants once (same angle for all faces)
    _cone_cos_a = _cone_sin_a = None
    _cone_overlay = None
    _cone_edges = []          # collect (origin, left_end, right_end, colour) for edge lines
    if gaze_cone_angle > 0.0:
        _ar = np.radians(gaze_cone_angle)
        _cone_cos_a, _cone_sin_a = np.cos(_ar), np.sin(_ar)
        _cone_overlay = frame.copy()

    for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
        ox, oy = int(origin[0]), int(origin[1])
        # Clamp endpoint to frame bounds to avoid OpenCV drawing artifacts
        ex = int(np.clip(ray_end[0], 0, w - 1))
        ey = int(np.clip(ray_end[1], 0, h - 1))

        plbl, pid = _resolve_pid(fi, face_track_ids, pid_map)
        locked_oi  = lock_info[fi][0] if lock_info and fi < len(lock_info) else None
        dwell_frac = lock_info[fi][1] if lock_info and fi < len(lock_info) else 0.0
        snapped    = ray_snapped[fi]  if ray_snapped and fi < len(ray_snapped) else False
        extended   = ray_extended[fi] if ray_extended and fi < len(ray_extended) else False

        # Locked rays are highlighted in the lock colour; others use face colour
        arrow_col = _LOCK_COL if locked_oi is not None else _face_colour(pid)
        thick = 3 if locked_oi is not None else 2

        # Optional: draw cone polygon onto the shared overlay (blended once after loop)
        if _cone_overlay is not None:
            dv = np.array([ex - ox, ey - oy], float)
            dl = np.linalg.norm(dv)
            if dl > 1e-6:
                udir = dv / dl
                left = np.array([udir[0] * _cone_cos_a - udir[1] * _cone_sin_a,
                                 udir[0] * _cone_sin_a + udir[1] * _cone_cos_a])
                right = np.array([udir[0] * _cone_cos_a + udir[1] * _cone_sin_a,
                                  -udir[0] * _cone_sin_a + udir[1] * _cone_cos_a])
                le = (int(ox + left[0] * dl), int(oy + left[1] * dl))
                re = (int(ox + right[0] * dl), int(oy + right[1] * dl))
                pts = np.array([[ox, oy], le, re], dtype=np.int32)
                cv2.fillPoly(_cone_overlay, [pts], arrow_col)
                _cone_edges.append(((ox, oy), le, re, arrow_col))

        # Main gaze arrow from eye centre to ray endpoint
        cv2.arrowedLine(frame, (ox, oy), (ex, ey), arrow_col, thick,
                        cv2.LINE_AA, tipLength=0.05)
        cv2.circle(frame, (ox, oy), 6, arrow_col, -1)  # origin dot

        # Endpoint markers — visual feedback for snap/lock/extend state
        if locked_oi is not None:
            # Solid white-rimmed dot: gaze is locked onto an object
            cv2.circle(frame, (ex, ey), 8, _LOCK_COL, -1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 8, _LOCK_RIM, 1, cv2.LINE_AA)
        elif snapped:
            # Green filled dot: adaptive snap found a nearby object
            cv2.circle(frame, (ex, ey), 9, _SNAP_FILL, -1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 9, _SNAP_RIM, 1, cv2.LINE_AA)
        elif extended:
            # Orange ring: ray was extended toward a distant object
            cv2.circle(frame, (ex, ey), 9, _EXTEND_COL, 2, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 4, _EXTEND_COL, -1, cv2.LINE_AA)

        # Dwell-progress arc: shows how close the participant is to locking on
        if DWELL_MIN_FRACTION < dwell_frac < 1.0 and locked_oi is None:
            r = DWELL_INDICATOR_RADIUS
            cv2.ellipse(frame, (ox, oy), (r, r), -90, 0,
                        int(360 * dwell_frac), _DWELL_COL, 2, cv2.LINE_AA)

        # Participant badge: shows which objects this person is looking at
        targets = face_targets.get(fi, [])
        lock_tag = " [LOCK]" if locked_oi is not None else ""
        badge = f"{plbl}: " + (", ".join(targets) or "\u2013") + lock_tag
        (bw, bh), _ = cv2.getTextSize(badge, _FONT, 0.45, 1)
        # Place badge to the right of the origin; flip left if it would clip
        bx = ox + 10 if ox + 10 + bw + 6 <= w else ox - bw - 16
        by = oy - 10
        cv2.rectangle(frame, (bx - 2, by - bh - 4), (bx + bw + 4, by + 2),
                      _BADGE_BG, -1)
        cv2.rectangle(frame, (bx - 2, by - bh - 4), (bx + bw + 4, by + 2),
                      arrow_col, 1)
        cv2.putText(frame, badge, (bx, by), _FONT, 0.45, arrow_col, 1,
                    cv2.LINE_AA)

        # Debug overlay: raw pitch/yaw in degrees next to the badge
        if gaze_debug and angles:
            cv2.putText(frame, (f"p={np.degrees(angles[0]):.1f}\u00b0"
                                f" y={np.degrees(angles[1]):.1f}\u00b0"),
                        (bx, by + bh + 12), _FONT, 0.35,
                        _face_colour(pid), 1, cv2.LINE_AA)

    # Single-pass cone blend: all cone polygons drawn onto one overlay, blended once
    if _cone_overlay is not None and _cone_edges:
        cv2.addWeighted(_cone_overlay, OVERLAY_BLEND_ALPHA, frame,
                        OVERLAY_BLEND_BETA, 0, frame)
        for (o, le, re, col) in _cone_edges:
            cv2.line(frame, o, le, col, 1, cv2.LINE_AA)
            cv2.line(frame, o, re, col, 1, cv2.LINE_AA)


def _draw_gaze_footer(frame, persons_gaze, face_track_ids, pid_map=None):
    """Draw a bottom-of-frame bar showing each person's gaze-ray tip coordinates."""
    if not persons_gaze:
        return
    h, w = frame.shape[:2]
    resolved = [_resolve_pid(fi, face_track_ids, pid_map)
                for fi in range(len(persons_gaze))]
    labels = [
        f"{plbl}: ({int(re[0])},{int(re[1])})"
        for (plbl, _tid), (_, re, _) in zip(resolved, persons_gaze)
    ]
    widths = [cv2.getTextSize(lbl, _FONT, 0.55, 1)[0][0] for lbl in labels]
    total = sum(widths) + 10 * (len(labels) - 1) + 12
    cv2.rectangle(frame, (0, h - 32), (min(w, total), h), _FOOTER_BG, -1)
    x = 6
    for (lbl, lw), (_plbl, tid) in zip(zip(labels, widths), resolved):
        cv2.putText(frame, lbl, (x, h - 10), _FONT, 0.55,
                    _face_colour(tid), 1, cv2.LINE_AA)
        x += lw + 10


def _draw_convergence_markers(frame, tip_convergences, persons_gaze,
                              tip_radius, face_track_ids, pid_map=None):
    """Draw circles and connecting lines where multiple gaze rays converge."""
    for faces_set, centroid in (tip_convergences or []):
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(frame, (cx, cy), tip_radius, _CONV_COL, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, _CONV_COL, -1, cv2.LINE_AA)
        # Draw lines from each converging ray tip to the shared centroid
        for fi in faces_set:
            if fi < len(persons_gaze):
                tx = int(persons_gaze[fi][1][0])
                ty = int(persons_gaze[fi][1][1])
                cv2.line(frame, (tx, ty), (cx, cy), _CONV_COL, 1, cv2.LINE_AA)
        tag = "+".join(
            _resolve_pid(fi, face_track_ids, pid_map)[0] for fi in sorted(faces_set))
        cv2.putText(frame, f"{UI_LABEL_CONVERGE} {tag}",
                    (cx + tip_radius + 4, cy + 5), _FONT, 0.45,
                    _CONV_COL, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Face anonymization
# ══════════════════════════════════════════════════════════════════════════════

class AnonSmoother:
    """Temporal smoother for face anonymization boxes.

    Boxes grow instantly to cover the detected face but shrink slowly
    (exponential decay) so that momentary detector jitter or brief
    drop-outs don't cause flicker.  A grace period keeps the box alive
    for a few frames after the face disappears entirely.

    *shrink_rate* controls how fast the smoothed box decays toward the
    current detection each frame (0 = frozen, 1 = instant snap).
    """

    def __init__(self, grace: int = 15, shrink_rate: float = 0.05):
        self._grace = grace
        self._shrink = shrink_rate
        # track_id → [sx1, sy1, sx2, sy2, frames_since_seen]
        self._state: dict[int, list] = {}

    def update(self, face_bboxes, face_track_ids):
        """Merge current detections and return smoothed boxes."""
        seen = set()
        a = self._shrink
        for tid, (x1, y1, x2, y2) in zip(face_track_ids, face_bboxes):
            seen.add(tid)
            prev = self._state.get(tid)
            if prev is not None:
                px1, py1, px2, py2, _ = prev
                # Grow instantly, shrink slowly via EMA
                sx1 = min(x1, px1 + a * (x1 - px1)) if x1 > px1 else x1
                sy1 = min(y1, py1 + a * (y1 - py1)) if y1 > py1 else y1
                sx2 = max(x2, px2 + a * (x2 - px2)) if x2 < px2 else x2
                sy2 = max(y2, py2 + a * (y2 - py2)) if y2 < py2 else y2
                self._state[tid] = [int(sx1), int(sy1), int(sx2), int(sy2), 0]
            else:
                self._state[tid] = [x1, y1, x2, y2, 0]

        # Age unseen tracks; drop those past grace period
        expired = []
        for tid in self._state:
            if tid not in seen:
                s = self._state[tid]
                s[4] += 1
                if s[4] > self._grace:
                    expired.append(tid)
        for tid in expired:
            del self._state[tid]

        return [(s[0], s[1], s[2], s[3]) for s in self._state.values()]


def apply_face_anonymization(frame, face_bboxes, mode, padding=0.3,
                             face_track_ids=None, smoother=None):
    """Blur or black-out detected face regions in *frame* (in-place).

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    face_bboxes : list[tuple[int, int, int, int]]
        Face bounding boxes as ``(x1, y1, x2, y2)`` in pixel coordinates.
    mode : str
        ``"blur"`` for heavy Gaussian blur, ``"black"`` for solid fill.
    padding : float
        Fraction of bbox width/height added on each side (default 0.3).
    face_track_ids : list[int] | None
        Per-face stable track IDs (required when *smoother* is used).
    smoother : AnonSmoother | None
        Optional temporal smoother to prevent flicker.
    """
    if smoother is not None and face_track_ids is not None:
        boxes = smoother.update(face_bboxes, face_track_ids)
    else:
        boxes = face_bboxes
    if not boxes:
        return
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        if x2 <= x1 or y2 <= y1:
            continue
        if mode == "blur":
            roi = frame[y1:y2, x1:x2]
            k = max(51, (x2 - x1) // 2 * 2 + 1)  # odd kernel scaled to face
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 30)
        elif mode == "black":
            frame[y1:y2, x1:x2] = 0


# ══════════════════════════════════════════════════════════════════════════════
# Frame overlay — main entry point
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(ctx, *, gaze_cfg=None, **kwargs):
    """Annotate a video frame with detection boxes, gaze rays, and phenomena indicators.

    Reads per-frame data from *ctx* and gaze display settings from *gaze_cfg*.
    Modifies ``ctx['frame']`` in-place and returns it.

    The overlay is drawn in layers (back-to-front):
      1. Object bounding boxes with attention-state colouring
      2. Face rectangles with Re-ID labels
      3. Gaze rays, cone fills, and endpoint markers
      4. Bottom coordinate bar
      5. Convergence circles
    """
    frame          = ctx['frame']
    persons_gaze   = ctx.get('persons_gaze', [])
    dets           = ctx.get('all_targets', [])
    hits           = ctx.get('hits', set())
    joint_objs     = ctx.get('confirmed_objs') or ctx.get('joint_objs') or set()
    lock_info      = ctx.get('lock_info')
    ray_snapped    = ctx.get('ray_snapped')
    ray_extended   = ctx.get('ray_extended')
    tip_convergences = ctx.get('tip_convergences')
    face_bboxes    = ctx.get('face_bboxes')
    face_track_ids = ctx.get('face_track_ids')
    pid_map        = ctx.get('pid_map')

    lite_overlay    = getattr(gaze_cfg, '_lite_overlay', False) if gaze_cfg else False
    gaze_debug      = False if lite_overlay else (gaze_cfg.gaze_debug if gaze_cfg else False)
    tip_radius      = gaze_cfg.tip_radius if gaze_cfg else 80
    gaze_cone_angle = 0.0 if lite_overlay else (gaze_cfg.gaze_cone_angle if gaze_cfg else 0.0)
    set_overlay_theme(
        getattr(gaze_cfg, '_overlay_theme', 'classic') if gaze_cfg else 'classic')

    # Pre-compute per-object watcher lists and per-face target names
    obj_watchers, face_targets = {}, {}
    for fi, oi in hits:
        obj_watchers.setdefault(oi, []).append(fi)
        if oi < len(dets):
            face_targets.setdefault(fi, []).append(dets[oi]['class_name'])

    locked_set = {oi for oi, _ in (lock_info or []) if oi is not None}

    # Layer 1: Object bounding boxes
    _draw_object_boxes(frame, dets, obj_watchers, joint_objs, locked_set,
                       face_track_ids, pid_map)

    # Layer 2: Face rectangles
    _draw_face_markers(frame, face_bboxes, face_track_ids, pid_map)

    # Layer 3: Gaze rays, cones, and participant badges
    _draw_gaze_rays(frame, persons_gaze, lock_info, ray_snapped, ray_extended,
                    face_targets, face_track_ids, gaze_cone_angle, gaze_debug,
                    pid_map)

    # Layer 4: Bottom coordinate bar
    _draw_gaze_footer(frame, persons_gaze, face_track_ids, pid_map)

    # Layer 5: Convergence circles
    _draw_convergence_markers(frame, tip_convergences, persons_gaze,
                              tip_radius, face_track_ids, pid_map)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard compositor
# ══════════════════════════════════════════════════════════════════════════════

# Singleton matplotlib renderer (created on first call, reused across frames)
_mpl_renderer = None


def compose_dashboard(ctx, **kwargs):
    """Build a wide composite frame: [left panel | annotated video | right panel].

    Uses a matplotlib-based renderer that produces clean, styled dashboard
    panels with proper text rendering, bar charts, and consistent card layout.

    Joint Attention is rendered as a first-class phenomenon card alongside all
    other trackers — there is no special-cased section.

    Each tracker provides structured data via ``dashboard_data()``; trackers
    that only implement the legacy ``dashboard_section()`` are skipped by the
    matplotlib renderer.
    """
    global _mpl_renderer
    if _mpl_renderer is None:
        from mindsight.outputs.dashboard_matplotlib import DashboardRenderer
        _mpl_renderer = DashboardRenderer()

    return _mpl_renderer.render(ctx['frame'], ctx)
