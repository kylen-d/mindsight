"""
DataCollection/dashboard_output.py — Display dashboard and frame overlay.

Responsibilities
----------------
- draw_overlay: draws gaze rays, object bounding boxes, face markers, and
  convergence circles onto the annotated video frame.
- compose_dashboard: builds the wide composite frame with left and right
  side panels showing all live phenomena metrics.
- Supporting drawing helpers: _draw_labelled_box, _face_colour,
  _draw_panel_section, _dash_line_h.
"""

from pathlib import Path

import cv2
import numpy as np

from constants import (
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
from constants import OUTPUTS_ROOT as _OUTPUTS_ROOT
from pipeline_config import resolve_display_pid

# ── Shared colour / font constants ────────────────────────────────────────────
_FACE_COLS  = [(100,100,255),(100,255,100),(255,100,100),(255,220,50),(255,80,255),(80,255,255)]
_JOINT_COL  = (0, 200, 255)
_LOCK_COL   = (0, 215, 255)
_CONV_COL   = (0, 220, 180)
_FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ── Dashboard panel constants ─────────────────────────────────────────────────
_DASH_BG   = (18, 18, 18)     # panel fill
_DASH_SEPL = (55, 55, 55)     # separator / border colour
_DASH_HEAD = (210, 210, 210)  # section heading colour
_DASH_DIM  = (70, 70, 70)     # placeholder / no-data colour
_DASH_FS   = DASH_FONT_SCALE
_DASH_PAD  = DASH_PADDING
_DASH_W    = DASH_WIDTH


def open_video_writer(save_arg, source, cap, *, no_dashboard=False):
    """Create and return a (VideoWriter, path) tuple for the annotated output.

    Parameters
    ----------
    save_arg     : True  → write to Outputs/Video/[stem]_Video_Output.mp4
                   str   → write to that path
                   None/False → do not record; returns (None, None)
    source       : video file path (str/Path) or webcam index (int).
    cap          : open cv2.VideoCapture used to query FPS and frame size.
    no_dashboard : if True, frames are raw (no side panels), so the writer
                   is sized to the original video dimensions.

    Returns
    -------
    (cv2.VideoWriter, str) or (None, None)
    """
    if not save_arg:
        return None, None
    if save_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        path = str(_OUTPUTS_ROOT / "Video" / f"{stem}_Video_Output.mp4")
    else:
        path = save_arg
    fps0   = cap.get(cv2.CAP_PROP_FPS) or 30
    fw, fh = int(cap.get(3)), int(cap.get(4))
    if no_dashboard:
        out_w = fw
    else:
        panel_w = max(280, int(fw * 0.22))
        out_w = fw + 2 * panel_w
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps0, (out_w, fh))
    print(f"Saving \u2192 {path}")
    return writer, path


def finalize_video(path):
    """Remux mp4v video to H.264 via ffmpeg for broad player compatibility.

    If ffmpeg is not available, the original mp4v file is kept as-is
    (playable in VLC and most players, but not QuickTime on macOS).
    """
    if path is None:
        return
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        print("Note: ffmpeg not found; video saved as MPEG-4 Part 2 (mp4v).\n"
              "      Install ffmpeg for H.264 output (QuickTime compatible).")
        return

    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             tmp],
            check=True, capture_output=True,
        )
        shutil.move(tmp, path)
        print(f"Video remuxed to H.264 \u2192 {path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg failed — keep the original mp4v file
        if Path(tmp).exists():
            Path(tmp).unlink()
        print("Note: H.264 remux failed; video saved as mp4v.")


def _face_colour(face_index):
    """Return a BGR colour for the given face index (cycles through 6 colours)."""
    return _FACE_COLS[face_index % len(_FACE_COLS)]


def _draw_labelled_box(frame, x1, y1, x2, y2, colour, label, thick=2):
    """Draw a bounding box with a filled label tab on *frame* (in-place)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
    (tw, th), bl = cv2.getTextSize(label, _FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1-th-bl-4), (x1+tw+4, y1), colour, -1)
    cv2.putText(frame, label, (x1+2, y1-bl-2), _FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)


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
                   else (0, 255, 0))
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
            cv2.circle(frame, (ex, ey), 8, (255, 255, 255), 1, cv2.LINE_AA)
        elif snapped:
            # Green filled dot: adaptive snap found a nearby object
            cv2.circle(frame, (ex, ey), 9, (180, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 9, (60, 180, 60), 1, cv2.LINE_AA)
        elif extended:
            # Orange ring: ray was extended toward a distant object
            cv2.circle(frame, (ex, ey), 9, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 4, (255, 165, 0), -1, cv2.LINE_AA)

        # Dwell-progress arc: shows how close the participant is to locking on
        if DWELL_MIN_FRACTION < dwell_frac < 1.0 and locked_oi is None:
            r = DWELL_INDICATOR_RADIUS
            cv2.ellipse(frame, (ox, oy), (r, r), -90, 0,
                        int(360 * dwell_frac), (255, 200, 0), 2, cv2.LINE_AA)

        # Participant badge: shows which objects this person is looking at
        targets = face_targets.get(fi, [])
        lock_tag = " [LOCK]" if locked_oi is not None else ""
        badge = f"{plbl}: " + (", ".join(targets) or "\u2013") + lock_tag
        (bw, bh), _ = cv2.getTextSize(badge, _FONT, 0.45, 1)
        # Place badge to the right of the origin; flip left if it would clip
        bx = ox + 10 if ox + 10 + bw + 6 <= w else ox - bw - 16
        by = oy - 10
        cv2.rectangle(frame, (bx - 2, by - bh - 4), (bx + bw + 4, by + 2),
                      (20, 20, 20), -1)
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
    cv2.rectangle(frame, (0, h - 32), (min(w, total), h), (0, 0, 0), -1)
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
        from DataCollection.dashboard_matplotlib import DashboardRenderer
        _mpl_renderer = DashboardRenderer()

    return _mpl_renderer.render(ctx['frame'], ctx)
