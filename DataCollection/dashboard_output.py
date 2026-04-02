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

from constants import (get_colour, OUTPUTS_ROOT as _OUTPUTS_ROOT,
                       DASH_WIDTH, DASH_FONT_SCALE, DASH_PADDING,
                       OVERLAY_BLEND_ALPHA, OVERLAY_BLEND_BETA,
                       DWELL_INDICATOR_RADIUS, DWELL_MIN_FRACTION,
                       BOX_THICKNESS,
                       UI_ARROW_LEFT, UI_LABEL_JOINT, UI_LABEL_LOCKED,
                       UI_LABEL_GHOST, UI_LABEL_CONVERGE)

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


def open_video_writer(save_arg, source, cap):
    """Create and return a VideoWriter for the annotated output video.

    Parameters
    ----------
    save_arg : True  → write to Outputs/Video/[stem]_Video_Output.mp4
               str   → write to that path
               None/False → do not record; returns None
    source   : video file path (str/Path) or webcam index (int).
    cap      : open cv2.VideoCapture used to query FPS and frame size.
    """
    if not save_arg:
        return None
    if save_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        path = str(_OUTPUTS_ROOT / "Video" / f"{stem}_Video_Output.mp4")
    else:
        path = save_arg
    fps0   = cap.get(cv2.CAP_PROP_FPS) or 30
    fw, fh = int(cap.get(3)), int(cap.get(4))
    out_w  = fw + 2 * _DASH_W
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps0, (out_w, fh))
    print(f"Saving \u2192 {path}")
    return writer


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
    for text, col in rows:
        while text and cv2.getTextSize(text, _FONT, _DASH_FS, 1)[0][0] > pw - _DASH_PAD * 4:
            text = text[:-1]
        cv2.putText(panel, text, (_DASH_PAD + 8, y + line_h - _DASH_PAD - 1),
                    _FONT, _DASH_FS, col, 1, cv2.LINE_AA)
        y += line_h
    return y + _DASH_PAD


# ══════════════════════════════════════════════════════════════════════════════
# Frame overlay — sub-functions
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_pid(fi, face_track_ids):
    """Map a per-frame face list index to the stable Re-ID track ID."""
    if face_track_ids and fi < len(face_track_ids):
        return face_track_ids[fi]
    return fi


def _draw_object_boxes(frame, dets, obj_watchers, joint_objs, locked_set,
                       face_track_ids):
    """Draw labelled bounding boxes for all detected objects.

    Colour and style vary by attention state:
    - Joint-attention objects get a double-outline in gold.
    - Locked objects get a crosshair marker.
    - Watched objects inherit the watcher's face colour.
    - Unwatched objects use the class-palette colour.
    """
    for oi, det in enumerate(dets):
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
            col = (_face_colour(watchers[0]) if len(watchers) == 1
                   else (0, 255, 0))
            thick = 3
            who = " ".join(
                f"P{_resolve_pid(fi, face_track_ids)}" for fi in watchers)
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


def _draw_face_markers(frame, face_bboxes, face_track_ids):
    """Draw thin rectangles and ID labels around each detected face."""
    if not face_bboxes:
        return
    for fi, (fx1, fy1, fx2, fy2) in enumerate(face_bboxes):
        pid = _resolve_pid(fi, face_track_ids)
        fc = _face_colour(pid)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fc, 2)
        cv2.putText(frame, f"F{pid}", (fx1 + 2, fy1 + 14),
                    _FONT, 0.4, fc, 1, cv2.LINE_AA)


def _draw_gaze_rays(frame, persons_gaze, lock_info, ray_snapped, ray_extended,
                    face_targets, face_track_ids, gaze_cone_angle, gaze_debug):
    """Draw gaze rays (arrows), cone fills, endpoint markers, and participant badges.

    Each ray is drawn as an arrowed line from the gaze origin (eye centre) to
    the ray endpoint.  Visual indicators distinguish snap, lock, and extension
    states so the operator can quickly assess gaze accuracy.
    """
    h, w = frame.shape[:2]

    for fi, (origin, ray_end, angles) in enumerate(persons_gaze):
        ox, oy = int(origin[0]), int(origin[1])
        # Clamp endpoint to frame bounds to avoid OpenCV drawing artifacts
        ex = int(np.clip(ray_end[0], 0, w - 1))
        ey = int(np.clip(ray_end[1], 0, h - 1))

        pid = _resolve_pid(fi, face_track_ids)
        locked_oi  = lock_info[fi][0] if lock_info and fi < len(lock_info) else None
        dwell_frac = lock_info[fi][1] if lock_info and fi < len(lock_info) else 0.0
        snapped    = ray_snapped[fi]  if ray_snapped and fi < len(ray_snapped) else False
        extended   = ray_extended[fi] if ray_extended and fi < len(ray_extended) else False

        # Locked rays are highlighted in the lock colour; others use face colour
        arrow_col = _LOCK_COL if locked_oi is not None else _face_colour(pid)
        thick = 3 if locked_oi is not None else 2

        # Optional: draw a semi-transparent vision cone when gaze_cone is active
        if gaze_cone_angle > 0.0:
            dv = np.array([ex - ox, ey - oy], float)
            dl = np.linalg.norm(dv)
            if dl > 1e-6:
                udir = dv / dl
                angle_rad = np.radians(gaze_cone_angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                # Rotate the unit direction by +/- cone half-angle
                left = np.array([udir[0] * cos_a - udir[1] * sin_a,
                                 udir[0] * sin_a + udir[1] * cos_a])
                right = np.array([udir[0] * cos_a + udir[1] * sin_a,
                                  -udir[0] * sin_a + udir[1] * cos_a])
                le = (int(ox + left[0] * dl), int(oy + left[1] * dl))
                re = (int(ox + right[0] * dl), int(oy + right[1] * dl))
                # Alpha-blend the cone polygon over the frame
                overlay = frame.copy()
                pts = np.array([[ox, oy], le, re], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], arrow_col)
                cv2.addWeighted(overlay, OVERLAY_BLEND_ALPHA, frame,
                                OVERLAY_BLEND_BETA, 0, frame)
                cv2.line(frame, (ox, oy), le, arrow_col, 1, cv2.LINE_AA)
                cv2.line(frame, (ox, oy), re, arrow_col, 1, cv2.LINE_AA)

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
        badge = f"P{pid}: " + (", ".join(targets) or "\u2013") + lock_tag
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
                        _face_colour(pid), 1)


def _draw_gaze_footer(frame, persons_gaze, face_track_ids):
    """Draw a bottom-of-frame bar showing each person's gaze-ray tip coordinates."""
    if not persons_gaze:
        return
    h, w = frame.shape[:2]
    labels = [
        f"P{_resolve_pid(fi, face_track_ids)}: ({int(re[0])},{int(re[1])})"
        for fi, (_, re, _) in enumerate(persons_gaze)
    ]
    widths = [cv2.getTextSize(lbl, _FONT, 0.55, 1)[0][0] for lbl in labels]
    total = sum(widths) + 10 * (len(labels) - 1) + 12
    cv2.rectangle(frame, (0, h - 32), (min(w, total), h), (0, 0, 0), -1)
    x = 6
    for fi, (lbl, lw) in enumerate(zip(labels, widths)):
        pid = _resolve_pid(fi, face_track_ids)
        cv2.putText(frame, lbl, (x, h - 10), _FONT, 0.55,
                    _face_colour(pid), 1, cv2.LINE_AA)
        x += lw + 10


def _draw_convergence_markers(frame, tip_convergences, persons_gaze,
                              tip_radius, face_track_ids):
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
            f"P{_resolve_pid(fi, face_track_ids)}" for fi in sorted(faces_set))
        cv2.putText(frame, f"{UI_LABEL_CONVERGE} {tag}",
                    (cx + tip_radius + 4, cy + 5), _FONT, 0.45,
                    _CONV_COL, 1, cv2.LINE_AA)


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
    joint_objs     = ctx.get('joint_objs') or set()
    lock_info      = ctx.get('lock_info')
    ray_snapped    = ctx.get('ray_snapped')
    ray_extended   = ctx.get('ray_extended')
    tip_convergences = ctx.get('tip_convergences')
    face_bboxes    = ctx.get('face_bboxes')
    face_track_ids = ctx.get('face_track_ids')

    gaze_debug      = gaze_cfg.gaze_debug if gaze_cfg else False
    tip_radius      = gaze_cfg.tip_radius if gaze_cfg else 80
    gaze_cone_angle = gaze_cfg.gaze_cone_angle if gaze_cfg else 0.0

    # Pre-compute per-object watcher lists and per-face target names
    obj_watchers, face_targets = {}, {}
    for fi, oi in hits:
        obj_watchers.setdefault(oi, []).append(fi)
        if oi < len(dets):
            face_targets.setdefault(fi, []).append(dets[oi]['class_name'])

    locked_set = {oi for oi, _ in (lock_info or []) if oi is not None}

    # Layer 1: Object bounding boxes
    _draw_object_boxes(frame, dets, obj_watchers, joint_objs, locked_set,
                       face_track_ids)

    # Layer 2: Face rectangles
    _draw_face_markers(frame, face_bboxes, face_track_ids)

    # Layer 3: Gaze rays, cones, and participant badges
    _draw_gaze_rays(frame, persons_gaze, lock_info, ray_snapped, ray_extended,
                    face_targets, face_track_ids, gaze_cone_angle, gaze_debug)

    # Layer 4: Bottom coordinate bar
    _draw_gaze_footer(frame, persons_gaze, face_track_ids)

    # Layer 5: Convergence circles
    _draw_convergence_markers(frame, tip_convergences, persons_gaze,
                              tip_radius, face_track_ids)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard compositor
# ══════════════════════════════════════════════════════════════════════════════

def compose_dashboard(ctx, **kwargs):
    """Build a wide composite frame: [left panel | annotated video | right panel].

    Reads from ctx: frame, fps, n_dets (len of hit_events), joint_pct,
    confirmed_objs, objects (dets), extra_hud, all_trackers.

    The ``frame`` should already have gaze rays, bounding-boxes, and face
    markers drawn on it.  All text data is placed in the flanking side panels so
    nothing overlaps the video region.

    Left panel  — System stats · Joint Attention · left-panel trackers
    Right panel — right-panel trackers

    Each tracker's ``dashboard_section()`` method draws its own section into the
    panel indicated by its ``dashboard_panel`` attribute (``"left"`` or ``"right"``).
    """
    frame = ctx['frame']
    fps = ctx.get('fps', 0.0)
    n_dets = ctx.get('n_dets', 0)
    joint_pct = ctx.get('joint_pct', 0.0)
    confirmed_objs = ctx.get('confirmed_objs', set())
    dets = ctx.get('objects', [])
    extra_hud = ctx.get('extra_hud')
    all_trackers = ctx.get('all_trackers')

    h, w = frame.shape[:2]
    lw = rw = _DASH_W
    canvas = np.full((h, w + lw + rw, 3), _DASH_BG, dtype=np.uint8)

    canvas[:, lw:lw + w] = frame
    cv2.line(canvas, (lw, 0),     (lw, h - 1),     _DASH_SEPL, 1)
    cv2.line(canvas, (lw + w, 0), (lw + w, h - 1), _DASH_SEPL, 1)

    line_h = _dash_line_h()

    # ── LEFT PANEL — built-in system sections ─────────────────────────────────
    lp = canvas[:, :lw]
    y_left = 8

    y_left = _draw_panel_section(lp, y_left, "SYSTEM", _DASH_HEAD, [
        (f"FPS : {fps:5.1f}", _DASH_HEAD),
        (f"Dets: {n_dets:4d}", _DASH_HEAD),
    ], line_h)

    ja_rows: list = []
    if confirmed_objs and dets:
        names = [dets[oi]['class_name'] for oi in sorted(confirmed_objs) if oi < len(dets)]
        ja_rows.append((f"@ {', '.join(names)}", (0, 255, 255)))
    else:
        ja_rows.append(("No joint attention", _DASH_DIM))
    ja_rows.append((f"JA frames: {joint_pct:.1f}%", (180, 180, 100)))
    if extra_hud:
        ja_rows.append((extra_hud, (180, 180, 100)))
    y_left = _draw_panel_section(lp, y_left, "JOINT ATTENTION", _JOINT_COL, ja_rows, line_h)

    # ── RIGHT PANEL — starts empty ───────────────────────────────────────────
    rp = canvas[:, lw + w:]
    y_right = 8

    # ── All trackers (built-in + plugins) draw their own sections ────────────
    panel_y = {'left': y_left, 'right': y_right}
    for tracker in (all_trackers or []):
        chosen = getattr(tracker, 'dashboard_panel', 'right')
        panel  = lp if chosen == 'left' else rp
        cur_y  = panel_y[chosen]
        if cur_y < panel.shape[0] - line_h * 3:
            new_y = tracker.dashboard_section(panel, cur_y, line_h)
            panel_y[chosen] = new_y

    return canvas
