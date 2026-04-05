"""
DataCollection/heatmap_output.py — Per-participant scene-based gaze heatmaps.

Responsibilities
----------------
- generate_participant_heatmap : render a single Gaussian density heatmap
  for one participant's gaze endpoints blended over a background frame.
- extract_mid_frame            : seek to the middle frame of a video and
  return it as the background for heatmap rendering.
- save_heatmaps                : iterate over all tracked participants,
  generate their heatmaps and write PNG files to a user-specified path.
"""

from pathlib import Path

import cv2
import numpy as np

from ms.constants import HEATMAP_ALPHA, HEATMAP_SIGMA
from ms.constants import OUTPUTS_ROOT as _OUTPUTS_ROOT
from ms.pipeline_config import resolve_display_pid


def resolve_heatmap_path(heatmap_arg, source) -> "str | None":
    """Resolve the --heatmap flag value to a path prefix or None.

    heatmap_arg : True  → Outputs/heatmaps/[stem]_Heatmap_Output (one PNG per participant)
                  str   → that path (treated as directory or prefix by save_heatmaps)
                  None/False → None (no heatmaps written)
    source      : video file path (str/Path) or webcam index (int).
    """
    if not heatmap_arg:
        return None
    if heatmap_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        return str(_OUTPUTS_ROOT / "heatmaps" / f"{stem}_Heatmap_Output")
    return heatmap_arg


def extract_mid_frame(source) -> np.ndarray | None:
    """Return the middle frame of *source* as a BGR numpy array.

    Parameters
    ----------
    source : video file path (str/Path) or webcam index (int).

    Returns
    -------
    H×W×3 BGR frame, or None if the video cannot be opened / seeked.
    """
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def generate_participant_heatmap(bg_frame: np.ndarray,
                                 gaze_points: list,
                                 sigma: int = HEATMAP_SIGMA,
                                 alpha: float = HEATMAP_ALPHA) -> np.ndarray:
    """Render a gaze density heatmap overlaid on *bg_frame*.

    Parameters
    ----------
    bg_frame    : H×W×3 BGR background image (e.g. middle video frame).
    gaze_points : sequence of (x, y) screen-space gaze endpoints.
    sigma       : Gaussian blur radius in pixels — controls spread of each
                  gaze sample.  Larger values give a smoother, wider heat blob.
    alpha       : maximum heatmap blend weight [0–1].  0 = background only,
                  1 = heatmap fully replaces background at peak density.

    Returns
    -------
    H×W×3 BGR image with the heatmap blended over the background.
    """
    h, w = bg_frame.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)

    for pt in gaze_points:
        x, y = pt[0], pt[1]
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            acc[yi, xi] += 1.0

    blurred = cv2.GaussianBlur(acc, (0, 0), sigma)
    mx = float(blurred.max())
    if mx > 0:
        blurred /= mx

    heat_u8    = (blurred * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # Blend only where density is meaningful — avoids flat blue tint everywhere
    weight  = (blurred[:, :, np.newaxis] * alpha).astype(np.float32)
    blended = (bg_frame.astype(np.float32) * (1.0 - weight) +
               heat_color.astype(np.float32) * weight).astype(np.uint8)
    return blended


def save_heatmaps(heatmap_path: str,
                  source,
                  bg_frame: np.ndarray,
                  heatmap_gaze: dict,
                  sigma: int = 40,
                  alpha: float = 0.65,
                  pid_map: dict = None) -> None:
    """Generate and save one heatmap PNG per participant.

    Parameters
    ----------
    heatmap_path : output directory or path prefix.
                   • If the path has no file extension it is treated as a
                     directory; files are written as
                     ``<dir>/<source_stem>_P<id>_heatmap.png``.
                   • If it has an extension the parent directory is used and
                     the stem becomes the filename prefix.
    source       : original video source path or webcam index (int).
                   Used only to derive a meaningful stem for file names.
    bg_frame     : representative background frame (H×W×3 BGR).
    heatmap_gaze : dict mapping face_track_id (int) -> list of (x, y) points.
    sigma        : forwarded to :func:`generate_participant_heatmap`.
    alpha        : forwarded to :func:`generate_participant_heatmap`.
    """
    out = Path(heatmap_path)
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        stem   = Path(str(source)).stem if not isinstance(source, int) else "webcam"
        prefix = out / stem
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        prefix = out.parent / out.stem

    for tid, pts in sorted(heatmap_gaze.items()):
        if not pts:
            continue
        img  = generate_participant_heatmap(bg_frame, pts, sigma=sigma, alpha=alpha)
        plbl = resolve_display_pid(tid, pid_map)
        name = str(prefix) + f"_{plbl}_heatmap.png"
        cv2.imwrite(name, img)
        print(f"Heatmap \u2192 {name}  ({len(pts)} gaze samples)")
