"""
Plugins/Phenomena/Pupillometry/iris_extraction.py -- Pupil/iris measurement.

Two modes:
  - RGB: MediaPipe iris landmarks + dark-region thresholding within iris contour.
  - IR:  OpenCV-only grayscale thresholding for infrared eye cameras.

Both return a pupil/iris diameter ratio (distance-independent).
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_ear(iris_data) -> float | None:
    """Compute the Eye Aspect Ratio (EAR) from MediaPipe iris data.

    Uses the 6-point EAR formula:
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

    Returns the average EAR of both eyes, or ``None`` if landmarks are
    unavailable.
    """
    if iris_data is None:
        return None

    ears = []
    for side in ('right', 'left'):
        if not getattr(iris_data, f'{side}_valid', False):
            continue
        eye_contour = getattr(iris_data, f'{side}_eye_contour', None)
        if eye_contour is None or len(eye_contour) < 6:
            continue

        # MediaPipe eye contour points: approximate EAR from the
        # eye outline landmarks (upper/lower lid distances vs width)
        pts = eye_contour
        # Horizontal (eye width): points 0 and 3 (or endpoints)
        horizontal = np.linalg.norm(pts[0] - pts[3])
        if horizontal < 1.0:
            continue
        # Vertical: approximate from upper/lower lid midpoints
        # Use points 1,5 and 2,4 as upper/lower pairs
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        ear = (v1 + v2) / (2.0 * horizontal)
        ears.append(ear)

    return float(np.mean(ears)) if ears else None


def measure_rgb(face_crop: np.ndarray, iris_data, *,
                upscale: float = 2.0) -> dict | None:
    """
    Measure pupil/iris ratio from an RGB face crop using MediaPipe iris data.

    Parameters
    ----------
    face_crop : BGR numpy array of the face region.
    iris_data : IrisData from ``extract_iris_data()``.
    upscale   : Factor to upscale the crop before processing (default 2.0).

    Returns
    -------
    dict with keys: pupil_radius, iris_radius, ratio, eye ('right'|'left'|'avg'),
    left_ratio, right_ratio (individual eye ratios when available),
    or ``None`` if measurement failed.
    """
    if iris_data is None:
        return None

    results = []
    offsets = []  # normalized iris offset from eye center per side

    for side in ('right', 'left'):
        valid = getattr(iris_data, f'{side}_valid')
        if not valid:
            continue

        iris_center = getattr(iris_data, f'{side}_iris_center')
        iris_contour = getattr(iris_data, f'{side}_iris_contour')
        eye_contour = getattr(iris_data, f'{side}_eye_contour')

        if iris_center is None or iris_contour is None:
            continue

        # Iris diameter from landmark spread
        iris_dists = np.linalg.norm(iris_contour - iris_center, axis=1)
        iris_radius = float(np.mean(iris_dists))
        if iris_radius < 1.0:
            continue

        # Compute iris offset from eye center (normalized by eye width)
        eye_center = np.mean(eye_contour, axis=0)
        eye_width = np.linalg.norm(eye_contour[0] - eye_contour[1])
        if eye_width > 1:
            offsets.append((iris_center - eye_center) / eye_width)

        # Extract ROI around iris for pupil segmentation
        scale = upscale
        cx, cy = int(iris_center[0] * scale), int(iris_center[1] * scale)
        roi_r = int(iris_radius * scale * 1.5)

        if scale != 1.0:
            h, w = face_crop.shape[:2]
            crop_up = cv2.resize(face_crop, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)
        else:
            crop_up = face_crop

        ch, cw = crop_up.shape[:2]
        y1 = max(0, cy - roi_r)
        y2 = min(ch, cy + roi_r)
        x1 = max(0, cx - roi_r)
        x2 = min(cw, cx + roi_r)

        if y2 - y1 < 4 or x2 - x1 < 4:
            continue

        roi = crop_up[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Create iris mask from contour
        center_scaled = (iris_center * scale).astype(np.int32)
        center_local = center_scaled - np.array([x1, y1])

        mask = np.zeros(gray.shape, dtype=np.uint8)
        iris_r_scaled = int(iris_radius * scale)
        cv2.circle(mask, tuple(center_local), iris_r_scaled, 255, -1)

        # Threshold dark region within iris to find pupil
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Adaptive threshold: pupil is darkest region
        valid_pixels = gray[mask > 0]
        if len(valid_pixels) < 10:
            continue

        thresh_val = int(np.percentile(valid_pixels, 25))
        _, binary = cv2.threshold(masked_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.bitwise_and(binary, binary, mask=mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find largest contour and fit min-enclosing circle
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 5:
            continue

        (_, _), pupil_radius_scaled = cv2.minEnclosingCircle(largest)
        pupil_radius = pupil_radius_scaled / scale

        ratio = (pupil_radius * 2) / (iris_radius * 2)
        # Sanity: pupil/iris ratio should be 0.1-0.8
        if 0.1 <= ratio <= 0.8:
            results.append({
                'pupil_radius': pupil_radius,
                'iris_radius': iris_radius,
                'ratio': ratio,
                'eye': side,
            })

    if not results:
        return None

    avg_offset = np.mean(offsets, axis=0) if offsets else np.zeros(2)

    if len(results) == 2:
        avg_ratio = (results[0]['ratio'] + results[1]['ratio']) / 2
        # Determine which result is left vs right
        left_r = None
        right_r = None
        for r in results:
            if r['eye'] == 'left':
                left_r = r['ratio']
            elif r['eye'] == 'right':
                right_r = r['ratio']
        return {
            'pupil_radius': (results[0]['pupil_radius'] + results[1]['pupil_radius']) / 2,
            'iris_radius': (results[0]['iris_radius'] + results[1]['iris_radius']) / 2,
            'ratio': avg_ratio,
            'eye': 'avg',
            'iris_offset': avg_offset,
            'left_ratio': left_r,
            'right_ratio': right_r,
        }

    # Single eye result
    single = results[0]
    single['iris_offset'] = avg_offset
    single['left_ratio'] = single['ratio'] if single['eye'] == 'left' else None
    single['right_ratio'] = single['ratio'] if single['eye'] == 'right' else None
    return single


def measure_ir(eye_crop: np.ndarray, *, threshold: int = 40) -> dict | None:
    """
    Measure pupil/iris ratio from an IR (infrared) eye camera crop.

    Uses dark-pupil technique: pupil is the darkest circular region in IR.
    Uses Otsu's method for adaptive thresholding with fallback to the
    user-provided threshold.

    Parameters
    ----------
    eye_crop  : Grayscale or BGR numpy array from IR eye camera.
    threshold : Fallback pixel intensity threshold (default 40).

    Returns
    -------
    dict with keys: pupil_radius, iris_radius, ratio, eye ('ir'),
    or ``None`` if measurement failed.
    """
    if eye_crop is None or eye_crop.size == 0:
        return None

    if len(eye_crop.shape) == 3:
        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_crop.copy()

    # Blur to reduce noise (larger kernel for better noise rejection)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Pupil: adaptive thresholding via Otsu ---
    otsu_thresh, pupil_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Fallback if Otsu gives unreasonable result
    if otsu_thresh > 120 or otsu_thresh < 10:
        _, pupil_binary = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pupil_binary = cv2.morphologyEx(pupil_binary, cv2.MORPH_OPEN, kernel)
    pupil_binary = cv2.morphologyEx(pupil_binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(pupil_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find most circular large contour
    best_pupil = None
    best_circularity = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > best_circularity:
            best_circularity = circularity
            best_pupil = cnt

    if best_pupil is None or best_circularity < 0.4:
        return None

    (px, py), pupil_radius = cv2.minEnclosingCircle(best_pupil)
    if pupil_radius < 2:
        return None

    # --- Iris: edge detection for boundary ---
    edges = cv2.Canny(gray, 30, 80)

    # Mask out the pupil region to find iris edge beyond it
    pupil_mask = np.zeros_like(gray)
    cv2.circle(pupil_mask, (int(px), int(py)), int(pupil_radius * 1.3), 255, -1)
    edges_outer = cv2.bitwise_and(edges, edges,
                                  mask=cv2.bitwise_not(pupil_mask))

    # Dilate edges and find circles via HoughCircles
    edges_outer = cv2.dilate(edges_outer, kernel, iterations=1)

    h, w = gray.shape[:2]
    min_r = int(pupil_radius * 1.5)
    max_r = min(h, w) // 2

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max_r,
        param1=80, param2=30, minRadius=min_r, maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick circle closest to pupil center
        dists = np.sqrt((circles[:, 0] - px)**2 + (circles[:, 1] - py)**2)
        best_idx = np.argmin(dists)
        iris_radius = float(circles[best_idx, 2])
    else:
        # Fallback: estimate iris as ~2.5x pupil radius
        iris_radius = pupil_radius * 2.5

    if iris_radius < pupil_radius:
        return None

    ratio = (pupil_radius * 2) / (iris_radius * 2)
    if 0.1 <= ratio <= 0.8:
        return {
            'pupil_radius': float(pupil_radius),
            'iris_radius': float(iris_radius),
            'ratio': float(ratio),
            'eye': 'ir',
        }
    return None
