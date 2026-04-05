"""
GazeBoost — ObjectDetection plugin that boosts confidence of detections
near the previous frame's gaze ray endpoints.

Objects that someone is looking at are more likely to be real detections.
A coffee mug at 0.20 confidence sitting far from any gaze is probably a
false positive, but the same mug at 0.20 right where someone is looking
is worth keeping.  GazeBoost uses the previous frame's gaze endpoints as
evidence to push borderline detections above the confidence threshold.

Enable via ``--gaze-boost``.  See ``--help`` for tuning parameters.
"""

from Plugins import ObjectDetectionPlugin


class GazeBoostPlugin(ObjectDetectionPlugin):
    """Boost confidence of detections near previous-frame gaze endpoints."""

    name = "gaze_boost"

    def __init__(self, boost_factor: float = 1.5,
                 radius_px: float = 100.0,
                 min_conf: float = 0.10,
                 max_boosted_conf: float = 0.95,
                 classes: list | None = None):
        self._boost = boost_factor
        self._radius = radius_px
        self.min_conf = min_conf       # read by detection_pipeline for sub-threshold rescue
        self._max_conf = max_boosted_conf
        self._classes = {c.lower() for c in classes} if classes else None

    # ── Detection hook ──────────────────────────────────────────────

    def detect(self, *, frame, detection_frame, all_dets, det_cfg, **kwargs):
        gaze_endpoints = kwargs.get('prev_persons_gaze', [])
        if not gaze_endpoints:
            return None  # no previous gaze data (first frame or no faces)

        modified = False
        for det in all_dets:
            cls_name = det['class_name'].lower()
            if cls_name == 'person':
                continue
            if self._classes is not None and cls_name not in self._classes:
                continue
            if self._near_gaze(det, gaze_endpoints):
                new_conf = min(self._max_conf, det['conf'] * self._boost)
                if new_conf != det['conf']:
                    det['conf'] = new_conf
                    modified = True
        return all_dets if modified else None

    def _near_gaze(self, det, gaze_endpoints) -> bool:
        """Return True if any gaze endpoint is within radius of the det bbox."""
        # Expand the detection bbox by the radius and check if gaze falls inside
        x1 = det['x1'] - self._radius
        y1 = det['y1'] - self._radius
        x2 = det['x2'] + self._radius
        y2 = det['y2'] + self._radius
        for _, ray_end, _ in gaze_endpoints:
            gx, gy = float(ray_end[0]), float(ray_end[1])
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                return True
        return False

    # ── CLI integration ─────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("Gaze Boost plugin")
        g.add_argument("--gaze-boost", action="store_true",
                       help="Boost confidence of detections near gaze endpoints.")
        g.add_argument("--gaze-boost-factor", type=float, default=1.5,
                       help="Multiplicative confidence boost (default: 1.5).")
        g.add_argument("--gaze-boost-radius", type=float, default=100.0,
                       help="Pixel radius around gaze endpoints (default: 100).")
        g.add_argument("--gaze-boost-min-conf", type=float, default=0.10,
                       help="Minimum YOLO conf for sub-threshold candidates (default: 0.10).")
        g.add_argument("--gaze-boost-max-conf", type=float, default=0.95,
                       help="Cap on boosted confidence (default: 0.95).")
        g.add_argument("--gaze-boost-classes", nargs="+", default=None,
                       help="Only boost these class names (default: all non-person classes).")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "gaze_boost", False):
            return None
        return cls(
            boost_factor=getattr(args, "gaze_boost_factor", 1.5),
            radius_px=getattr(args, "gaze_boost_radius", 100.0),
            min_conf=getattr(args, "gaze_boost_min_conf", 0.10),
            max_boosted_conf=getattr(args, "gaze_boost_max_conf", 0.95),
            classes=getattr(args, "gaze_boost_classes", None),
        )


PLUGIN_CLASS = GazeBoostPlugin
