"""
Plugins/TEMPLATE/my_plugin.py — Skeleton phenomena tracker plugin.

How to create a new phenomena plugin
-------------------------------------
1. Copy the entire TEMPLATE/ folder to Plugins/Phenomena/YourPluginName/.
2. Rename this file (or keep it — only the class and PLUGIN_CLASS matter).
3. Fill in the class below with your tracking logic.
4. Set a unique ``name`` class attribute.
5. Implement ``add_arguments`` / ``from_args`` for CLI activation.
6. Ensure ``PLUGIN_CLASS = YourTracker`` is at the bottom of the file.

The plugin will be auto-discovered by the registry on startup.

Activation
----------
    python MindSight.py --your-flag [options ...]

Available kwargs in update()
----------------------------
Every frame, your ``update(**kwargs)`` receives:

    frame_no       : int — current frame index
    persons_gaze   : list of (origin, ray_end, angles) per face
    face_bboxes    : list of (x1, y1, x2, y2) per face
    hit_events     : list[dict] with face_idx, object, object_conf, bbox
    joint_objs     : set of object indices with joint attention
    dets           : list[dict] non-person YOLO detections
    n_faces        : int — number of visible faces
    face_track_ids : list[int] stable re-ID track IDs
    hits           : set of (face_idx, obj_idx) intersection pairs
    aux_frames     : dict[(pid, stream_type), ndarray | None] — auxiliary
                     per-participant video frames (e.g. eye cameras).
                     Empty dict when no auxiliary streams are configured.

Pull only what you need — your plugin doesn't break if new keys are added.

Dashboard drawing helpers
-------------------------
Import from ``DataCollection.dashboard_output``:
    _draw_panel_section(panel, y, title, title_col, rows, line_h) -> int
    _DASH_DIM  (colour for placeholder text)
"""

from __future__ import annotations

from Plugins import PhenomenaPlugin


class MyTracker(PhenomenaPlugin):
    """One-line description of your tracker."""

    name = "my_tracker"          # unique identifier (used in CSV, dashboard)
    dashboard_panel = "right"    # "left" or "right" side panel

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        # Add your state variables here

    # ── Per-frame update (required) ──────────────────────────────────────────

    def update(self, **kwargs) -> dict:
        """Process one frame.  Pull only the kwargs you need."""
        frame_no = kwargs['frame_no']  # noqa: F841
        persons_gaze = kwargs.get('persons_gaze', [])  # noqa: F841
        hits = kwargs.get('hits', set())  # noqa: F841

        # Access auxiliary video streams (if configured):
        # aux = kwargs.get('aux_frames', {})
        # eye_frame = aux.get(('S70', 'eye_camera'))  # ndarray or None

        # ... your tracking logic here ...

        return {}  # return plugin-specific live state (may be empty)

    # ── Frame overlay (optional) ─────────────────────────────────────────────

    def draw_frame(self, frame) -> None:
        """Draw annotations on the video frame (in-place).  Optional."""
        pass

    # ── Dashboard section (optional) ─────────────────────────────────────────

    def dashboard_section(self, panel, y: int, line_h: int) -> int:
        """Draw a section in the side panel.  Return new y after the section."""
        from ms.DataCollection.dashboard_output import _DASH_DIM, _draw_panel_section
        rows = [("--", _DASH_DIM)]
        return _draw_panel_section(
            panel, y, "MY TRACKER", (200, 200, 200), rows, line_h)

    # ── CSV output (optional) ────────────────────────────────────────────────

    def csv_rows(self, total_frames: int) -> list:
        """Return rows to append to the summary CSV.  Optional."""
        return []

    # ── Console summary (optional) ───────────────────────────────────────────

    def console_summary(self, total_frames: int) -> str | None:
        """Return a string for post-run stdout output.  Optional."""
        return None

    # ── CLI protocol (required for activation) ───────────────────────────────

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Add CLI flags for this plugin."""
        g = parser.add_argument_group("My Tracker plugin")
        g.add_argument("--my-tracker", action="store_true",
                       help="Enable the my-tracker plugin.")
        g.add_argument("--my-threshold", type=float, default=0.5,
                       help="Example threshold parameter (default: 0.5).")

    @classmethod
    def from_args(cls, args):
        """Return an instance if activated, else None."""
        if not getattr(args, "my_tracker", False):
            return None
        return cls(threshold=getattr(args, "my_threshold", 0.5))


# ── Exported symbol consumed by PluginRegistry.discover() ────────────────────
PLUGIN_CLASS = MyTracker
