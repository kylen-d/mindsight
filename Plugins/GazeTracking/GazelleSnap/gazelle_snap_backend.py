"""
Plugins/GazeTracking/GazelleSnap/gazelle_snap_backend.py — DEPRECATED.

**This plugin is deprecated.**  Gazelle blend is now a core feature of the
RayForming module (``ms.PostProcessing.RayForming``).  Use the core flags
instead:

    --gazelle-model PATH    (replaces --gs-gazelle-model)
    --gazelle-name NAME     (replaces --gs-gazelle-name)
    --gazelle-interval N    (replaces --gs-snap-interval)
    --blend-strength F      (replaces --gs-heatmap-weight)

The pitch/yaw backend is selected as normal (e.g. ``--l2cs-model``).  No
``--gazelle-snap`` flag is needed -- the system auto-detects Gazelle blend
when ``--gazelle-model`` is provided alongside a pitch/yaw backend.

This file is retained for backward compatibility: existing CLI invocations
using ``--gazelle-snap`` will still work but print a deprecation warning.
The plugin now delegates estimation to the inner pitch/yaw backend and
lets the core pipeline handle Gazelle heatmap inference and belief blending.
"""
from __future__ import annotations

import warnings

from Plugins import GazePlugin


class GazelleSnapPlugin(GazePlugin):
    """Deprecated composite backend -- delegates to core RayForming.

    When ``--gazelle-snap`` is used, this plugin activates but only handles
    pitch/yaw estimation per face.  Gazelle model loading and heatmap
    inference are handled by the core ``GazelleProvider`` in
    ``ms.PostProcessing.RayForming``.
    """

    name = "gazelle_snap"
    mode = "per_face"
    is_fallback = False

    def __init__(self, pitchyaw_engine):
        self._py_engine = pitchyaw_engine

    def estimate(self, face_bgr):
        return self._py_engine.estimate(face_bgr)

    # NOTE: No run_pipeline() -- the core gaze_pipeline.py will use
    # _estimate_pitchyaw + RayForming (Path A) for this per-face backend.

    # ── CLI protocol ─────────────────────────────────────────────────────────

    @classmethod
    def add_arguments(cls, parser):
        g = parser.add_argument_group("Gazelle-Snap (DEPRECATED -- use core flags)")
        g.add_argument(
            "--gazelle-snap", action="store_true", default=False,
            help="[DEPRECATED] Enable Gazelle blend. Use --gazelle-model "
                 "instead. This flag is retained for backward compatibility.",
        )
        g.add_argument(
            "--gs-gazelle-model", default=None, metavar="PATH",
            help="[DEPRECATED] Use --gazelle-model instead.",
        )
        g.add_argument(
            "--gs-gazelle-name",
            default="gazelle_dinov2_vitb14",
            choices=sorted([
                "gazelle_dinov2_vitb14", "gazelle_dinov2_vitl14",
                "gazelle_dinov2_vitb14_inout", "gazelle_dinov2_vitl14_inout",
            ]),
            metavar="NAME",
            help="[DEPRECATED] Use --gazelle-name instead.",
        )
        g.add_argument("--gs-snap-interval", type=int, default=30, metavar="N",
                        help="[DEPRECATED] Use --gazelle-interval instead.")
        g.add_argument("--gs-heatmap-threshold", type=float, default=0.5, metavar="F",
                        help="[DEPRECATED] Legacy heatmap threshold.")
        g.add_argument("--gs-heatmap-weight", type=float, default=1.0, metavar="F",
                        help="[DEPRECATED] Use --blend-strength instead.")
        g.add_argument("--gs-direction-decay", type=float, default=0.98, metavar="F",
                        help="[DEPRECATED] Replaced by belief map diffusion.")
        g.add_argument("--gs-length-decay", type=float, default=0.99, metavar="F",
                        help="[DEPRECATED] Replaced by belief map diffusion.")
        g.add_argument("--gs-direction-threshold", type=float, default=0.05, metavar="F",
                        help="[DEPRECATED]")
        g.add_argument("--gs-length-threshold", type=float, default=0.05, metavar="F",
                        help="[DEPRECATED]")
        g.add_argument("--gs-obj-snap", default="all",
                        choices=["all", "faces_only", "off"],
                        help="[DEPRECATED] Use --obj-snap-targets instead.")
        g.add_argument("--gs-smooth-snap", action="store_true", default=False,
                        help="[DEPRECATED] Use --smooth-snap instead.")
        g.add_argument("--gs-smooth-snap-alpha", type=float, default=0.20, metavar="F",
                        help="[DEPRECATED] Use --smooth-snap-alpha instead.")

    @classmethod
    def from_args(cls, args):
        if not getattr(args, "gazelle_snap", False):
            return None

        warnings.warn(
            "--gazelle-snap is deprecated. Use --gazelle-model PATH with a "
            "pitch/yaw backend instead. The core pipeline now handles Gazelle "
            "blend via the RayForming module.",
            DeprecationWarning, stacklevel=2,
        )

        # Forward --gs-* flags to --rf-* flags for the core GazelleProvider.
        # Uses rf_ prefix to avoid collision with standalone Gazelle plugin.
        gs_model = getattr(args, "gs_gazelle_model", None)
        if gs_model and not getattr(args, "rf_gazelle_model", None):
            args.rf_gazelle_model = gs_model
        gs_name = getattr(args, "gs_gazelle_name", None)
        if gs_name and not getattr(args, "rf_gazelle_name", None):
            args.rf_gazelle_name = gs_name
        gs_interval = getattr(args, "gs_snap_interval", None)
        if gs_interval and not getattr(args, "rf_gazelle_interval", None):
            args.rf_gazelle_interval = gs_interval

        # Create the inner pitch/yaw backend
        from ms.GazeTracking.gaze_factory import create_gaze_engine
        args.gazelle_snap = False
        try:
            pitchyaw_engine = create_gaze_engine(plugin_args=args)
        finally:
            args.gazelle_snap = True

        if getattr(pitchyaw_engine, "mode", None) != "per_face":
            raise ValueError(
                "--gazelle-snap requires a per-face pitch/yaw backend "
                f"(got mode={getattr(pitchyaw_engine, 'mode', '?')})"
            )
        print(f"Backend: {getattr(pitchyaw_engine, 'name', '?')} "
              f"(Gazelle blend via core RayForming)")

        return cls(pitchyaw_engine)


# ── Exported symbol consumed by PluginRegistry.discover() ────────────────────
PLUGIN_CLASS = GazelleSnapPlugin
