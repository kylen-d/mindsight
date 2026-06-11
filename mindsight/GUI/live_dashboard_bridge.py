"""
GUI/live_dashboard_bridge.py — Data transport from pipeline to live Qt dashboard.

This is an internal PhenomenaPlugin that is only activated in GUI mode.
It extracts dashboard-relevant metrics from the per-frame pipeline context and
pushes structured snapshots to a queue that the Qt dashboard widget drains.

Implemented as a PhenomenaPlugin (rather than DataCollectionPlugin) because
the run loop calls ``update()`` on all trackers with the full frame context,
making it the simplest integration point.

Not placed in the Plugins/ directory because it is not user-facing.
"""

from __future__ import annotations

import queue

from Plugins import PhenomenaPlugin


class LiveDashboardBridge(PhenomenaPlugin):
    """Bridges the pipeline worker thread to the Qt live dashboard.

    Constructed with a ``queue.Queue`` that the dashboard panel polls.
    Each frame, a lightweight snapshot dict is pushed containing both
    aggregate scalars and per-series rich metrics.
    """

    name = "_live_dashboard_bridge"
    dashboard_panel = "right"

    def __init__(self, dashboard_q: queue.Queue, throttle: int = 0):
        self._q = dashboard_q
        self._throttle = throttle   # skip update if frame_no % throttle != 0

    def update(self, **kwargs) -> dict:
        # Throttle: skip building the snapshot on most frames when enabled
        if self._throttle > 1:
            fn = kwargs.get('frame_no', 0)
            if fn % self._throttle != 0:
                return {}

        # Build per-tracker metrics for the live dashboard.
        # Prefer latest_metrics() (per-series) over latest_metric() (scalar).
        tracker_rich_metrics = {}  # name -> dict of series
        tracker_colours = {}      # name -> BGR colour tuple
        tracker_instances = {}    # name -> tracker (for custom widget discovery)
        for t in kwargs.get('_all_trackers', []):
            if t is self:
                continue
            name = getattr(t, 'name', None)
            if not name or name.startswith('_'):
                continue

            colour = getattr(t, '_COLOUR', None)
            if colour:
                tracker_colours[name] = colour

            tracker_instances[name] = t

            # Try rich metrics first, fall back to scalar
            if hasattr(t, 'latest_metrics'):
                rich = t.latest_metrics()
                if rich is not None:
                    tracker_rich_metrics[name] = rich
                    continue
            if hasattr(t, 'latest_metric'):
                m = t.latest_metric()
                if m is not None:
                    tracker_rich_metrics[name] = {
                        '_aggregate': {
                            'value': m,
                            'label': name.replace('_', ' ').title(),
                            'y_label': '',
                        }
                    }

        snapshot = {
            'frame_no': kwargs.get('frame_no', 0),
            'fps': kwargs.get('fps', 0.0),
            'n_faces': kwargs.get('n_faces', 0),
            'joint_pct': kwargs.get('joint_pct', 0.0),
            'n_dets': kwargs.get('n_dets', 0),
            'tracker_rich_metrics': tracker_rich_metrics,
            'tracker_colours': tracker_colours,
            'tracker_instances': tracker_instances,
        }
        try:
            self._q.put_nowait(snapshot)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(snapshot)
            except queue.Full:
                pass
        return {}

    def dashboard_data(self, *, pid_map=None) -> dict:
        return {}
