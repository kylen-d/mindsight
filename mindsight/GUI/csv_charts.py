"""
csv_charts.py -- display-only phenomena charts from a run's written CSVs.

Shared by the Analyze Footage Charts tab and the Projects overview data pane
(eyes-on 2026-07-11): both render the SAME figure from a
:class:`~mindsight.GUI.run_outputs.RunOutputs`, so the charts a user sees are
identical wherever they look.  Nothing here writes into the project's
Outputs/ tree (D11).
"""

from __future__ import annotations

_DIM = "#cccccc"
_PANEL = "#1e1e1e"
_EDGE = "#2a2a2a"


def draw_look_time(ax, table: dict):
    """Grouped per-participant bars of % look time per object."""
    objects = sorted({o for objs in table.values() for o in objs})
    participants = sorted(table)
    width = 0.8 / max(1, len(participants))
    for pi, who in enumerate(participants):
        vals = [table[who].get(o, 0.0) for o in objects]
        xs = [i + pi * width for i in range(len(objects))]
        ax.bar(xs, vals, width=width, label=who)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(objects))])
    ax.set_xticklabels(objects, rotation=30, ha="right",
                       color=_DIM, fontsize=7)
    ax.set_ylabel("% of video", color=_DIM, fontsize=8)
    ax.set_title("Object look time", color=_DIM, fontsize=9, loc="left")


def draw_timeline(ax, objects: list, per: dict):
    """Per-participant scatter of which object is gazed at over time."""
    for who in sorted(per):
        xs, ys = per[who]
        ax.scatter(xs, ys, s=4, label=who)
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects, color=_DIM, fontsize=7)
    ax.set_xlabel("t (seconds)", color=_DIM, fontsize=8)
    ax.set_title("Gaze target timeline", color=_DIM, fontsize=9, loc="left")


def build_charts_figure(out, *, panel_height: float = 3.2):
    """A Figure with every renderable panel for *out* (RunOutputs), or None
    when its CSVs yield nothing chartable."""
    from matplotlib.figure import Figure

    from .run_outputs import gaze_timeline, look_time_table

    panels = []
    if out is not None and out.summary is not None:
        try:
            table = look_time_table(out.summary)
            if table:
                panels.append(("look", table))
        except Exception:
            pass
    if out is not None and out.events is not None:
        try:
            objects, per = gaze_timeline(out.events)
            if objects and per:
                panels.append(("timeline", (objects, per)))
        except Exception:
            pass
    if not panels:
        return None

    fig = Figure(figsize=(6, panel_height * len(panels)), facecolor="#121212")
    for i, (kind, data) in enumerate(panels):
        ax = fig.add_subplot(len(panels), 1, i + 1)
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_EDGE)
        if kind == "look":
            draw_look_time(ax, data)
        else:
            draw_timeline(ax, *data)
        ax.legend(fontsize=7, facecolor=_PANEL, labelcolor=_DIM,
                  edgecolor=_EDGE)
    fig.tight_layout()
    return fig
