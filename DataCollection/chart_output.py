"""
DataCollection/chart_output.py — Post-run chart generation for phenomena trackers.

Generates matplotlib time-series charts after a run completes, saved alongside
CSV summary and heatmap outputs.  Each tracker that implements ``time_series_data()``
contributes one subplot.  An optional Joint Attention time-series can be included.

The dark theme matches the dashboard side-panel aesthetic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Theme colours (matching dashboard_matplotlib.py)
_BG = '#121212'
_CARD_BG = '#1e1e1e'
_TEXT = '#cccccc'
_GRID = '#2a2a2a'
_ACCENT_JA = '#00c8ff'


def _bgr_to_rgb(bgr: tuple) -> tuple:
    """Convert a BGR 0–255 colour to an RGB 0–1 tuple."""
    return (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)


def resolve_chart_path(charts_arg, source) -> Path | None:
    """Resolve the output path for the combined chart image.

    Parameters
    ----------
    charts_arg : True | str | None | False
        True  → auto-name under Outputs/Charts/
        str   → use as explicit path
        None/False → no charts
    source : str | int
        Video file path or webcam index.
    """
    if not charts_arg:
        return None
    from constants import OUTPUTS_ROOT
    if charts_arg is True:
        stem = Path(str(source)).stem if not isinstance(source, int) else 'webcam'
        return OUTPUTS_ROOT / 'Charts' / f'{stem}_Charts.png'
    return Path(charts_arg)


def generate_run_charts(
    output_path: str | Path,
    all_trackers: list,
    total_frames: int,
    fps: float,
    pid_map: dict | None = None,
    data_plugins: list | None = None,
) -> list[str]:
    """Generate post-run time-series charts for all trackers with data.

    Parameters
    ----------
    output_path : Path to save the combined chart PNG.
    all_trackers : List of PhenomenaPlugin instances.
    total_frames : Total frame count of the run.
    fps : Frames per second (for time axis).
    pid_map : Participant ID map.
    data_plugins : Optional list of DataCollectionPlugin instances.

    Returns
    -------
    List of file paths that were created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []

    # Collect all series from trackers (including JA, which is now a tracker)
    all_series = []

    for tracker in (all_trackers or []):
        ts = tracker.time_series_data()
        if ts:
            title = getattr(tracker, 'name', '?').upper().replace('_', ' ')
            all_series.append({
                'title': title,
                'series': ts,
            })

    if not all_series:
        return saved

    n_panels = len(all_series)
    fig_h = max(4, 2.5 * n_panels)
    fig = Figure(figsize=(12, fig_h), dpi=150, facecolor=_BG)

    for idx, entry in enumerate(all_series):
        ax = fig.add_subplot(n_panels, 1, idx + 1)
        ax.set_facecolor(_CARD_BG)
        ax.set_title(entry['title'], color=_TEXT, fontsize=10, fontweight='bold',
                     fontfamily='sans-serif', loc='left', pad=8)

        for series_name, s in entry['series'].items():
            x = np.array(s['x'])
            y = np.array(s['y'])
            color = _bgr_to_rgb(s.get('color', (180, 180, 180)))
            label = s.get('label', series_name)
            chart_type = s.get('chart_type', 'line')

            if chart_type == 'area':
                ax.fill_between(x, y, alpha=0.3, color=color)
                ax.plot(x, y, color=color, linewidth=1.2, label=label)
            elif chart_type == 'step':
                ax.step(x, y, where='post', color=color, linewidth=1.2,
                        label=label)
            else:  # line
                ax.plot(x, y, color=color, linewidth=1.2, label=label)

        # Styling
        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.set_xlim(0, max(total_frames, 1))
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.5,
                  facecolor=_CARD_BG, edgecolor=_GRID, labelcolor=_TEXT)

        # Frame number on x-axis (bottom subplot only gets label)
        if idx == n_panels - 1:
            ax.set_xlabel('Frame', color=_TEXT, fontsize=8)
            # Secondary time axis
            if fps > 0:
                ax2 = ax.secondary_xaxis('top',
                    functions=(lambda f: f / fps, lambda t: t * fps))
                ax2.set_xlabel('Time (s)', color=_TEXT, fontsize=7)
                ax2.tick_params(colors=_TEXT, labelsize=6)
        else:
            ax.tick_params(labelbottom=False)

    fig.tight_layout(pad=1.5)
    fig.savefig(str(output_path), facecolor=_BG)
    plt.close(fig)
    saved.append(str(output_path))
    print(f'Charts saved \u2192 {output_path}')

    # Individual per-tracker chart PNGs
    chart_dir = output_path.parent
    for entry in all_series:
        slug = entry['title'].lower().replace(' ', '_')
        ind_path = chart_dir / f'{output_path.stem}_{slug}.png'
        ind_fig = Figure(figsize=(8, 3), dpi=150, facecolor=_BG)
        ax = ind_fig.add_subplot(1, 1, 1)
        ax.set_facecolor(_CARD_BG)
        ax.set_title(entry['title'], color=_TEXT, fontsize=10,
                     fontweight='bold', fontfamily='sans-serif', loc='left')

        for series_name, s in entry['series'].items():
            x = np.array(s['x'])
            y = np.array(s['y'])
            color = _bgr_to_rgb(s.get('color', (180, 180, 180)))
            label = s.get('label', series_name)
            chart_type = s.get('chart_type', 'line')
            if chart_type == 'area':
                ax.fill_between(x, y, alpha=0.3, color=color)
                ax.plot(x, y, color=color, linewidth=1.2, label=label)
            elif chart_type == 'step':
                ax.step(x, y, where='post', color=color, linewidth=1.2,
                        label=label)
            else:
                ax.plot(x, y, color=color, linewidth=1.2, label=label)

        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.set_xlim(0, max(total_frames, 1))
        ax.set_xlabel('Frame', color=_TEXT, fontsize=8)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.5,
                  facecolor=_CARD_BG, edgecolor=_GRID, labelcolor=_TEXT)
        ind_fig.tight_layout(pad=1.0)
        ind_fig.savefig(str(ind_path), facecolor=_BG)
        plt.close(ind_fig)
        saved.append(str(ind_path))

    # Let DataCollectionPlugins contribute their own charts
    if data_plugins:
        for plugin in data_plugins:
            if hasattr(plugin, 'generate_charts'):
                try:
                    paths = plugin.generate_charts(
                        str(chart_dir),
                        total_frames=total_frames,
                        fps=fps,
                        all_trackers=all_trackers,
                        pid_map=pid_map,
                    )
                    saved.extend(paths or [])
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Chart generation failed for plugin "
                        f"'{getattr(plugin, 'name', '?')}': {exc}",
                        RuntimeWarning,
                    )

    return saved
