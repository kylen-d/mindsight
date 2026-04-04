"""
DataCollection — Output and data-recording pipeline stage.

Handles frame overlay rendering (dashboard annotations), per-frame CSV
event logging, post-run summary CSV generation, and per-participant
gaze heatmap output.
"""

from .data_pipeline import collect_frame_data, finalize_run
from .dashboard_output import draw_overlay, compose_dashboard, open_video_writer, finalize_video
from .csv_output import write_summary_csv
from .heatmap_output import save_heatmaps
from .global_csv import generate_global_csv, generate_condition_csvs

__all__ = [
    "collect_frame_data",
    "finalize_run",
    "draw_overlay",
    "compose_dashboard",
    "open_video_writer",
    "finalize_video",
    "write_summary_csv",
    "save_heatmaps",
    "generate_global_csv",
    "generate_condition_csvs",
]
