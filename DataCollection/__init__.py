"""
DataCollection — Output and data-recording pipeline stage.

Handles frame overlay rendering (dashboard annotations), per-frame CSV
event logging, post-run summary CSV generation, and per-participant
gaze heatmap output.
"""

from .csv_output import write_summary_csv
from .dashboard_output import compose_dashboard, draw_overlay, finalize_video, open_video_writer
from .data_pipeline import collect_frame_data, finalize_run
from .global_csv import generate_condition_csvs, generate_global_csv
from .heatmap_output import save_heatmaps

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
