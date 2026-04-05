"""
GUI/pipeline_dialog.py — Pipeline YAML import and export for the MindSight GUI.

Provides two functions:
  - import_pipeline(parent) -> Namespace | None
  - export_pipeline(parent, ns)
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QFileDialog, QMessageBox


def import_pipeline(parent) -> Namespace | None:
    """Show a file dialog to select a YAML pipeline file, load it, and return a Namespace.

    Returns None if the user cancels or an error occurs.
    """
    path, _ = QFileDialog.getOpenFileName(
        parent, "Import Pipeline Configuration", "",
        "YAML (*.yaml *.yml);;All (*)")
    if not path:
        return None

    try:
        from pipeline_loader import load_pipeline
        ns = load_pipeline(path, Namespace())
        QMessageBox.information(
            parent, "Pipeline Imported",
            f"Loaded pipeline configuration from:\n{Path(path).name}")
        return ns
    except Exception as e:
        QMessageBox.critical(
            parent, "Import Error",
            f"Failed to import pipeline:\n{e}")
        return None


def export_pipeline(parent, ns: Namespace) -> bool:
    """Show a file dialog and export the current settings as a YAML pipeline file.

    Returns True if the file was saved successfully.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent, "Export Pipeline Configuration", "",
        "YAML (*.yaml *.yml);;All (*)")
    if not path:
        return False
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"

    try:
        cfg = _namespace_to_yaml_dict(ns)
        Path(path).write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        QMessageBox.information(
            parent, "Pipeline Exported",
            f"Saved pipeline configuration to:\n{Path(path).name}")
        return True
    except Exception as e:
        QMessageBox.critical(
            parent, "Export Error",
            f"Failed to export pipeline:\n{e}")
        return False


def _namespace_to_yaml_dict(ns: Namespace) -> dict:
    """Convert a Namespace to a structured YAML dict matching the pipeline format.

    Reverses the flattening done by pipeline_loader to produce a readable,
    structured YAML file with detection, gaze, output, and phenomena sections.
    """
    d = vars(ns) if hasattr(ns, '__dict__') else {}

    cfg = {}

    # Source
    source = d.get("source", "0")
    if source and source != "0":
        cfg["source"] = source

    # Detection section
    detection = {}
    if d.get("model") and d.get("model") != "yolov8n.pt":
        detection["model"] = d["model"]
    if d.get("conf") and d["conf"] != 0.35:
        detection["conf"] = d["conf"]
    if d.get("classes"):
        detection["classes"] = d["classes"]
    if d.get("blacklist"):
        detection["blacklist"] = list(d["blacklist"]) if isinstance(d["blacklist"], set) else d["blacklist"]
    if d.get("detect_scale") and d["detect_scale"] != 1.0:
        detection["detect_scale"] = d["detect_scale"]
    if d.get("vp_file"):
        detection["vp_file"] = d["vp_file"]
    if d.get("vp_model") and d["vp_model"] != "yoloe-26l-seg.pt":
        detection["vp_model"] = d["vp_model"]
    if d.get("skip_frames") and d["skip_frames"] != 1:
        detection["skip_frames"] = d["skip_frames"]
    if d.get("obj_persistence") and d["obj_persistence"] != 0:
        detection["obj_persistence"] = d["obj_persistence"]
    if detection:
        cfg["detection"] = detection

    # Gaze section
    gaze = {}
    if d.get("ray_length") and d["ray_length"] != 1.0:
        gaze["ray_length"] = d["ray_length"]
    ar = d.get("adaptive_ray", "off")
    if ar and ar != "off":
        gaze["adaptive_ray"] = ar
    if d.get("snap_dist") and d["snap_dist"] != 150.0:
        gaze["snap_dist"] = d["snap_dist"]
    if d.get("snap_bbox_scale") is not None and d["snap_bbox_scale"] != 0.0:
        gaze["snap_bbox_scale"] = d["snap_bbox_scale"]
    if d.get("snap_w_dist") is not None and d["snap_w_dist"] != 1.0:
        gaze["snap_w_dist"] = d["snap_w_dist"]
    if d.get("snap_w_size") is not None and d["snap_w_size"] != 0.0:
        gaze["snap_w_size"] = d["snap_w_size"]
    if d.get("snap_w_intersect") is not None and d["snap_w_intersect"] != 0.5:
        gaze["snap_w_intersect"] = d["snap_w_intersect"]
    if d.get("conf_ray"):
        gaze["conf_ray"] = True
    if d.get("gaze_tips"):
        gaze["gaze_tips"] = True
    if d.get("tip_radius") and d["tip_radius"] != 80:
        gaze["tip_radius"] = d["tip_radius"]
    if d.get("gaze_cone") and d["gaze_cone"] != 0.0:
        gaze["gaze_cone"] = d["gaze_cone"]
    if d.get("gaze_lock"):
        gaze["gaze_lock"] = True
    if d.get("dwell_frames") and d["dwell_frames"] != 15:
        gaze["dwell_frames"] = d["dwell_frames"]
    if d.get("lock_dist") and d["lock_dist"] != 100:
        gaze["lock_dist"] = d["lock_dist"]
    if d.get("gaze_debug"):
        gaze["gaze_debug"] = True
    if d.get("snap_switch_frames") and d["snap_switch_frames"] != 8:
        gaze["snap_switch_frames"] = d["snap_switch_frames"]
    if d.get("reid_grace_seconds") and d["reid_grace_seconds"] != 1.0:
        gaze["reid_grace_seconds"] = d["reid_grace_seconds"]
    if gaze:
        cfg["gaze"] = gaze

    # Output section
    output = {}
    if d.get("save"):
        output["save_video"] = d["save"] if isinstance(d["save"], str) else True
    if d.get("log"):
        output["log_csv"] = d["log"]
    if d.get("summary"):
        output["summary_csv"] = d["summary"] if isinstance(d["summary"], str) else True
    if d.get("heatmap"):
        output["heatmaps"] = d["heatmap"] if isinstance(d["heatmap"], str) else True
    if d.get("anonymize"):
        output["anonymize"] = d["anonymize"]
        padding = d.get("anonymize_padding", 0.3)
        if padding != 0.3:
            output["anonymize_padding"] = padding
    if output:
        cfg["output"] = output

    # Phenomena section
    phenomena = []
    # Simple toggles
    _toggle_map = {
        "joint_attention": "joint_attention",
        "mutual_gaze": "mutual_gaze",
        "social_ref": "social_referencing",
        "gaze_follow": "gaze_following",
        "gaze_aversion": "gaze_aversion",
        "scanpath": "scanpath",
        "gaze_leader": "gaze_leadership",
        "attn_span": "attention_span",
    }
    _param_map = {
        "joint_attention": {"ja_window": "ja_window", "ja_quorum": "ja_quorum",
                           "ja_window_thresh": "ja_window_thresh"},
        "social_referencing": {"social_ref_window": "window"},
        "gaze_following": {"gaze_follow_lag": "lag"},
        "gaze_aversion": {"aversion_window": "aversion_window", "aversion_conf": "aversion_conf"},
        "scanpath": {"scanpath_dwell": "dwell"},
    }
    for attr, yaml_name in _toggle_map.items():
        if d.get(attr):
            params = {}
            for ns_key, yaml_key in _param_map.get(yaml_name, {}).items():
                val = d.get(ns_key)
                if val is not None:
                    params[yaml_key] = val
            if params:
                phenomena.append({yaml_name: params})
            else:
                phenomena.append(yaml_name)

    if phenomena:
        cfg["phenomena"] = phenomena

    # Performance section
    performance = {}
    if d.get("fast"):
        performance["fast"] = True
    if d.get("skip_phenomena") and d["skip_phenomena"] > 0:
        performance["skip_phenomena"] = d["skip_phenomena"]
    if d.get("lite_overlay"):
        performance["lite_overlay"] = True
    if d.get("no_dashboard"):
        performance["no_dashboard"] = True
    if d.get("profile"):
        performance["profile"] = True
    if performance:
        cfg["performance"] = performance

    return cfg
