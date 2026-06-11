"""
mindsight.io -- Input/output boundary for the pipeline.

- ``sources``: primary video/image/webcam capture + auxiliary-stream handling.
- ``writers``: annotated-video writer + per-frame event-CSV handles.

These modules isolate all file/device IO from the pipeline logic so the
GUI-consumable ``Pipeline`` API can own its own IO lifecycle.
"""
