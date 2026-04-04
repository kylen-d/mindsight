# Developer Guide

This guide is for contributors and plugin authors who want to understand how MindSight works internally or extend it with new functionality. It covers the core architecture, each processing module, the plugin system, and the test suite.

---

<div class="grid cards" markdown>

-   **[Architecture Deep Dive](architecture.md)**

    High-level data flow, threading model, and module boundaries.

-   **[FrameContext Reference](frame-context.md)**

    The per-frame data structure passed through every pipeline stage.

-   **[Object Detection Module](object-detection-module.md)**

    How detections are produced, filtered, and attached to FrameContext.

-   **[Gaze Processing Module](gaze-processing-module.md)**

    Face crop extraction, backend dispatch, ray geometry, and gaze target resolution.

-   **[Phenomena Engine](phenomena-engine.md)**

    Tracker lifecycle, the `update()` contract, and built-in detector internals.

-   **[Data Collection Module](data-collection-module.md)**

    CSV writer, heatmap generator, video overlay renderer, and dashboard hooks.

-   **[Plugin System](plugin-system.md)**

    Discovery, registration, base classes, and the argument injection mechanism.

-   **[Writing a Plugin](writing-a-plugin.md)**

    Step-by-step instructions for creating a plugin from scratch.

-   **[Plugin Tutorials](plugin-tutorial.md)**

    Worked examples for all four plugin types: [Phenomena](phenomena-plugin-tutorial.md), [Gaze](gaze-plugin-tutorial.md), [Object Detection](object-detection-plugin-tutorial.md), [Data Collection](data-collection-plugin-tutorial.md).

-   **[Pipeline Configuration (YAML)](pipeline-yaml.md)**

    Declarative pipeline setup, key mapping, and precedence rules.

-   **[Testing](testing.md)**

    Running the test suite and writing tests for new code and plugins.

</div>
