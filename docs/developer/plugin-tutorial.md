# Plugin Tutorials

Worked examples for each of MindSight's four plugin types. Each tutorial walks through a real or realistic plugin implementation, explaining design decisions and best practices.

---

## By Plugin Type

<div class="grid cards" markdown>

-   **[Phenomena Plugin: Novel Salience](phenomena-plugin-tutorial.md)**

    Build a custom gaze-phenomena tracker. Walks through the real NovelSalience saccade detector that ships with MindSight.

-   **[Gaze Plugin: Building a Backend](gaze-plugin-tutorial.md)**

    Build a custom gaze estimation backend. Walks through the real Gazelle (DINOv2 scene-level) plugin.

-   **[Object Detection Plugin: Post-Processing](object-detection-plugin-tutorial.md)**

    Build a detection post-processor that augments or filters YOLO detections.

-   **[Data Collection Plugin: Custom Output](data-collection-plugin-tutorial.md)**

    Build a custom data output plugin for per-frame collection and post-run reporting.

</div>

---

## Before You Start

1. Read the [Plugin System](plugin-system.md) page to understand discovery, registration, and base classes.
2. Read [Writing a Plugin](writing-a-plugin.md) for the step-by-step template walkthrough covering all four plugin types.
3. Choose the tutorial that matches your use case above.
