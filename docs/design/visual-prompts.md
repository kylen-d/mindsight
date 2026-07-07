# Visual Prompts

Most studies care about objects that are not ordinary COCO classes -- a
particular toy, a specific piece of apparatus, a bespoke stimulus. Rather than
collecting a training set and retraining a detector, MindSight uses **YOLOE**
and lets you define the object classes *visually*: you show the detector a few
example images with boxes drawn around your objects, and it detects things that
look like them. That definition lives in a **visual prompt (VP) file**.

This page explains what a VP file is, how to build one, and how to make prompts
that actually detect well. It is written generically -- every study supplies
its own objects and its own reference footage.

## What a VP file is

A VP file is JSON with the extension `.vp.json`. It encodes two things: the
list of object classes your study cares about, and one or more reference images
with bounding-box annotations showing YOLOE what each class looks like.

```json
{
  "version": 1,
  "classes": [
    {"id": 0, "name": "knife"},
    {"id": 1, "name": "plate"}
  ],
  "references": [
    {
      "image": "/absolute/path/to/reference.jpg",
      "annotations": [
        {"cls_id": 0, "bbox": [x1, y1, x2, y2]},
        {"cls_id": 1, "bbox": [x1, y1, x2, y2]}
      ]
    }
  ]
}
```

A few rules the format enforces:

- `classes` use sequential integer IDs starting at `0`, and they must be
  contiguous.
- `references` is a list of reference images, each with a set of annotated
  boxes referencing those class IDs.
- The **first** reference image initialises YOLOE's class embeddings.
  Additional reference images are currently reserved for future use, so the
  first image is the one that matters.

## Building one in the VP Builder

You do not hand-edit that JSON. MindSight's GUI includes a **VP Builder** tab
that produces valid `.vp.json` files by pointing and clicking:

1. **Add images** -- one or more frames containing the objects you want to
   detect. Sampling from your actual study footage works far better than using
   stock photos.
2. **Add a class** for each object category your study cares about.
3. **Draw bounding boxes** by click-and-drag on the canvas. Each box is
   assigned to whichever class is currently selected.
4. **Save the VP file** to write the `.vp.json`.
5. **Test inference** -- point a YOLOE model at a folder of test images and
   preview the detections before committing to a full run.

Once you have a VP file, place it in your project's `Inputs/Prompts/` directory
(or point at it with the `vp_file` key in your pipeline config). In project
mode, if you do not name one explicitly, the first `.vp.json` found in
`Inputs/Prompts/` is loaded automatically -- so keep one VP file per project to
avoid surprises.

## Making prompts that detect well

Visual prompting is powerful but sensitive to how you set it up. The following
guidance is drawn from the MindSight paper and from practical tuning
experience.

- **Sample from your study footage.** Prompts defined from your actual
  recordings consistently beat prompts built from external reference images.
  The detector is matching *appearance*, so the closer the reference is to what
  the camera actually sees, the better.
- **Match prompt resolution to video resolution.** YOLOE encodes class
  embeddings from the *pixel size* of the example objects, not their semantic
  identity. If your prompt images are 4K and your video is 720p, objects appear
  at different scales and confidence collapses. Normalising the prompt
  resolution to roughly match your expected video resolution makes a noticeable
  difference.
- **Watch for low colour contrast.** Objects that do not visually pop against
  their background are the most common detection failure. If you are choosing
  study materials, prefer items that clearly contrast with the table, floor, or
  walls they will sit on.
- **Use a lower confidence threshold than you would expect.** Visual prompts
  often need a detection confidence around `0.20`--`0.30`, whereas text-class
  YOLO is comfortable around `0.35`--`0.50`. Pair the lower threshold with
  overlap merging to suppress duplicate boxes.
- **Larger and less occluded is more reliable.** Small, ambiguous, or
  frequently occluded objects are inherently harder to detect. Where you have
  control over the scene, keep task-relevant items visually distinct and
  minimally occluded.

## Where the parameters live

The detection settings that go with a visual prompt -- which YOLOE model, the
confidence threshold, overlap merging -- are documented in the
[pipeline.yaml schema](../reference/pipeline-yaml-schema.md) and set for you in
the shipped `configs/pipeline_known_good.yaml`, validated on classroom-style
footage. The [CLI flags reference](../reference/cli-flags.md) lists the
equivalent command-line options. For how detection fits into the run as a
whole, see [The MindSight Pipeline](../concepts/pipeline.md).
