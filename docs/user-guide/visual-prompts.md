# Visual Prompts

## What are Visual Prompts

Traditional object detection requires training a model on text-labeled classes. MindSight supports an alternative approach: **visual prompts**. Instead of specifying class names for a standard YOLO model, you provide reference images with annotated bounding boxes. MindSight feeds these to a YOLOE backbone for zero-shot detection of arbitrary objects -- no retraining required.

This means you can detect custom objects (a specific toy, a medical device, a branded product) simply by showing the model a few example images with bounding boxes drawn around the target objects.

## .vp.json File Format

A visual prompt is stored as a `.vp.json` file with the following schema:

```json
{
  "version": 1,
  "classes": [
    { "id": 0, "name": "coffee_mug" },
    { "id": 1, "name": "phone" }
  ],
  "references": [
    {
      "image": "references/img_001.jpg",
      "annotations": [
        { "cls_id": 0, "bbox": [120, 45, 310, 280] },
        { "cls_id": 1, "bbox": [400, 100, 550, 350] }
      ]
    },
    {
      "image": "references/img_002.jpg",
      "annotations": [
        { "cls_id": 0, "bbox": [80, 60, 260, 240] }
      ]
    }
  ]
}
```

Key rules:

- **version** must be `1`.
- **classes** is an array of objects with `id` (integer) and `name` (string). Class IDs must be contiguous starting from 0.
- **references** is an array of reference images. Each entry contains an `image` path (relative to the `.vp.json` file) and an `annotations` array.
- Each annotation specifies a `cls_id` (matching a class ID) and a `bbox` in `[x1, y1, x2, y2]` pixel coordinates.
- The **first reference** in the array is used for YOLOE initialization. Subsequent references provide additional examples to improve detection.

## Creating with VP Builder GUI

The VP Builder is available as Tab 2 in the MindSight GUI. The workflow is:

1. **Add Images** -- Click "Add Reference Images" to load one or more images that contain your target objects.
2. **Draw Boxes** -- Click and drag on the image canvas to draw bounding boxes around each object instance.
3. **Assign Classes** -- Select a class from the class list (or create a new class) and assign it to each bounding box.
4. **Save VP File** -- Click "Save" to export the visual prompt as a `.vp.json` file along with the reference images.
5. **Test Inference** -- Use the "Test" button to run detection on a sample image and verify that your visual prompt produces good results.

<!-- screenshot: VP Builder -->

## CLI Usage

To use a visual prompt from the command line:

```bash
python MindSight.py --vp-file prompt.vp.json --vp-model yoloe-26l-seg.pt
```

- `--vp-file` specifies the path to your `.vp.json` visual prompt file.
- `--vp-model` specifies the YOLOE model weights to use for visual-prompt-based detection.

These flags replace the standard `--model` and `--classes` arguments. When a VP file is provided, MindSight uses YOLOE instead of the default YOLO text-class pipeline.

## Tips

- **Clear reference images**: Use well-lit, sharp images where the target object is clearly visible and unoccluded.
- **Multiple annotations per class**: Providing several examples of the same class (across different reference images or within the same image) improves detection robustness.
- **Variety matters**: Include references showing the object from different angles, scales, and lighting conditions when possible.
- **Test broadly**: After creating a visual prompt, test it against a variety of scenes to verify that detection generalizes beyond your reference images.
- **Keep it focused**: Each visual prompt works best when targeting a small, well-defined set of object classes.
