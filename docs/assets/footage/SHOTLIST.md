# GUI Guides -- demo footage shot list

This is the committed shot list for the short screen-capture clips ("Demo coming
soon" placeholders) that the GUI Guides pages reference. Every `SHOT:<id>` that
appears in a guide page has exactly one row here, and every row here is
referenced by a page -- keep the two in sync.

## Recording notes

- **Scale.** Record at 2x display scale (Retina / "More Space" off) so text
  stays crisp when the clip is scaled down in the docs.
- **Aspect.** Frame roughly 16:10 -- the app's default window shape. Crop tightly
  to the panel that matters; do not capture the whole desktop.
- **Length.** Target under 25 seconds. Trim to a clean loop (start and end on the
  same idle state) so the clip reads as a short repeatable demo.
- **Content.** Use scratch or staged content ONLY -- the `Projects/ExampleStudy/`
  sample project, throwaway recordings, or synthetic footage. **Never** record
  real participant footage or a real study's outputs.
- **State.** Run against a fake HOME so no real `~/.mindsight` / `~/MindSight` is
  touched; delete any camera recordings the capture creates.
- **Naming (B3b).** `<id>.png`/`<id>.gif` for the light-mode capture and
  `<id>-dark` for the dark-mode variant, inserted as `=== "Light"` / `=== "Dark"`
  content tabs.

## Shots

| SHOT id | Page | What to capture | Est. length |
|---------|------|-----------------|-------------|
| `gui-tour` | getting-started/quickstart-gui.md | Slow walkthrough of the full window: land on Analyze Footage, click across all six tabs, then open the menu bar (File / View / Tools / Help). | ~25s |
| `theme-toggle` | getting-started/quickstart-gui.md, guides/about-and-theming.md | View > Theme switched auto -> light -> dark, showing the whole window recolour live. | ~10s |
| `quick-analysis` | guides/analyze-footage.md | Analyze Footage in Video File mode: drag a clip onto the tab, output folder auto-fills, press Analyze, live charts fill the left pane. | ~20s |
| `plan-session` | guides/projects-and-sessions.md | Project overview: click Plan Session, name it and set participant/condition tags, the new row appears as "awaiting recording". | ~18s |
| `record-live-session` | guides/projects-and-sessions.md | Record Session dialog: pick camera, choose a planned session (tags prefill), Start Recording, the timer/frame counter tick, End Session, auto-analysis kicks off. | ~25s |
| `vp-annotate` | guides/visual-prompts.md | VP Builder: add a reference image, add a class, drag a bounding box, assign it, Save VP File. | ~20s |
| `vp-export-portable` | guides/visual-prompts.md | VP Builder: Export Portable... writes a `.vp.zip`; on a second window, Load VP File unpacks it and the references reappear. | ~15s |
| `crop-adjust` | guides/crop-and-adjust.md | Crop & Adjust dialog: step to a video, drag a crop rectangle, set a new fps, queue it, Apply to the batch. | ~22s |
| `tuning-live` | guides/inference-settings-and-tuning.md | Inference Tuning tab: load a clip, Start, the live gaze overlay and the dashboard update as a slider is dragged. | ~20s |
| `about-reader` | guides/about-and-theming.md | About tab: click a guide card, the doc opens in the in-app reader; click an internal link (stays in-app) then an external link (opens the browser). | ~18s |
| `projects-wizard` | guides/projects-and-sessions.md | Build New Project wizard stepping through all five pages (Study -> Videos -> Tag -> Pipeline -> Review) and Create. | ~25s |
