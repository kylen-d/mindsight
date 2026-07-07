# Visual prompts go here

Save your study's visual prompt as a `.vp.json` file in this folder (export it
from the VP Builder). The known-good pipeline preset uses a YOLOE detector that
is designed to run with a visual prompt.

- One `.vp.json` per study is typical; the pipeline picks it up automatically.
- The prompt's reference image should match your video resolution -- YOLOE
  encodes pixel size, not semantics.

No prompt file is shipped here on purpose: the visual prompt is study data you
prepare. This placeholder can be deleted once you add a real `.vp.json`.
