# MindSight release checklist

The exact manual sequence for cutting a MindSight release. Every step here is a
**[USER]** action -- the release workflow (`.github/workflows/release.yml`) is
prepared but never triggers itself: it runs only when a human pushes a version
tag. Nothing below is automated by the repo.

## 0. Preconditions

- [ ] `main` is green: `pytest -q` passes and `ruff check .` is clean (use
      whatever dev environment the repo's contributing notes specify).
- [ ] The three CLI smokes match the golden baselines (blend SSIM >= 0.99,
      Done-lines 0/1037/615, `Backend: MGaze ONNX`, CSVs byte-identical).
- [ ] `weights_manifest.json` lists the four required weights with correct
      `sha256`/`size` and the version in `pyproject.toml` is the one you intend
      to ship (the version is static -- edit it deliberately, in its own commit).
- [ ] The wheel builds locally: `python -m build` produces a `.whl` and
      `.tar.gz` under `dist/` without error.

## 1. Review

- [ ] Read the diff since the previous tag. Confirm no weights, `Outputs/`,
      `localref/`, or local dev-environment paths are staged.
- [ ] Dry-run both installer zips and confirm the content census passes:
      ```
      python installer/make_release_zip.py --platform win --out dist/
      python installer/make_release_zip.py --platform mac --out dist/
      ```
      Each run must end in `RESULT: PASS (zero forbidden hits)`.

## 2. Push and tag

- [ ] `git push origin main`
- [ ] Create the version tag (matching `pyproject.toml`) and push it:
      ```
      git tag v1.0.0
      git push origin v1.0.0
      ```
      Pushing the `v*` tag is what starts the Release workflow.

## 3. Verify the workflow artifacts

- [ ] Watch the `Release` workflow run to completion (Actions tab). It builds
      the wheel + sdist and both installer zips (each zip's census must pass),
      then publishes a GitHub Release for the tag.
- [ ] Confirm the Release carries every asset: `MindSight-*-win.zip`,
      `MindSight-*-mac.zip`, the `.whl`, the `.tar.gz` sdist, `uv.lock`,
      `weights_manifest.json`, both installer scripts, and both INSTALL docs.

## 4. Install from the release (both platforms)

- [ ] **Windows**: download `MindSight-*-win.zip` from the Release, extract it,
      double-click `Install-MindSight.bat`. Confirm it reaches
      `MindSight install: PASS`, the Desktop/Start Menu shortcuts appear, and
      the GUI launches. (This is the G-WIN labmate check when a fresh machine
      is available.)
- [ ] **macOS**: download `MindSight-*-mac.zip`, extract it, right-click
      `Install-MindSight.command` > Open. Confirm `MindSight install: PASS`,
      the Desktop launcher appears, and the GUI launches.

## Install modes (for reference)

The installers ship in **local-zip mode** by default: the zip bundles an `app/`
source tree, MindSight is installed editable from it, and `MINDSIGHT_HOME` stays
unset so `PROJECT_ROOT` resolves to that tree (byte-identical to running from a
git checkout).

**Release mode** is prepared for a future wheel-only distribution: a zip built
without an `app/` tree, with `MINDSIGHT_RELEASE_WHEEL_URL` pointing at the wheel
asset on the Release. In that mode the installer creates a venv, installs the
wheel non-editable, and sets `MINDSIGHT_HOME` to the install's data folder so
`Weights/`, `Outputs/`, and `Projects/` never land under `site-packages`.
Local-zip mode remains the default until v1.0 wheel assets are published.
