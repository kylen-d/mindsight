# Installing MindSight on macOS

This is the interim macOS installer for MindSight. It sets everything up for
you: a self-contained Python, all of MindSight's dependencies, the model
weights, and a launcher to open the app. You do not need to install Python or
anything else first.

## What you need

- macOS on Apple Silicon or Intel.
- About 4 GB of free disk space (dependencies, plus the model weights).
- An internet connection for the one-time setup.

On Apple Silicon, MindSight uses the Mac's GPU (Metal / MPS) automatically
through the standard PyTorch build. There is nothing to configure; on an Intel
Mac it runs on the CPU.

## Install

1. Download the release zip (for example `MindSight-SP4.0-mac.zip`).
2. Double-click the zip in Finder to extract it. Move the extracted folder
   somewhere you can find again (your Desktop or Downloads is fine).
3. Open the extracted folder. **Right-click** (or Control-click)
   `Install-MindSight.command` and choose **Open**.
4. macOS Gatekeeper may warn that the file is from an unidentified developer.
   Because you used **right-click > Open**, the dialog gives you an **Open**
   button -- click it. (Double-clicking a downloaded `.command` only offers
   "Move to Trash"; right-click > Open is the way past this. It is expected for
   an in-house tool that is not code-signed.)
5. A Terminal window opens and walks through six steps, printing a line for
   each one. Leave it running -- installing the dependencies and downloading
   the weights can take several minutes. When it finishes it prints either:
   - `MindSight install: PASS`, or
   - `MindSight install: FAILED at step N` with a short explanation.
6. Press Return to close the window when it is done.

The installer is safe to run again. Re-running it updates an existing install
and skips work that is already complete.

## First launch

The installer creates a **MindSight** app in your Applications folder
(`/Applications/MindSight.app`, or `~/Applications/MindSight.app` if
`/Applications` needs an administrator) and puts a matching **MindSight** link
on your Desktop. Open either one to launch MindSight -- it shows the MindSight
name and icon in the Dock. The first time you open it, Gatekeeper may prompt
again -- use **right-click > Open** once, and after that a plain double-click
works.

The four required model weights (about 115 MB total) are downloaded during the
install step, not at launch, so by the time you open the app they are already
in place:

| Model                         | File                       | Size    |
| ----------------------------- | -------------------------- | ------- |
| YOLO detector                 | `yolov8n.pt`               | ~6 MB   |
| Gaze-LLE (DINOv2 ViT-B/14)    | `gazelle_dinov2_vitb14.pt` | ~12 MB  |
| MobileGaze (ResNet-50)        | `resnet50_gaze.onnx`       | ~91 MB  |
| MobileGaze (MobileOne-S0)     | `mobileone_s0_gaze.onnx`   | ~5 MB   |

Before it analyzes footage, MindSight runs a short **preflight** check. Part of
that check verifies each model weight against a published checksum, so you know
the files downloaded correctly and were not corrupted or tampered with. If a
weight is missing or fails its checksum, preflight reports it in plain language.

## Where your files live

Everything MindSight installs lives under:

```
~/MindSight
```

That is your home folder's `MindSight` directory (visible in Finder under your
account). Inside it:

- `app/` -- the MindSight program and its `Weights/`, `Outputs/`, and
  `Projects/` folders. This is where your analysis outputs are written, so it
  is the folder you will open to find your results.
- `venv/` -- the private Python environment. You never need to touch this.

## Uninstall

To remove MindSight completely:

1. Delete the folder `~/MindSight`.
2. Delete **MindSight.app** from `/Applications` (or `~/Applications`).
3. Delete the **MindSight** link from your Desktop.

That's it. (Back up anything you want to keep from `app/Outputs/` or
`app/Projects/` first.)

## Troubleshooting

| Symptom                                            | What to do |
| -------------------------------------------------- | ---------- |
| Double-clicking the installer only offers "Move to Trash" | Use **right-click > Open** instead of double-click, then click **Open** in the Gatekeeper dialog. |
| `FAILED at step 1` (installing uv)                 | Check your internet connection and that your network allows downloads from `astral.sh`. Then run the installer again. |
| `FAILED at step 4` (installing dependencies)       | Usually a dropped connection. Re-run the installer; it resumes where it left off. |
| `FAILED at step 5` (downloading weights)           | A network hiccup during the model download. Re-run the installer, or later run `mindsight-weights --required` from `~/MindSight/venv/bin/`. |
| The app will not open                              | Use **right-click > Open** on **MindSight.app** the first time. macOS remembers your choice after that. |
| I want to update to a newer MindSight              | Extract the new zip and run `Install-MindSight.command` again. It updates in place. |

If a step fails, the installer leaves a partial folder at `~/MindSight`. You
can safely delete that folder and run the installer again from scratch.
