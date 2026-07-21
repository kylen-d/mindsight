# Installing MindSight on Windows

This is the interim Windows installer for MindSight. It sets everything up for
you: a self-contained Python, all of MindSight's dependencies, the model
weights, and shortcuts to launch the app. You do not need to install Python or
anything else first.

## What you need

- Windows 10 or Windows 11 (64-bit).
- About 4 GB of free disk space (dependencies, plus the model weights).
- An internet connection for the one-time setup.
- Optional: an NVIDIA GPU. If you have one, the installer automatically
  installs the GPU build of PyTorch. If you do not, MindSight runs on the CPU --
  everything still works, just slower.

## Install

1. Download the release zip (for example `MindSight-SP4.0-win.zip`).
2. Right-click the zip and choose **Extract All...**, then extract it to a
   folder you can find again (your Desktop or Downloads is fine).
3. Open the extracted folder and double-click **`Install-MindSight.bat`**.
4. Windows may show a blue **"Windows protected your PC"** SmartScreen box
   because the installer is not signed. Click **More info**, then **Run
   anyway**. (This is expected for an in-house tool.)
5. A console window opens and walks through seven steps, printing a line for
   each one. Leave it running -- installing the dependencies can take a few
   minutes. When it finishes it prints either:
   - `MindSight install: PASS`, or
   - `MindSight install: FAILED at step N` with a short explanation.
6. Press a key to close the window when it is done.

The installer is safe to run again. Re-running it updates an existing install
and skips work that is already complete.

## First launch

Launch MindSight from the **MindSight** shortcut on your Desktop or in the
Start Menu. Both shortcuts carry the MindSight icon.

On the first run, MindSight downloads the six required model weights (about
141 MB total). This happens during the install step, not at launch, so by the
time you open the app they are already in place:

| Model                              | File                       | Size    |
| ---------------------------------- | -------------------------- | ------- |
| YOLO detector                      | `yolov8n.pt`               | ~6 MB   |
| YOLO detector (v11 default)        | `yolo11n.pt`               | ~6 MB   |
| Gaze-LLE DINOv3-distilled (pico)   | `gazelle_hgnetv2_pico_inout_distill_1x3x640x640_1xNx4.onnx` | ~16 MB |
| Gaze-LLE (DINOv2 ViT-B/14)         | `gazelle_dinov2_vitb14.pt` | ~12 MB  |
| MobileGaze (ResNet-50)             | `resnet50_gaze.onnx`       | ~91 MB  |
| MobileGaze (MobileOne-S0)          | `mobileone_s0_gaze.onnx`   | ~5 MB   |

Before it analyzes footage, MindSight runs a short **preflight** check. Part of
that check verifies each model weight against a published checksum, so you know
the files downloaded correctly and were not corrupted or tampered with. If a
weight is missing or fails its checksum, preflight reports it in plain language.

## Where your files live

Everything MindSight installs lives under:

```
%LOCALAPPDATA%\MindSight
```

That is usually `C:\Users\<you>\AppData\Local\MindSight`. Inside it:

- `app\` -- the MindSight program and its `Weights\`, `Outputs\`, and
  `Projects\` folders. This is where your analysis outputs are written, so it
  is the folder you will open to find your results.
- `venv\` -- the private Python environment. You never need to touch this.

## Uninstall

MindSight does not write to the Windows registry. To remove it completely:

1. Delete the folder `%LOCALAPPDATA%\MindSight`.
2. Delete the **MindSight** shortcut from your Desktop and from the Start Menu.

That's it. (Back up anything you want to keep from `app\Outputs\` or
`app\Projects\` first.)

## Troubleshooting

| Symptom                                            | What to do |
| -------------------------------------------------- | ---------- |
| SmartScreen blocks the installer                   | Click **More info**, then **Run anyway**. The installer is unsigned; this is expected. |
| `FAILED at step 1` (installing uv)                 | Check your internet connection and that your network allows downloads from `astral.sh`. Then run the installer again. |
| `FAILED at step 4` (installing dependencies)       | Usually a dropped connection. Re-run the installer; it resumes where it left off. |
| `FAILED at step 6` (downloading weights)           | A network hiccup during the model download. Re-run the installer, or later run `mindsight-weights --required` from the install's `venv\Scripts\` folder. |
| Antivirus quarantines a file                       | Some antivirus tools flag freshly downloaded executables. Allow the `MindSight` folder under `%LOCALAPPDATA%`, then re-run. |
| No NVIDIA GPU                                       | Nothing to do. MindSight runs on the CPU automatically. Analysis is slower but produces identical results. |
| I have an NVIDIA GPU but it is not being used      | The installer only picks the GPU build when `nvidia-smi` reports a working GPU. Make sure your NVIDIA driver is installed, then re-run the installer. |
| I want to update to a newer MindSight              | Extract the new zip and double-click `Install-MindSight.bat` again. It updates in place. |

If a step fails, the installer leaves a partial folder at
`%LOCALAPPDATA%\MindSight`. You can safely delete that folder and run the
installer again from scratch.
