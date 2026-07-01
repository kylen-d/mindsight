# G-WIN dry run -- Windows install checklist (user-assisted)

This is the SP4.0 acceptance gate that cannot be run from macOS: a real
Windows machine must run the installer from the actual release zip. Hand the
built `MindSight-SP4.0-win.zip` to the Windows labmate (or a Windows VM) and
walk through the steps below. Copy the console transcript back for the record.

## Before you start (record these)

- [ ] Windows version (`winver`): __________________
- [ ] NVIDIA GPU present? (`nvidia-smi` in a terminal) yes / no
- [ ] SHA-256 of the zip handed over matches the one the builder printed:
      `certutil -hashfile MindSight-SP4.0-win.zip SHA256`
      value: __________________

## Install

1. [ ] Extract the zip to a normal folder (Desktop or Downloads).
2. [ ] Double-click `Install-MindSight.bat`.
3. [ ] SmartScreen appears -> **More info** -> **Run anyway**.
4. [ ] Watch each step print `[ N/7] ... OK`. Note any step that stalls.
5. [ ] Confirm the final line is `MindSight install: PASS`.
   - If it prints `FAILED at step N`, copy the whole transcript and stop --
     report step N and its message.

## Verify the install

6. [ ] A **MindSight** shortcut exists on the Desktop and in the Start Menu.
7. [ ] The folder `%LOCALAPPDATA%\MindSight\app\Weights` contains the four
       required weights (`yolov8n.pt`, `gazelle_dinov2_vitb14.pt`,
       `resnet50_gaze.onnx`, `mobileone_s0_gaze.onnx`).
8. [ ] Launch MindSight from the shortcut. Confirm the GUI window opens.
9. [ ] (If a GPU is present) confirm step 5 of the transcript said
       "NVIDIA GPU detected -- installing CUDA build of PyTorch ... OK".
       (If no GPU) confirm it said "keeping CPU PyTorch ... OK".

## Idempotency

10. [ ] Double-click `Install-MindSight.bat` a second time. Confirm it prints
        "already"/"present"-style lines for finished work and still ends in
        `MindSight install: PASS` (a re-run must not break the install).

## Report back

- [ ] Full console transcript of the first run.
- [ ] Whether the GUI launched.
- [ ] GPU present? and which torch build step 5 chose.
- [ ] Any step that failed, with its message.

Any failure that installer-script edits can fix -> fix-forward within Batch C,
rebuild the zip, re-hand it. A failure that needs a dependency/wheel change
(a package missing a Windows 3.12 wheel, a CUDA torch conflict) is an
escalation -- report it, do not force pins.
