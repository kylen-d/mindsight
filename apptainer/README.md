# MindSight on UBC ARC (Sockeye) — Apptainer runbook

## What this is

A single, reproducible Apptainer image (`.sif`) containing MindSight v1.3.2,
its complete locked dependency set, and its default model weights — built for
Sockeye, where compute nodes have **no internet access**.

Reproducibility guarantees:

- **Base image pinned by digest** (`ubuntu:24.04@sha256:4fbb8e…`), never a
  floating tag.
- **Dependencies installed with `uv sync --frozen`** against the repo's
  committed `uv.lock` — the exact CUDA-12 torch wheel set MindSight releases
  with; no resolution happens at build time. `uv` itself is version-pinned and
  checksummed.
- **The 6 required model weights (~141 MB) are baked in**, each sha256-verified
  against `weights_manifest.json` during the build.
- The source commit and build date are stamped into `/opt/mindsight/BUILD_INFO`
  inside the image; OCI labels carry version/license/base-digest.
- The **`.sif` sha256 printed in the CI job summary is the canonical artifact
  identity** — record it alongside results for provenance.

## Lab layout on Sockeye

```
/arc/project/st-<alloc>-1/mindsight/
├── mindsight-v1.3.2.sif      # the image (pull once per version)
├── weights-shared/           # optional weights, downloaded once for the lab
└── outputs/<user>/...        # copied-back results
```

`/arc/project` is backed up and quota'd; `/scratch` is large and transient.
Per-user working homes (below) can live in either — scratch for heavy runs,
with results rsynced back (the sbatch template does this).

## One-time lab setup (admin, on a LOGIN node)

1. **Pull the image** (login nodes have internet; compute nodes do not):

   ```bash
   cd /arc/project/st-<alloc>-1/mindsight
   apptainer pull mindsight-v1.3.2.sif oras://ghcr.io/kylen-d/mindsight:v1.3.2
   ```

   > **GHCR visibility:** the first CI push creates the package **private**.
   > Either make it public once (GitHub → Packages → mindsight → settings), or
   > every puller must `apptainer registry login -u <user> --password-stdin
   > oras://ghcr.io` with a PAT that has `read:packages`.

2. **Download the lab's optional weights** into the shared dir (any subset;
   `--all` is ~1.5 GB):

   ```bash
   export MINDSIGHT_HOME=/arc/project/st-<alloc>-1/mindsight/weights-tmp-home
   apptainer exec --bind /arc mindsight-v1.3.2.sif \
       mindsight-seed-home "$MINDSIGHT_HOME"
   apptainer exec --bind /arc mindsight-v1.3.2.sif \
       mindsight-weights --all          # or: --backend Gazelle --backend YOLO
   # real files land next to the symlinks; move them into the shared dir:
   mkdir -p weights-shared
   rsync -a --no-links "$MINDSIGHT_HOME/Weights/" weights-shared/
   rm -rf "$MINDSIGHT_HOME"
   ```

3. **`mobileclip_blt.ts` caveat (YOLOE text prompts only):** this file is
   marked `ultralytics-auto` in the manifest — ultralytics fetches it on first
   use, which **fails on a compute node**. If your study uses YOLOE **text
   prompts**, run one tiny YOLOE text-prompt invocation on the login node
   first (with `MINDSIGHT_HOME` set as above) so the file is cached in the
   writable home/shared dir before batch jobs need it.

## Per-user setup

```bash
export MINDSIGHT_HOME=/scratch/st-<alloc>-1/$USER/mindsight-home   # or project space
apptainer exec --bind /arc,/scratch /arc/project/st-<alloc>-1/mindsight/mindsight-v1.3.2.sif \
    mindsight-seed-home "$MINDSIGHT_HOME" \
    --shared-weights /arc/project/st-<alloc>-1/mindsight/weights-shared
```

`mindsight-seed-home` symlinks the baked + shared weights and the manifest
into your home and creates a writable `Outputs/`. It never replaces a real
file you downloaded yourself, and re-running it (e.g. after an image upgrade)
is safe. Add the `export MINDSIGHT_HOME=...` line to your `~/.bashrc` or your
job scripts.

- Home in **project space**: outputs backed up, but they count against the
  allocation quota.
- Home in **`/scratch`**: big and fast, but transient — copy results back
  (see the sbatch template).

## Batch runs

Start from [`sbatch/gpu-example.sh`](sbatch/gpu-example.sh) — replace
`<alloc>` and the input paths, then `sbatch` it. The essentials:

- **`--nv` exposes the GPU** to the container. Without it, torch silently
  runs CPU-only.
- **`--bind /arc,/scratch` is required** — Sockeye's filesystems are not
  auto-bound into the container.
- Single video: `mindsight --source <video.mp4> --save`
- Whole study: `mindsight --project <study-dir>` (processes all staged
  videos using the project's `Pipeline/pipeline.yaml`, with resume).

## GUI

```bash
ssh -X sockeye.arc.ubc.ca          # from a machine with an X server
apptainer exec --bind /arc,/scratch $SIF mindsight-gui
```

Runs CPU-only on the login node — fine for configuration, project setup, and
reviewing outputs; do the heavy processing via sbatch. The image includes the
full Qt xcb library stack **including `libxcb-cursor0`**, the library whose
absence broke the GUI in the pre-container ARC attempt.

## Building a new version

**Paved path (CI):** push a `v*` tag — the `build-sif` workflow builds the
image, runs its self-tests, pushes `oras://ghcr.io/kylen-d/mindsight:<tag>`,
and prints the `.sif` sha256 in the job summary. For a trial build from a
branch, use *Actions → build-sif → Run workflow* with a ref; it pushes a
`dev-<run_id>` tag instead.

**Local fallback (any Linux box):**

```bash
apptainer build ~/mindsight-vX.Y.Z.sif apptainer/mindsight.def
```

**Always build outside the source tree** — the def's `%files` stanza copies
the whole tree, so an in-tree `.sif` would be embedded into the next build.

## On-cluster validation checklist (after each new image)

1. GPU visible: `apptainer exec --nv $SIF python -c "import torch; print(torch.cuda.is_available())"`
   → `True` on a GPU node.
2. One real clip end-to-end on a GPU node; CSVs appear under
   `$MINDSIGHT_HOME/Outputs/`.
3. `mindsight-gui` opens over `ssh -X` on a login node.
4. A shared-dir-only optional weight resolves (run a model that exists only
   in `weights-shared/`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not load the Qt platform plugin "xcb"` | Missing X libraries — should not happen with this image. Check you used `ssh -X` and `$DISPLAY` is set; `could not connect to display` means X forwarding, not the image. |
| `torch.cuda.is_available()` → `False` | Forgot `--nv`, or the job isn't on a GPU node. |
| `note: MINDSIGHT_HOME ... is not writable` | You're using the baked read-only home. Run `mindsight-seed-home` and export `MINDSIGHT_HOME` (see Per-user setup). |
| `Permission denied` under `/arc` or `/scratch` | Missing `--bind /arc,/scratch`. |
| Weight download fails mid-job | Expected — compute nodes have no internet. Download on a login node (shared dir or your home) first. |

## Future

Remote batch offload — the desktop GUI submitting Slurm jobs to Sockeye over
SSH using this same image as the payload — is designed-for but not yet
implemented; see `docs/superpowers/specs/2026-07-27-apptainer-packaging-design.md`.
