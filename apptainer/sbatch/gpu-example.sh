#!/bin/bash
# MindSight GPU batch job for UBC ARC Sockeye.
# Copy this file, then replace every <alloc> and the input paths.
#
#SBATCH --job-name=mindsight
#SBATCH --account=st-<alloc>-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=mindsight-%j.out
set -euo pipefail

module load apptainer 2>/dev/null || true  # harmless if apptainer is already on PATH

LAB_DIR=/arc/project/st-<alloc>-1/mindsight
SIF="$LAB_DIR/mindsight-v1.3.2.sif"
SHARED_WEIGHTS="$LAB_DIR/weights-shared"

# Per-user writable home on scratch (fast + large); results copied back below.
# MINDSIGHT_HOME is passed with --env on every exec: HPC apptainer configs
# often strip host env vars (cleanenv), and --env survives that.
export MINDSIGHT_HOME=/scratch/st-<alloc>-1/$USER/mindsight-home
apptainer exec --env MINDSIGHT_HOME="$MINDSIGHT_HOME" --bind /arc,/scratch "$SIF" \
    mindsight-seed-home "$MINDSIGHT_HOME" --shared-weights "$SHARED_WEIGHTS"

# --- Single video ---
apptainer exec --nv --env MINDSIGHT_HOME="$MINDSIGHT_HOME" --bind /arc,/scratch "$SIF" \
    mindsight --source /arc/project/st-<alloc>-1/videos/session01.mp4 --save

# --- Or project mode: process every staged video in a study directory ---
# apptainer exec --nv --env MINDSIGHT_HOME="$MINDSIGHT_HOME" --bind /arc,/scratch "$SIF" \
#     mindsight --project /arc/project/st-<alloc>-1/studies/study-A

# Keep results: copy outputs to backed-up project space.
rsync -a "$MINDSIGHT_HOME/Outputs/" "$LAB_DIR/outputs/$USER/job-$SLURM_JOB_ID/"
