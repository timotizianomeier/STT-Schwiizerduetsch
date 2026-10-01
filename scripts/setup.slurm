#!/bin/bash
# Phase 3 setup as a batch job. gpucluster2 is a 4-core submit VM ("solely
# for submitting sbatch jobs"), so environment install, extraction and the
# dataset build run on a compute node. CPU-only work: no GPU requested.
#
# Submit from the repo root:   sbatch scripts/setup.slurm
# Watch:                       tail -f logs/setup-<jobid>.out
#SBATCH --job-name=stt-setup
#SBATCH --partition=a16
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/setup-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  start: $(date)"
WORKERS=$SLURM_CPUS_PER_TASK bash scripts/cluster_setup.sh
echo "end: $(date)"
