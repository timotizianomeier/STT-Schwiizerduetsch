#!/bin/bash
# Phase 4: zero-shot whisper-large-v3 on the held-out test documents.
# Inference in fp16 needs ~4 GB VRAM, so the default A16 partition (16 GB)
# is the smallest GPU that fits; no reason to take an A100 for this.
#
# Submit from the repo root:   sbatch scripts/baseline.slurm
#SBATCH --job-name=stt-baseline
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/baseline-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BASE=/vol/bitbucket/$USER
PROC=$BASE/stt/data/processed
export UV_CACHE_DIR=$BASE/uv-cache HF_HOME=$BASE/hf PATH=$HOME/.local/bin:$PATH
SPLIT=${SPLIT:-test}
MODEL=${MODEL:-openai/whisper-large-v3}
OUT=$BASE/stt/runs/baseline_$(basename "$MODEL")_$SPLIT

echo "node: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
uv run python evaluate.py \
    --manifest "$PROC/archimob16k/manifest_$SPLIT.jsonl" \
    --data-root "$PROC/archimob16k" \
    --variants "$PROC/dieth_to_norm.json" \
    --model "$MODEL" --batch-size 16 --examples 30 \
    --title "Zero-shot $MODEL on $SPLIT (Phase 4 baseline)" \
    --out "$OUT"
echo "outputs in $OUT"; echo "end: $(date)"
