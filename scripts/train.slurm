#!/bin/bash
# Phase 5: LoRA fine-tune. Same script for the smoke job and the real run:
#   smoke:  RUN=smoke MAX_STEPS=300 EVAL_EVERY=100 sbatch --time=01:00:00 scripts/train.slurm
#   real:   RUN=v1 MAX_STEPS=4000 sbatch scripts/train.slurm
# Re-submitting with the same RUN resumes from runs/<RUN>/last.
#
# A40 (48 GB): large-v3 + LoRA at batch 16 in bf16 fits without gradient
# checkpointing. Smaller cards (A30 24 GB) need BATCH=8 ACCUM=4.
#SBATCH --job-name=stt-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BASE=/vol/bitbucket/$USER
PROC=$BASE/stt/data/processed
export UV_CACHE_DIR=$BASE/uv-cache HF_HOME=$BASE/hf PATH=$HOME/.local/bin:$PATH
RUN=${RUN:-v1}

echo "node: $(hostname)  run: $RUN  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
uv run python train.py \
    --train-manifest "$PROC/archimob16k/manifest_train.jsonl" \
    --dev-manifest "$PROC/archimob16k/manifest_dev.jsonl" \
    --data-root "$PROC/archimob16k" \
    --variants "$PROC/dieth_to_norm.json" \
    --out "$BASE/stt/runs/$RUN" \
    --model "${MODEL:-openai/whisper-large-v3}" \
    --max-steps "${MAX_STEPS:-4000}" \
    --batch-size "${BATCH:-16}" --accum "${ACCUM:-2}" \
    --eval-every "${EVAL_EVERY:-250}" --save-every "${SAVE_EVERY:-250}" \
    --lr "${LR:-2e-4}" --workers 6
echo "end: $(date)"
