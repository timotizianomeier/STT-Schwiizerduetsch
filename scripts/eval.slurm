#!/bin/bash
# Phase 6: score a fine-tuned adapter (or the bare model) on a split.
#   RUN=v2_folded sbatch scripts/eval.slurm              # runs/<RUN>/best on test
#   RUN=v1_accents SPLIT=dev sbatch scripts/eval.slurm
# A16 is enough: fp16 inference, ~30 min for the 3.8 h test set.
#SBATCH --job-name=stt-eval
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/eval-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
BASE=/vol/bitbucket/$USER
PROC=$BASE/stt/data/processed
export UV_CACHE_DIR=$BASE/uv-cache HF_HOME=$BASE/hf PATH=$HOME/.local/bin:$PATH
RUN=${RUN:?set RUN to a directory under runs/}
SPLIT=${SPLIT:-test}
CKPT=${CKPT:-best}
OUT=$BASE/stt/runs/$RUN/eval_${SPLIT}_${CKPT}

echo "node: $(hostname)  run: $RUN/$CKPT  split: $SPLIT  start: $(date)"
cd src
uv run python evaluate.py \
    --manifest "$PROC/archimob16k/manifest_$SPLIT.jsonl" \
    --data-root "$PROC/archimob16k" \
    --variants "$PROC/dieth_to_norm.json" \
    --model "${MODEL:-openai/whisper-large-v3}" \
    --adapter "$BASE/stt/runs/$RUN/$CKPT" \
    --batch-size 16 --examples 50 \
    --title "$RUN ($CKPT) on $SPLIT" \
    --out "$OUT"
echo "outputs in $OUT"; echo "end: $(date)"
