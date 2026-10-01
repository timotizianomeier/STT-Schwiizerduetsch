#!/usr/bin/env bash
# One-shot cluster setup for Phase 3: environment, extraction, dataset build.
# Idempotent: each step is skipped if its output already exists, so a killed
# run can simply be started again.
#
# Everything lives under /vol/bitbucket/$USER/stt because the home quota is
# nearly full (DECISIONS 2026-08-25): repo, venv, uv cache, HF cache, data.
#
# Usage (from the repo root on the cluster):
#     bash scripts/cluster_setup.sh            # 4 workers, niced
#     WORKERS=8 bash scripts/cluster_setup.sh  # inside an srun/salloc allocation
set -euo pipefail

BASE=/vol/bitbucket/$USER
STT=$BASE/stt
RAW=$STT/data/raw
PROC=$STT/data/processed
WORKERS=${WORKERS:-4}

# Reuse the caches that already exist on bitbucket (discovered 2026-10-01):
# uv-cache (4 GB of wheels from earlier projects) and hf (model hub cache).
export UV_CACHE_DIR=$BASE/uv-cache
export UV_PYTHON_INSTALL_DIR=$BASE/uv-python
export HF_HOME=$BASE/hf
export PATH=$HOME/.local/bin:$BASE/bin:$PATH      # uv lives in ~/.local/bin
mkdir -p "$BASE/bin" "$UV_CACHE_DIR" "$HF_HOME" "$PROC"

step() { echo; echo "=== $* ==="; }

step "0. sanity"
ls -lh "$RAW"/swissubase_2277_1_0.zip "$RAW"/swissubase_2269_1_0.zip
df -h "$BASE" | tail -1
quota -s 2>/dev/null | tail -2 || true

step "1. uv"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BASE/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

step "2. python environment (uv sync)"
uv sync
uv run python -c "import torch, transformers, peft, soundfile, librosa; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

step "3. transcriptions"
if [ ! -d "$RAW/archimob_r2_text/Archimob_Release_2" ]; then
    unzip -q -o "$RAW/swissubase_2269_1_0.zip" -d "$RAW/archimob_r2_text"
    unzip -q -o "$RAW/archimob_r2_text/Archimob_Release_2.zip" -d "$RAW/archimob_r2_text"
fi
ls "$RAW/archimob_r2_text/Archimob_Release_2" | wc -l

step "4. audio extraction (23.5 GB, 78k files)"
if [ ! -f "$RAW/audio/.extracted" ]; then
    nice uv run python scripts/extract_audio.py "$RAW/swissubase_2277_1_0.zip" "$RAW/audio"
    touch "$RAW/audio/.extracted"
fi
find "$RAW/audio" -name '*.wav' | wc -l

step "5. utterance table + splits"
uv run python src/parse_archimob.py "$RAW/archimob_r2_text/Archimob_Release_2" "$PROC/archimob_utterances.jsonl"
uv run python src/splits.py "$PROC/archimob_utterances.jsonl"
uv run python src/metrics.py "$PROC/archimob_utterances.jsonl" "$PROC/dieth_to_norm.json" | head -2

step "6. dataset build (resample to 16 kHz, filter, manifests)"
nice uv run python src/prepare.py "$PROC/archimob_utterances.jsonl" "$RAW/audio" "$PROC/archimob16k" --workers "$WORKERS"

step "7. tests"
uv run pytest -q

step "DONE — paste everything from here down back to Claude"
cat "$PROC/archimob16k/summary.json"
du -sh "$PROC/archimob16k" "$RAW/audio"
wc -l "$PROC"/archimob16k/manifest_*.jsonl
sinfo -o "%P %G %l %D %t" 2>/dev/null | head -30 || true
