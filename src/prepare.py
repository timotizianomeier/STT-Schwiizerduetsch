"""Phase 3: utterance table + audio chunks -> training-ready dataset.

Input : data/processed/archimob_utterances.jsonl (from parse_archimob + splits)
        <audio_root>/audio_segmented_anonymized/<doc_part>/d<doc_part>_T<n>.wav
Output: <out>/wav16/<split>/<utt_id>.wav     16 kHz mono int16
        <out>/manifest_<split>.jsonl          path, duration, text, metadata
        <out>/summary.json                    hours and drop counts per split

Why not a Hugging Face `DatasetDict` with an `Audio` column, as the brief
says? `datasets` 5.x decodes audio through torchcodec, which needs FFmpeg
shared libraries on every machine. Pre-resampled 16 kHz wavs + a manifest
need only `soundfile`, load in microseconds, and can be listened to with any
player when a transcription looks odd. train.py builds its Dataset from the
manifest (path + text columns) and decodes in the collator.

Audio facts (discovered 2026-09-30, not assumed): chunks are already cut per
utterance; mono; sample rate varies by recording (48 kHz and 44.1 kHz seen)
so every file is resampled individually; the file name is the XML media
pointer with "-" -> "_", and 1,390 files in 1082_3 carry a doubled
"1082_3d1082_3_..." prefix that is stripped.

Filters, in this order, each counted: no wav file; shorter than MIN_SEC
(1.0 s, brief); longer than MAX_SEC (30 s, Whisper's window).

Run (local dev on two extracted docs):
    python src/prepare.py data/processed/archimob_utterances.jsonl \
        data/raw/audio_dev data/processed/dev_local --docs 1225 1055
Run (cluster, everything, 16 workers):
    python src/prepare.py ... --workers 16
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000
MIN_SEC = 1.0
MAX_SEC = 30.0
ODD_PREFIX = re.compile(r"^\d+_\d+(?=d)")


def wav_index(audio_root: Path) -> dict[str, Path]:
    """audio_pointer (XML form, e.g. d1082_3-TLI_12) -> wav path."""
    index = {}
    for p in (audio_root / "audio_segmented_anonymized").rglob("*.wav"):
        base = ODD_PREFIX.sub("", p.stem)
        m = re.match(r"^(d\d+(?:_\d+)?)_(T(?:LI_)?\d+)$", base)
        if m:
            index[f"{m.group(1)}-{m.group(2)}"] = p
    return index


def convert(job: tuple[str, str, str]) -> tuple[str, float, str]:
    """Read, mono-mix, resample, write int16. Returns (utt_id, seconds, status)."""
    utt_id, src, dst = job
    audio, sr = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                      # mono (all files are, but be safe)
    seconds = len(audio) / sr
    if seconds < MIN_SEC:
        return utt_id, seconds, "too_short"
    if seconds > MAX_SEC:
        return utt_id, seconds, "too_long"
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(audio, -1, 1), TARGET_SR, subtype="PCM_16")
    return utt_id, len(audio) / TARGET_SR, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("utterances")
    ap.add_argument("audio_root")
    ap.add_argument("out")
    ap.add_argument("--docs", nargs="*", help="restrict to these doc ids (local dev)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    rows = [json.loads(l) for l in open(args.utterances, encoding="utf-8")]
    rows = [r for r in rows if r["split"]]
    if args.docs:
        rows = [r for r in rows if r["doc"] in set(args.docs)]
    index = wav_index(Path(args.audio_root))
    print(f"{len(rows)} split-assigned utterances, {len(index)} wav files indexed")

    jobs, status = [], Counter()
    by_id = {r["utt_id"]: r for r in rows}
    for r in rows:
        src = index.get(r["audio_pointer"])
        if src is None:
            status[(r["split"], "no_wav")] += 1
            continue
        jobs.append((r["utt_id"], str(src), str(out / "wav16" / r["split"] / f"{r['utt_id']}.wav")))

    results = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (utt_id, seconds, st) in enumerate(ex.map(convert, jobs, chunksize=64), 1):
            results[utt_id] = (seconds, st)
            status[(by_id[utt_id]["split"], st)] += 1
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    keep = ("utt_id", "doc", "speaker", "region", "transcriber", "tool", "text",
            "normalised", "n_words", "split", "eval_ok")
    hours, counts = defaultdict(float), Counter()
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: open(out / f"manifest_{s}.jsonl", "w", encoding="utf-8") for s in ("train", "dev", "test")}
    for r in rows:
        if r["utt_id"] not in results or results[r["utt_id"]][1] != "ok":
            continue
        seconds = results[r["utt_id"]][0]
        rec = {k: r[k] for k in keep}
        rec["path"] = str(Path("wav16") / r["split"] / f"{r['utt_id']}.wav")
        rec["seconds"] = round(seconds, 3)
        writers[r["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
        hours[r["split"]] += seconds / 3600
        counts[r["split"]] += 1
    for w in writers.values():
        w.close()

    summary = {
        "kept": {s: {"utterances": counts[s], "hours": round(hours[s], 2)} for s in counts},
        "dropped": {f"{s}/{st}": n for (s, st), n in sorted(status.items()) if st != "ok"},
        "min_sec": MIN_SEC, "max_sec": MAX_SEC, "sample_rate": TARGET_SR,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
