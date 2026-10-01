"""Transcribe a manifest with Whisper (optionally + LoRA adapter) and score it.

Used for Phase 4 (zero-shot baseline) and Phase 6 (fine-tuned model): same
code path, so the two numbers are comparable by construction.

Outputs in --out:
  predictions.jsonl   one row per utterance: ref, raw hyp, normalised hyp
  metrics.json        CER / WER / FlexWER overall, per speaker, per region
  examples.md         side-by-side reference vs hypothesis, for reading

Decoding choices (kept boring on purpose):
  * language forced to German, task=transcribe. Whisper has no Swiss German
    language token; "de" is what every Swiss German system uses. Zero-shot it
    will emit Standard German — that is the expected floor (brief, Phase 4).
  * greedy decoding, no timestamps, no previous-text conditioning: each
    utterance is independent and short.
  * utterances are sorted by duration before batching so a batch pads to
    similar lengths (Whisper pads to 30 s anyway, but generation length is
    what costs time).

Run (local smoke test, tiny model, CPU):
    python src/evaluate.py --manifest data/processed/dev_local/manifest_test.jsonl \
        --data-root data/processed/dev_local --model openai/whisper-tiny \
        --variants data/processed/dieth_to_norm.json --out runs/smoke --limit 24
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from metrics import all_metrics, load_table
from normalise import normalise_hyp


def load_rows(manifest: str, limit: int | None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    rows = [r for r in rows if r.get("eval_ok", True)]
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)      # fixed subsample, same every run
        rows = rows[:limit]
    return sorted(rows, key=lambda r: r["seconds"])


@torch.inference_mode()
def transcribe(rows, data_root, model, processor, device, dtype, batch_size, max_new_tokens=128):
    hyps = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        audio = [sf.read(str(Path(data_root) / r["path"]), dtype="float32")[0] for r in batch]
        enc = processor(audio, sampling_rate=16_000, return_tensors="pt", return_attention_mask=True)
        ids = model.generate(enc.input_features.to(device, dtype),
                             attention_mask=enc.attention_mask.to(device),
                             language="german", task="transcribe",
                             max_new_tokens=max_new_tokens, num_beams=1)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return hyps


def grouped(rows, refs, hyps, key, table):
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r[key]].append(i)
    return {k: {**all_metrics([refs[i] for i in v], [hyps[i] for i in v], table), "n": len(v)}
            for k, v in sorted(idx.items())}


def write_examples(path, rows, refs, hyps_raw, hyps, n, title, seed=0):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{n} random utterances (seed {seed}). "
                "REF = Dieth reference, HYP = model output as emitted, "
                "NORM = model output as scored.\n\n")
        for i in order[:n]:
            r = rows[i]
            f.write(f"**{r['utt_id']}** · {r['region']} · {r['seconds']:.1f}s\n\n"
                    f"- REF:  {refs[i]}\n- HYP:  {hyps_raw[i].strip()}\n- NORM: {hyps[i]}\n"
                    f"- (Standard German gloss: {r['normalised']})\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variants", required=True, help="dieth_to_norm.json for FlexWER")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA directory (Phase 6)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--examples", type=int, default=30)
    ap.add_argument("--title", default="Examples")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(device).eval()

    rows = load_rows(args.manifest, args.limit)
    hours = sum(r["seconds"] for r in rows) / 3600
    print(f"{len(rows)} utterances, {hours:.2f} h, model {args.model}, device {device}")
    t0 = time.time()
    hyps_raw = transcribe(rows, args.data_root, model, processor, device, dtype, args.batch_size)
    elapsed = time.time() - t0

    refs = [r["text"] for r in rows]
    hyps = [normalise_hyp(h) for h in hyps_raw]
    table = load_table(args.variants)
    metrics = {
        "model": args.model, "adapter": args.adapter, "manifest": args.manifest,
        "utterances": len(rows), "hours": round(hours, 3),
        "decode_seconds": round(elapsed, 1), "rtf": round(elapsed / (hours * 3600), 4),
        "empty_hypotheses": sum(1 for h in hyps if not h),
        "overall": all_metrics(refs, hyps, table),
        "per_speaker": grouped(rows, refs, hyps, "speaker", table),
        "per_region": grouped(rows, refs, hyps, "region", table),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r, raw, h in zip(rows, hyps_raw, hyps):
            f.write(json.dumps({"utt_id": r["utt_id"], "ref": r["text"], "hyp_raw": raw, "hyp": h,
                                "speaker": r["speaker"], "region": r["region"]}, ensure_ascii=False) + "\n")
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, ensure_ascii=False)
    write_examples(out / "examples.md", rows, refs, hyps_raw, hyps, args.examples, args.title)
    print(json.dumps({k: metrics[k] for k in ("utterances", "hours", "rtf", "empty_hypotheses", "overall", "per_region")}, indent=2))


if __name__ == "__main__":
    main()
