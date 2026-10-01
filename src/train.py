"""Phase 5: LoRA fine-tune of Whisper on Dieth transcripts.

A plain PyTorch loop rather than Seq2SeqTrainer, on purpose: every step
(batching, loss, eval, checkpoint) is visible in ~200 lines, and nothing
depends on Trainer internals that change between transformers versions.

What is trained
  The base model is frozen. LoRA adds a low-rank update B·A (rank r) next to
  each attention projection (q, k, v, out). Decoder rank is `--r` (default
  32, alpha 2r): the decoder is what has to learn to *write* Dieth. The
  encoder already hears German well, so it gets a smaller rank
  (`--encoder-r`, default 8; 0 = encoder untouched). Brief: "if you're
  rank-constraining anywhere, constrain the encoder first".

Targets
  labels = <|de|><|transcribe|><|notimestamps|> text <|endoftext|>, padding
  masked with -100. Whisper builds the decoder input by shifting labels
  right and prepending <|startoftranscript|>, so that token is stripped
  from the labels here. The language token stays "de": Whisper has no Swiss
  German token, and we are deliberately re-teaching what "de + transcribe"
  means for this adapter.

Logging (the brief's priority order)
  1. every --eval-every steps: 10 fixed dev utterances, reference next to
     hypothesis, appended to <out>/samples/step_XXXXXX.txt
  2. CER / WER / FlexWER on a fixed dev subsample -> TensorBoard + metrics.jsonl
  3. train loss -> TensorBoard

Checkpointing
  <out>/last/ is overwritten every --save-every steps (adapter + optimizer +
  scheduler + step + RNG state, written to a temp dir then renamed so a kill
  mid-write cannot corrupt it). Restarting the same command resumes from it.
  <out>/best/ holds the adapter with the lowest dev CER.

Not used: gradient checkpointing (brief lists it). With a frozen base it
needs a hook to keep gradients flowing through the checkpointed encoder and
it slows steps ~30%. Without it, large-v3 at batch 16 does NOT fit in an A40
(46 GB, measured); batch 8 x accumulation 4 does, and gives the same
effective batch, so that is the default in scripts/train.slurm.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluate import transcribe
from metrics import all_metrics, load_table
from normalise import normalise_hyp

ATTN = r"(q_proj|k_proj|v_proj|out_proj)"


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, data_root: str):
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        self.root = Path(data_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio, sr = sf.read(str(self.root / r["path"]), dtype="float32")
        assert sr == 16_000
        return audio, r["text"]


class Collator:
    def __init__(self, processor: WhisperProcessor):
        self.p = processor
        self.sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, batch):
        audio, text = zip(*batch)
        feats = self.p.feature_extractor(list(audio), sampling_rate=16_000, return_tensors="pt").input_features
        tok = self.p.tokenizer(list(text), return_tensors="pt", padding=True)
        labels = tok.input_ids.masked_fill(tok.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.sot).all():      # model prepends it when shifting right
            labels = labels[:, 1:]
        return feats, labels


def lora_config(r: int, encoder_r: int, dropout: float) -> LoraConfig:
    if encoder_r > 0:
        return LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=dropout, bias="none",
            target_modules=rf".*\.{ATTN}",
            rank_pattern={rf"model\.encoder\..*\.{ATTN}": encoder_r},
            alpha_pattern={rf"model\.encoder\..*\.{ATTN}": 2 * encoder_r},
        )
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, bias="none",
                      target_modules=rf".*decoder.*\.{ATTN}")


def save_state(out: Path, name: str, model, opt, sched, step: int, best_cer: float) -> None:
    tmp = out / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)                                  # adapter weights only (~tens of MB)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                "best_cer": best_cer, "torch_rng": torch.get_rng_state(),
                "py_rng": random.getstate()}, tmp / "trainer_state.pt")
    final = out / name
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)


def evaluate(model, processor, rows, data_root, table, device, dtype, out: Path, step: int, n_samples: int):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        hyps_raw = transcribe(rows, data_root, model, processor, device, dtype, batch_size=16)
    model.train()
    refs = [r["text"] for r in rows]
    hyps = [normalise_hyp(h) for h in hyps_raw]
    m = all_metrics(refs, hyps, table)
    (out / "samples").mkdir(exist_ok=True)
    with open(out / "samples" / f"step_{step:06d}.txt", "w", encoding="utf-8") as f:
        f.write(f"step {step}  cer {m['cer']:.3f}  wer {m['wer']:.3f}  flexwer {m['flexwer']:.3f}\n\n")
        for r, raw in list(zip(rows, hyps_raw))[:n_samples]:   # rows are fixed -> same 10 every time
            f.write(f"[{r['utt_id']} {r['region']}]\nREF: {r['text']}\nHYP: {raw.strip()}\n\n")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--encoder-r", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--dev-size", type=int, default=200)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    processor = WhisperProcessor.from_pretrained(args.model, language="german", task="transcribe")
    # Explicit float32: large-v3 is *stored* in fp16 and transformers 5 loads the
    # stored dtype by default. fp16 master weights + AdamW is numerically fragile;
    # we keep fp32 weights and get the speed from bf16 autocast instead.
    base = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    resume = (out / "last" / "trainer_state.pt").exists()
    if resume:
        model = PeftModel.from_pretrained(base, out / "last", is_trainable=True)
    else:
        model = get_peft_model(base, lora_config(args.r, args.encoder_r, args.dropout))
    model.to(device)
    model.print_trainable_parameters()

    train = ManifestDataset(args.train_manifest, args.data_root)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, collate_fn=Collator(processor),
                        persistent_workers=args.workers > 0)
    dev_rows = [json.loads(l) for l in open(args.dev_manifest, encoding="utf-8")]
    random.Random(0).shuffle(dev_rows)                           # fixed subsample across runs/resumes
    dev_rows = dev_rows[:args.dev_size]
    table = load_table(args.variants)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) *
                                              max(0.0, (args.max_steps - s) / max(1, args.max_steps)))
    step, best_cer = 0, math.inf
    if resume:
        st = torch.load(out / "last" / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best_cer = st["step"], st["best_cer"]
        torch.set_rng_state(st["torch_rng"]); random.setstate(st["py_rng"])
        print(f"resumed from step {step} (best dev CER {best_cer:.3f})")

    tb = SummaryWriter(out / "tb")
    hours = sum(r["seconds"] for r in train.rows) / 3600
    print(f"train {len(train)} utts / {hours:.1f} h, dev subsample {len(dev_rows)}, device {device}, "
          f"effective batch {args.batch_size * args.accum}, steps {step}->{args.max_steps}")

    if step == 0:                                                # step-0 samples = the zero-shot floor
        m = evaluate(model, processor, dev_rows, args.data_root, table, device, torch.float32, out, 0, args.samples)
        print(f"step 0  dev {m}")

    model.train()
    t0, start_step, running, micro = time.time(), step, 0.0, 0
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for feats, labels in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(input_features=feats.to(device), labels=labels.to(device)).loss
            (loss / args.accum).backward()
            running += loss.item(); micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                avg = running / (10 * args.accum); running = 0.0
                tb.add_scalar("train/loss", avg, step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                print(f"step {step}  loss {avg:.3f}  lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / (step - start_step):.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                m = evaluate(model, processor, dev_rows, args.data_root, table, device, torch.float32, out, step, args.samples)
                for k, v in m.items():
                    tb.add_scalar(f"dev/{k}", v, step)
                with open(out / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, **m}) + "\n")
                print(f"step {step}  dev {m}", flush=True)
                if m["cer"] < best_cer:
                    best_cer = m["cer"]
                    save_state(out, "best", model, opt, sched, step, best_cer)
            if step % args.save_every == 0 or step == args.max_steps:
                save_state(out, "last", model, opt, sched, step, best_cer)
            if step >= args.max_steps:
                break
    tb.close()
    print(f"done. best dev CER {best_cer:.3f}; adapter in {out / 'best'}")


if __name__ == "__main__":
    main()
