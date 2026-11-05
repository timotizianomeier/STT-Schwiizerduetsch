"""Transcribe audio with the fine-tuned Whisper (base + LoRA adapter).

The inference core behind the local app (src/app.py), and usable on its own:

    uv run python src/transcribe.py clip.wav
    uv run python src/transcribe.py clip.wav --adapter runs/v2_folded_final

Decoding is deliberately identical to src/evaluate.py, so what you hear here is
what was scored: language forced to German (Whisper has no Swiss German token),
task=transcribe, greedy decoding, LoRA merged into fp16 weights.

Output is the model's own text. It was trained on lowercase, punctuation-free
Dieth (policy v2: accents folded, umlauts kept), so it already comes out in
that style; we do NOT run it through normalise_hyp, which exists for scoring.

Whisper sees at most 30 s at a time. Longer audio is cut into 30 s pieces at
the quietest point near the boundary, which is crude (no overlap, no context
carried over) but keeps words from being sliced in half most of the time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL = "openai/whisper-large-v3"
DEFAULT_ADAPTER = Path(__file__).resolve().parent.parent / "runs" / "v2_folded_final"
SR = 16_000
WINDOW = 30 * SR            # Whisper's fixed input length
SEARCH = 4 * SR             # look this far back from the 30 s mark for a pause
MIN_SECONDS = 0.3           # below this there is nothing to transcribe


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Transcriber:
    def __init__(self, adapter: str | Path = DEFAULT_ADAPTER, base: str = BASE_MODEL,
                 device: str | None = None):
        self.device = device or pick_device()
        # fp16 on GPU/MPS (as in the scored cluster runs); fp32 on CPU, where fp16 is slow.
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.processor = WhisperProcessor.from_pretrained(base)
        model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=self.dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        self.model = model.to(self.device).eval()

    @staticmethod
    def load_audio(path: str) -> np.ndarray:
        """Any format ffmpeg/soundfile can read -> 16 kHz mono float32."""
        audio, _ = librosa.load(path, sr=SR, mono=True)
        return audio

    @staticmethod
    def prepare(audio: np.ndarray, sr: int) -> np.ndarray:
        """Raw array (e.g. from a browser microphone) -> 16 kHz mono float32."""
        audio = np.asarray(audio)
        if audio.dtype.kind in "iu":                       # int16 etc. -> [-1, 1]
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return librosa.resample(audio, orig_sr=sr, target_sr=SR) if sr != SR else audio

    @staticmethod
    def split(audio: np.ndarray) -> list[np.ndarray]:
        """Cut at the quietest 50 ms frame in the last 4 s before each 30 s mark."""
        pieces, start = [], 0
        while len(audio) - start > WINDOW:
            lo, hi = start + WINDOW - SEARCH, start + WINDOW
            frame = SR // 20
            seg = np.abs(audio[lo:hi])[: (hi - lo) // frame * frame].reshape(-1, frame).mean(axis=1)
            cut = lo + int(seg.argmin()) * frame + frame // 2
            pieces.append(audio[start:cut])
            start = cut
        pieces.append(audio[start:])
        return pieces

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128) -> str:
        """audio: 16 kHz mono float32. Returns the model's text."""
        if len(audio) < MIN_SECONDS * SR:
            return ""
        texts = []
        for piece in self.split(audio):
            enc = self.processor(piece, sampling_rate=SR, return_tensors="pt",
                                 return_attention_mask=True)
            ids = self.model.generate(
                enc.input_features.to(self.device, self.dtype),
                attention_mask=enc.attention_mask.to(self.device),
                language="german", task="transcribe",
                max_new_tokens=max_new_tokens, num_beams=1)
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return " ".join(t for t in texts if t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="audio file(s): wav, mp3, m4a, ...")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                    help="LoRA adapter directory, or 'none' for the zero-shot base model")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tr = Transcriber(None if args.adapter == "none" else args.adapter, device=args.device)
    print(f"loaded on {tr.device} in {time.time() - t0:.0f}s")
    for path in args.audio:
        audio = tr.load_audio(path)
        t0 = time.time()
        text = tr.transcribe(audio)
        dt = time.time() - t0
        print(f"\n{path}  ({len(audio) / SR:.1f}s audio, {dt:.1f}s, RTF {dt / (len(audio) / SR):.2f})\n{text}")


if __name__ == "__main__":
    main()
