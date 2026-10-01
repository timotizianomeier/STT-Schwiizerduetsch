# Results

Test set: 4 held-out documents (1225 ZH, 1121 BE, 1195 LU, 1263 BS), 4,243
interviewee utterances, 3.83 h. No speaker, recording or document overlap
with train. References are normalised Dieth; hypotheses are scored after
`normalise_hyp` (lowercase, ß→ss, punctuation and digits stripped).

CER is the primary metric. WER is reported for comparability only: 94 % of
tokens have more than one attested Dieth spelling. FlexWER accepts any
attested spelling of the same word.

## Phase 4 — zero-shot baseline (2026-10-01)

`openai/whisper-large-v3`, language forced to German, greedy decoding.
Job 294727, A16, real-time factor 0.11.

Scored against the v2 references (accents folded, 2026-10-01; re-scored
from the saved outputs with `src/rescore.py`). Against the original v1
references the overall figures were 49.2 / 90.1 / 57.0.

| | CER | WER | FlexWER | n |
|---|---|---|---|---|
| **all** | **48.7 %** | 89.3 % | 56.7 % | 4,243 |
| ZH (1225) | 44.7 % | 84.0 % | 49.8 % | 850 |
| BS (1263) | 45.1 % | 87.8 % | 56.7 % | 1,377 |
| BE (1121) | 50.2 % | 91.6 % | 56.7 % | 996 |
| LU (1195) | 54.6 % | 93.4 % | 62.8 % | 1,020 |

Reading it:

- The model emits **Standard German**, as expected: "han ich gsait guet aso
  jez" → "Da habe ich gesagt, gut, das ist...". It is translating, often
  correctly in meaning, and that is why WER is ~90 % while the content is
  frequently recoverable. This is the floor the brief asked for and the
  evidence that the task is distinct from Standard-German-target ASR.
- **FlexWER (57 %) is far below WER (90 %)** even here, because common
  function words coincide with an attested Dieth spelling (*und*, *ich*,
  *das*, *in*). So FlexWER is lenient; it must be read next to CER, not
  instead of it.
- Zürich is the easiest region and Luzern the hardest, by 12 CER points.
  One speaker per region, so this is as much speaker as dialect.
- 3 empty outputs out of 4,243; hallucination on very short clips shows up
  as fluent but unrelated German ("suppe ggä" → "Super gern.").
- Prior reference point: Nigmatulina & Kew (VarDial 2020), Kaldi, Dieth
  target, roughly 60 % WER on their (not speaker-disjoint) split. Not
  directly comparable; ours is stricter.

30 side-by-side examples: `notes/baseline_examples.md` (local only,
gitignored: the corpus licence forbids redistribution and the repo is
public).

## Phase 5 — smoke job (2026-10-01, job 294732; v1 targets, accents kept)

300 optimizer steps, A40, batch 8 × accumulation 4 (effective 32), LoRA
decoder r=32 / encoder r=8 on q/k/v/out projections (23.6 M trainable
parameters, 1.5 % of the model), bf16 autocast, 5 s/step, 31 minutes.
300 steps × 32 = 9,600 utterances, i.e. **16 % of one epoch**. Peak learning
rate only 7.5e-5 because warm-up (200) and decay (300) overlap in such a
short schedule.

Fixed 200-utterance dev subsample (docs 1055 ZH, 1142 BE, 1261 LU):

| step | train loss | CER | WER | FlexWER |
|---|---|---|---|---|
| 0 (zero-shot) | — | 52.2 % | 92.5 % | 57.1 % |
| 100 | 1.37 | 35.5 % | 79.8 % | 50.8 % |
| 200 | 1.13 | 23.5 % | 63.2 % | 34.5 % |
| 300 | 1.07 | **21.6 %** | 61.1 % | 33.1 % |

What the samples show (same ten utterances at every step,
`notes/smoke_examples.md`, local only):

- By step 100 the output has switched register completely: lowercase, no
  punctuation, dialect vowels and endings ("het", "nöd", "öis", "ghaa",
  "jaar"). It no longer reads as Standard German.
- By step 200–300 several utterances are exact or one character off
  ("ä bi öis guet aber"; "zwänzg jaar rehabilitiert").
- Remaining errors are of three kinds: spelling variants that are arguably
  fine ("ez" for "iz", "häig" for "haig"); diacritics (graves mostly
  dropped: "und" for "ùnd"); and genuine mishearings that also fooled the
  zero-shot model ("marschinerweise" for "waarschiinlech").
- One Standard German / English leak survives: "die things" for "de dings".

Not a result yet: 200 dev utterances, 16 % of an epoch, and dev not test.
But it answers the brief's question in the affirmative direction: a
LoRA-tuned Whisper does produce text that reads like written Swiss German,
and after 31 minutes it is already at WER ≈ 61 %, the level of the 2020
Kaldi system.

## Phase 5/6 — full run

Pending Timo's go-ahead (brief: show smoke samples before anything long).
Proposed: `RUN=v1 MAX_STEPS=4000 sbatch scripts/train.slurm` — about
2.2 epochs, ~6.5 h on an A40 including evals, peak LR 2e-4.

