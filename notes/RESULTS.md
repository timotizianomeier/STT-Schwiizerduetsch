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

| | CER | WER | FlexWER | n |
|---|---|---|---|---|
| **all** | **49.2 %** | 90.1 % | 57.0 % | 4,243 |
| ZH (1225) | 44.7 % | 84.0 % | 49.9 % | 850 |
| BS (1263) | 45.1 % | 87.8 % | 56.8 % | 1,377 |
| BE (1121) | 50.2 % | 91.6 % | 56.8 % | 996 |
| LU (1195) | 56.6 % | 96.3 % | 63.5 % | 1,020 |

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

## Phase 5/6 — LoRA fine-tune

Pending.
