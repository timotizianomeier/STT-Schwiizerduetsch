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

## Phase 5/6 — full runs

Two 4,000-step runs exist (≈2.2 epochs, A40, batch 8 × 4, LoRA decoder r=32
/ encoder r=8, peak LR 2e-4). They differ only in the training targets:

- `v1_accents` (job 294741, 2026-10-01): Dieth accents kept. Unintended
  run, see DECISIONS 2026-10-02. Best dev checkpoint at step 3000.
- `v2_folded` (job 294936, 2026-10-02): accents folded, the approved policy.

All test numbers below are scored the same way: references and hypotheses
both folded (policy v2), so the two models are comparable.

### Test set (4,243 utterances, 3.83 h)

| model | CER | WER | FlexWER |
|---|---|---|---|
| zero-shot large-v3 | 48.7 % | 89.3 % | 56.7 % |
| `v1_accents` best | **14.6 %** | 44.5 % | 24.5 % |
| `v2_folded` best (step 3500) | **14.5 %** | 43.9 % | 24.0 % |

The two fine-tuned models are indistinguishable (0.1 CER points): folding
the accents cost nothing. `v2_folded` is the final model.

`v1_accents` per region:

| | CER | WER | FlexWER | n |
|---|---|---|---|---|
| ZH (1225) | 11.1 % | 37.6 % | 18.4 % | 850 |
| BE (1121) | 13.7 % | 44.2 % | 22.3 % | 996 |
| BS (1263) | 14.7 % | 42.7 % | 27.2 % | 1,377 |
| LU (1195) | 18.2 % | 52.5 % | 29.1 % | 1,020 |

Reading it: CER falls from 48.7 % to 14.6 % on speakers the model never
heard; WER 44.5 % is well below the ~60 % of the 2020 Kaldi system (on an
easier, non-speaker-disjoint split). Zürich is best, Luzern worst, the same
ordering as zero-shot. No empty outputs. 50 examples:
`notes/v1_accents_test_examples.md` (local only).

### Dev subsample during `v2_folded` training (200 utterances, v2 targets)

| step | 0 | 250 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|---|
| CER | 50.7 % | 18.9 % | 17.3 % | 16.0 % | 16.1 % | 15.9 % |
| WER | 89.5 % | 52.1 % | 46.4 % | 45.2 % | 45.6 % | 44.7 % |

Most of the gain arrives in the first 250 steps (a quarter of an epoch);
after step 1000 the curve is nearly flat. Full curve (CER, %): 250: 18.9,
1000: 16.0, 2000: 15.9, 3000: 15.6, **3500: 15.2** (best), 4000: 15.8.

### `v2_folded` on test, in detail (eval job 294938, RTF 0.12, 0 empty outputs)

| | CER | WER | FlexWER | n |
|---|---|---|---|---|
| all | **14.5 %** | 43.9 % | 24.0 % | 4,243 |
| ZH (1225) | 11.2 % | 38.2 % | 17.4 % | 850 |
| BE (1121) | 13.6 % | 43.4 % | 21.8 % | 996 |
| BS (1263) | 14.4 % | 41.0 % | 26.7 % | 1,377 |
| LU (1195) | 18.1 % | 52.0 % | 29.1 % | 1,020 |

One speaker per region, so region and speaker are confounded (per-speaker
and per-region tables are identical).

Per-utterance CER distribution: median 12.0 %, mean 15.0 %; 438 utterances
(10 %) exact matches; 10.6 % above 30 % CER; 2.0 % above 50 %; 6 (0.1 %)
above 100 %. The overall figure is carried by many small errors, not a few
disasters.

50 random test utterances (seed 0): `notes/v2_folded_test_examples.md`
(local only). Read on 2026-11-05: the output reads as written Swiss German
throughout.

### Failure analysis (`v2_folded`, all 4,243 test utterances)

1. **Collapse into Standard German: rare and word-level.** No full
   Standard German sentences among the worst utterances. Function words
   such as *ich*, *mit*, *sind* are also Swiss German and match the
   references. The one clear shift is *wenn* (4 in references as `wenn`,
   34 in output; references mostly write *wen*/*wänn*). Many of those are
   legitimate variants; a few are the model substituting the Standard
   German conjunction for a word it did not catch ("da het me" → "wenn
   me"). Smaller shifts: *doch* 30 → 50, *mich* 12 → 22, *wir* 0 → 4.
2. **Hallucination on very short clips.** The six utterances above 100 %
   CER have 1–12 character references and the model adds words ("jä" →
   "jä sii wi si"; "mietwonig" → "wie wie wie d woonig").
3. **Fluent but wrong on hard audio.** Worst long utterances are mostly
   Luzern and Basel; the model writes plausible dialect that does not match
   ("isch e schwären aschmatigger gsi" → "süsch wäär en aschmatiker gseend
   früer"), and sometimes drops most of an utterance.
4. **Vowel length and variant spelling dominate the small errors**
   ("schöön hööch" vs "schön höch", *mer*/*me*, *lüüt*/*lüt*). Largely
   annotator inconsistency, which is why CER overstates the real error.
5. **Stripped fillers ("e", "ä").** Pure deletion of the reference token
   "e" is 6 % (22 of 363) against 1.2 % for words overall; the model
   emits 352 "e" vs 363 in the references. "ä" is under-produced by about
   half (40 vs 76) and mostly written as "e". "ä" is 0.2 % of test tokens,
   so the effect on CER is negligible. Alignment is word-level `difflib`,
   so "replaced" counts are inflated by mis-heard phrases.
