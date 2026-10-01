# WALKTHROUGH.md — what every file does and why

Written so the author can explain the repo without notes, and can pick the project up
after weeks away. Updated at the end of each phase. "Core" means the file embodies a
modelling or data choice that changes the result; "plumbing" means it makes the pipeline
reproducible but contains no judgement. Phase numbers follow `notes/BRIEF.md`.

---

## Phase 0 — scaffold and environment

### How the pieces fit (as of Phase 2; later stages are the plan, not code yet)

```
data/raw/archimob_r2_text/Archimob_Release_2/*.xml      (TEI XML, 52 files, 43 interviews)
                     │
                     ▼
        src/parse_archimob.py  ──uses──►  src/normalise.py   (Phase 2 policy)
                     │
                     ▼
data/processed/archimob_utterances.jsonl   (one row per utterance, flags, usable yes/no)
                     │
        src/verify_against_kew.py ◄── third_party/two-headed-master/data/archimob.csv
                     │                 (one-off cross-check, done)
                     ▼
            [Phase 3] src/prepare.py  + audio chunks  ──►  HF DatasetDict on /vol/bitbucket
                     ▼
            [Phase 4] zero-shot Whisper baseline      ──►  notes/baseline_examples.md
                     ▼
            [Phase 5] src/train.py (LoRA)             ──►  checkpoints + sample dumps
                     ▼
            [Phase 6] src/evaluate.py (CER/WER/FlexWER) ──►  notes/RESULTS.md
```

### The goal in three sentences

Transcribe spoken Swiss German into **written Swiss German** (Dieth orthography), not into
Standard German. Every public Swiss German ASR system does the latter, which is really speech
translation; the only prior dialect-to-dialect result is Nigmatulina & Kew (VarDial 2020) at
roughly 60 % WER with Kaldi. v1 asks one question: does a LoRA-fine-tuned Whisper produce
output that *reads* like Swiss German? CER is the proxy, Timo's ear is the judge.

### Conventions and documentation (plumbing, but read them first)

| File | What it is |
|---|---|
| `notes/BRIEF.md` | Timo's original brief, verbatim. Phases 0–6, working rules, definition of done. Never edited; deviations go in DECISIONS.md. |
| `notes/DECISIONS.md` | One dated line per non-obvious choice with the reason. Also records where reality contradicted the brief (three times so far: Zenodo DOI is Release 1; audio needs a contract, not a direct download; Kew's CSV predates the 2019 corrections). |
| `notes/DATA.md` | What the corpus actually contains: provenance, XML structure, element inventory, character table summary, the ambiguity number, the v1 filter counts, blockers. |
| `notes/char_table.txt` | Full character frequency table of the Dieth layer (Phase 2 deliverable, shown to Timo). |
| `README.md` | Public face: purpose, layout, data citation (required by the LaRS contract), prior work, setup. |
| `WALKTHROUGH.md` | This file. |
| `.gitignore` | Keeps corpus data, audio, checkpoints, `third_party/`, `.venv/` out of git. The contract forbids redistributing the data in any form, so this matters. |

### Project files (plumbing)

**`pyproject.toml`** — the `uv` project. Dependencies and what each is for:
- `torch`, `transformers` — Whisper model and tokenizer.
- `peft` — LoRA adapters (train a few million parameters on top of a frozen 1.5 B model).
- `accelerate` — device placement and mixed precision for the training loop.
- `bitsandbytes` — 8-bit optimizer / quantised loading if VRAM is tight. Linux-only wheels,
  hence the `sys_platform == 'linux'` marker: the Mac env is for data work only.
- `datasets`, `librosa`, `soundfile` — audio loading and resampling to 16 kHz.
- `jiwer`, `evaluate` — WER/CER.
- `tensorboard` — training curves.
- dev: `pytest`.

**`uv.lock`** — pinned versions so the cluster environment equals the laptop one.

### Cluster facts (discovered 2026-08-25, confirmed active 2026-09-29)

Imperial DoC SLURM cluster, `gpucluster2.doc.ic.ac.uk`, reached through `shell4` (the
`gpucluster2` alias in `~/.ssh/config` does the jump). Partitions a40/a30/t4/a16/a100/long;
3-day limit except `long`. Home quota is nearly full, so **everything** (data, HF cache,
checkpoints) lives under `/vol/bitbucket/ttm25/stt/`.

---

## Phase 1 — data acquisition

### What the corpus actually is (this differs from the brief)

**ArchiMob Release 2 (2019)**, 43 oral-history interviews with people born 1905–1932, in
14 dialect regions (ZH dominates with 12). Hosted on SwissUbase / LaRS, **not** Zenodo (the
brief's DOI is Release 1). Two datasets, each behind a per-user research contract:

| Dataset | SwissUbase ref | Contents | Status |
|---|---|---|---|
| Transcriptions | 2269 | `Archimob_Release_2.zip` (TEI XML) + guidelines PDFs + `Metadata.txt` | downloaded, unpacked to `data/raw/archimob_r2_text/` |
| Audio | 2277 | `archimob_r2_audio_share.zip`, 19.43 GB | contract submitted 2026-09-29, pending |

The contract (research/teaching only, no redistribution in original or edited form, delete at
project end, 3-month term) is stricter than the brief's "CC BY-NC-SA, direct download". See
DECISIONS 2026-09-29 for the consequences.

**Format**: TEI XML, one file per recording (52 files, some interviews split into parts).
Each `<u>` (utterance) has a `who` (speaker) and a `start` media pointer such as
`d1007-T31` that names the audio chunk. Timestamps are *not* in the XML; the pointer file is
not shipped, so segment durations wait for the audio. Each word is
`<w normalised="können" tag="VMFIN">chönd</w>`: the Dieth spelling is the element text, the
Standard-German-like form is the `normalised` attribute, and a POS tag comes free. Both
layers exist for every one of the 581,974 words.

### `third_party/` — prior work (gitignored clones, reference only)

- `Kaldi-for-ASR-of-Swiss-German` (Nigmatulina): the Kaldi recipe from the VarDial paper.
  Read `archimob/process_exmaralda_xml.py` to learn their XML→CSV mapping; nothing executed.
- `two-headed-master` (Kew): same lineage, plus `data/archimob.csv` (the whole text layer),
  `data/flexwer_mapping.json` (the VarDial variant table, reused in Phase 6) and their
  splits (not reused: not speaker-disjoint).

---

## Phase 2 — orthography (the hard part)

### The idea in four sentences

Dieth spelling is a *convention* for writing dialect phonetically, not a standard, so the same
word is spelled many ways across (and within) transcribers. We measured it: 25.8 % of word types
and **94 % of running tokens** have more than one attested spelling (*haben* has 147). Word
error rate therefore punishes correct output for not matching one arbitrary spelling, so
**CER is primary** and a variant-tolerant FlexWER is secondary. The normalisation policy keeps
every diacritic that carries phonetic meaning and removes only annotation artefacts.

### `src/analyse_text.py` (core, historical)

The Phase 2 reconnaissance, run on Kew's CSV before the XML was available. Produces the
character table, the flag counts, and the ambiguity estimate. Its operational definition of
ambiguity is "number of distinct Dieth spellings per normalised form", the same one FlexWER
uses. Keep it; it documents where the 94 % number comes from.

### `src/normalise.py` (core; read line by line)

Two functions, and a character whitelist.

- `normalise_dieth(text)` applies the signed-off policy to one string, in this order:
  1. strip Kaldi meta tokens (`<SPOKEN_NOISE>` etc.) — must happen *before* lowercasing,
     which is the bug the tests caught on the first run;
  2. Unicode **NFC**: the corpus has 759 decomposed combining graves (`u` + U+0300); NFC
     composes them to `ù`. Exception worth knowing: there is no precomposed "ö with grave",
     so `ö̀` stays two codepoints. Harmless for a byte-level tokenizer, but CER counts it as
     two characters;
  3. drop the single stray combining acute;
  4. lowercase, strip the five stray parentheses, collapse whitespace.
- `ALLOWED` is the set of characters the policy accounts for: a–z, umlauts, the Dieth graves
  `à ì è ò ù ǜ` (open vowel), tildes `õ ã ẽ ĩ` (nasal), the rare acutes. `unexpected_chars`
  reports anything else so a new character can never silently become a training target.
- `usable(row)` decides whether an utterance is a valid target from its structural flags,
  in a fixed order so drop reasons can be tallied: anonymised (`***` masks) → overlap (shares
  its audio chunk with another utterance) → gap (unintelligible speech with no text) → empty.

### `src/parse_archimob.py` (core)

Turns the TEI XML into one JSON line per utterance. Why not just use Kew's CSV, which we
verified? Because the CSV flattens the annotation elements into three Kaldi tokens and
loses what we need for target selection. The parser counts each element type per utterance:

| element | count | what it is | v1 treatment |
|---|---|---|---|
| `<w>` | 581,974 | a word, Dieth + normalised | kept |
| `<pause/>` | 10,796 | silence | stripped |
| `<vocal><desc>` | 12,928 | eh / ää / [lacht] | stripped (model learns to omit fillers) |
| `<del type="truncation">` | 8,836 | fragment like `zw/` | stripped |
| `<unclear>` | 3,348 | transcriber unsure; wraps words | words kept |
| `<gap reason="unintelligible"/>` | 1,549 | speech with **no** text | utterance dropped |
| `<incident>` etc. | 416 | non-speech events | stripped |

Also derived per utterance: speaker (`interviewer`, `otherPerson`, or a `person_db` id),
`audio_pointer`, `overlap` (two utterances in one file share a pointer: 77,159 unique pointers
for 82,437 utterances — this reproduces Kew's `speech_in_speech` flag), `anonymised`.
The report it prints is the source of the filter table in `notes/DATA.md`.

### `src/verify_against_kew.py` (plumbing, one-off, keep for the record)

Joins our rows to Kew's CSV on (file, utterance index) and compares text and flags. Result:
82,432 identical, 3 differ only by stray `()`, 2 differ by one word, 2 extra empty rows in Kew.
All four real differences match the "second correction phase" in `release_2_notes.pdf`, so
Kew's CSV was built from the pre-correction XML. That is why our parse is now the source of
truth.

### `tests/test_normalise.py`

Pins the policy. One test per rule: NFC composition, the two-codepoint `ö̀`, every kept
diacritic, meta-token stripping, lowercasing, the drop-reason order, the combining acute.
Run with `uv run pytest`. If a future "small cleanup" changes a training target, a test fails.

### What came out, and the three things to be able to say

- **The task is different from every public system, measurably.** 94 % of tokens are
  spelling-ambiguous under Dieth; a Standard German target would not have this property.
- **The corpus is smaller than it looks.** 82,437 utterances → 68,895 usable after dropping
  overlap (10,482), empties (1,671), gaps (910), anonymised (479); missing-audio drops (6,318
  in Kew's count) come once the audio is here.
- **Interviewers are 16 % of the speech and unidentified**, so they go to train only. Splits
  are by document, not utterance, because speaker IDs are document-scoped.

### Artefacts produced (gitignored)

- `data/raw/archimob_r2_text/` — the transcription bundle, unpacked.
- `data/processed/archimob_utterances.jsonl` — 82,437 rows, regenerate with
  `uv run python src/parse_archimob.py data/raw/archimob_r2_text/Archimob_Release_2 data/processed/archimob_utterances.jsonl`.

---

## Phase 3 (text side) and Phase 6 (metric) — done ahead of the audio

### `src/splits.py` (core; the single easiest place to fool ourselves)

Assigns every utterance to train/dev/test **by document**, with the held-out documents
listed as constants with a one-line reason each. Read the module docstring: it explains why
utterance-level splits leak (same voice, room, microphone on both sides), why "by speaker"
equals "by document" here (speaker ids are document-scoped), and why interviewer utterances
of held-out documents are dropped rather than moved to train (unidentified, recurring voices
sharing the test recording). `assign(row)` returns `(split, eval_ok)`; `eval_ok` is true only
for the interviewee in dev/test. Held-out documents cover ZH, BE, LU, BS and three
transcriber/tool combinations. Result: train 36 docs / 452k words, dev 3 / 23k, test 4 / 31k (dev doc 1235 replaced by
1261 once the audio showed 1235 is half missing).

### `src/metrics.py` (core; the metric the brief asks for)

- `build_dieth_to_norm(rows)` — the variant table: each Dieth spelling → the normalised forms
  it was aligned with. Built from the whole usable corpus (a scoring lexicon, not model input;
  same as Kew, so numbers stay comparable).
- `FlexWER.same(a, b)` — two tokens are the same word if equal or if they share a normalised
  form. So *hät / het / hèt* all count as *hat*.
- `_edit_distance(ref, hyp, same)` — a 15-line Levenshtein over word lists with a pluggable
  equality; `jiwer` cannot do that, which is why it is hand-written. Single-row DP, O(n·m).
- `all_metrics(refs, hyps, table)` — returns CER and WER from `jiwer` plus FlexWER.
- Run as a script it writes `data/processed/dieth_to_norm.json` and prints the biggest
  variant sets (*haben* 136 spellings).

### `tests/test_metrics.py`

Four tests on a three-utterance toy corpus: the table groups spellings correctly; a variant
spelling scores 0 under FlexWER but > 0 under WER; a real error still costs; insertions and
deletions are counted; unknown words fall back to exact match.

### `scripts/extract_audio.py` (plumbing, cluster)

The SwissUbase wrapper stores the 20.8 GB inner zip uncompressed, so Python's `zipfile` can
open the inner archive *inside* the outer one and extract members straight to disk. Saves
20 GB of scratch space and one full pass over the data on the cluster.

### `src/prepare.py` (core for Phase 3; the audio side)

- `wav_index(root)` maps every wav file back to its XML media pointer. Four name patterns
  exist (plain `d1209_T821`, multi-part `d1082_2_T5`, EXMARaLDA `TLI_n`, and a doubled
  `1082_3d1082_3_…` prefix on 1,390 files). `tests/test_prepare.py` covers all four, because a
  silent regex miss here would drop thousands of utterances with no error.
- `convert(job)` reads one chunk with `soundfile`, mixes to mono, resamples to 16 kHz with
  `librosa` (soxr_hq) **per file** because the source rate varies by recording, clips and
  writes int16. Returns the duration and one of `ok / too_short / too_long`.
- `main()` runs the conversions in a process pool, then writes `manifest_<split>.jsonl`
  (path, seconds, text, normalised, speaker, region, transcriber, split, eval_ok) and
  `summary.json` (utterances and hours kept per split, drop counts by reason). Filters are
  applied in a fixed order and every drop is counted, so the hours reported are honest.
- The `--docs` flag restricts to a few documents, which is how it was developed locally on
  1225 and 1055 in five seconds; the cluster run uses `--workers 16`.

Why no Hugging Face `Audio` column: see DECISIONS 2026-09-30. The manifest is the dataset;
`train.py` will build a `Dataset` from it and decode in the collator.

---

## Phases 4–5 tooling — evaluation and training scripts (written and smoke-tested locally)

### How the cluster side fits

```
scripts/setup.slurm ──► scripts/cluster_setup.sh     env, extraction, parse, splits, prepare
                                   │
                                   ▼
        /vol/bitbucket/ttm25/stt/data/processed/archimob16k/{wav16/, manifest_*.jsonl}
                                   │
scripts/baseline.slurm ──► src/evaluate.py           zero-shot large-v3 on test  (Phase 4)
scripts/train.slurm    ──► src/train.py              LoRA fine-tune              (Phase 5)
                                   │  writes runs/<name>/{last,best,samples,tb,metrics.jsonl}
                                   ▼
                          src/evaluate.py --adapter runs/<name>/best            (Phase 6)
```

`gpucluster2` is a 4-core submit VM and says so in its banner: nothing but `sbatch` runs there.

### `src/evaluate.py` (core; one code path for baseline and fine-tuned model)

- `load_rows` keeps `eval_ok` rows, optionally takes a fixed random subsample, sorts by duration.
- `transcribe` batches audio through the feature extractor and calls `model.generate` with
  `language="german"`, `task="transcribe"`, greedy. Whisper has no Swiss German token, so
  "German" is the only honest choice; zero-shot it writes Standard German, which is the floor.
- Scores are computed on `normalise_hyp(output)` vs the Dieth reference: lowercase, ß→ss,
  punctuation and digits stripped. Digits are *not* spelled out, so "1917" costs errors.
- Writes `predictions.jsonl`, `metrics.json` (overall, per speaker, per region, real-time
  factor, count of empty outputs) and `examples.md` (REF / HYP as emitted / HYP as scored /
  Standard German gloss). The example files are gitignored: the repo is public and the corpus
  licence forbids redistribution.
- `--adapter DIR` loads a LoRA adapter and merges it, which is all Phase 6 needs.

### `src/train.py` (core; read the module docstring first)

A plain PyTorch loop, about 200 lines, instead of `Seq2SeqTrainer`.
- `ManifestDataset` / `Collator`: read a 16 kHz wav, compute log-mel features (always padded
  to 30 s, that is how Whisper works), tokenise the Dieth text with the `<|de|><|transcribe|>
  <|notimestamps|>` prefix, mask padding with −100, and strip the leading
  `<|startoftranscript|>` because the model re-adds it when it shifts labels right.
- `lora_config`: LoRA on the four attention projections; decoder rank 32 (alpha 64), encoder
  rank 8 via `rank_pattern`. On whisper-tiny this is 2.3 % of parameters.
- Loop: bf16 autocast on GPU, gradient accumulation, gradient clipping at 1.0, linear warm-up
  then linear decay.
- `evaluate()`: every `--eval-every` steps, transcribes a fixed dev subsample, logs
  CER/WER/FlexWER, and writes `samples/step_XXXXXX.txt` with the **same ten utterances every
  time** so you can watch individual sentences change. Step 0 is dumped too: that is the
  zero-shot output for comparison.
- `save_state()`: adapter + optimizer + scheduler + step + RNG into `last/`, written to a temp
  directory and renamed, so a job killed mid-save cannot leave a broken checkpoint. Running the
  same command again resumes. `best/` keeps the lowest dev CER.
- Local smoke test (whisper-tiny, CPU, 30 steps on 0.7 h): loss 3.9 → 3.0, and the output
  flips from "Ich glaube, also..." to lowercase dialect-looking strings. Says nothing about
  quality; proves the plumbing.

### `scripts/*.slurm` and `scripts/cluster_setup.sh` (plumbing)

- `cluster_setup.sh`: idempotent seven-step setup; reuses the uv and HF caches already on
  bitbucket. `setup.slurm` runs it on a compute node.
- `baseline.slurm`: A16 partition (16 GB is plenty for fp16 inference).
- `train.slurm`: A40 by default (48 GB fits large-v3 LoRA at batch 16 without gradient
  checkpointing); `MAX_STEPS`, `RUN` and friends are environment variables so the smoke job and
  the real run are the same script.

### What came out so far (Phases 4 and 5-smoke), and what to be able to say

- **Baseline** (zero-shot large-v3, test): CER 49 %, WER 90 %, FlexWER 57 %. It writes
  Standard German. That is the floor and the proof the task is different.
- **Smoke job** (300 steps, 16 % of an epoch): dev CER 52 % → 22 %. The register flips to
  written dialect within the first 100 steps.
- **Three things broke on the way and are logged in DECISIONS**: the submit node is not for
  computing; large-v3 is stored in fp16 while tiny is fp32, so the laptop smoke test could not
  catch a dtype crash; batch 16 does not fit an A40 without gradient checkpointing.
- Numbers live in `notes/RESULTS.md`; example dumps are local-only files.

### Open after the smoke job (2026-10-01)

1. Timo reads `notes/baseline_examples.md` and `notes/smoke_examples.md`, then decides on
   the full run (`RUN=v1 MAX_STEPS=4000 sbatch scripts/train.slurm`, about 6.5 h on an A40).
2. Policy additions made without prior sign-off, veto possible: drop `<gap>` utterances;
   interviewer utterances train-only; dev doc 1235 → 1261; manifest instead of HF `Audio`.
