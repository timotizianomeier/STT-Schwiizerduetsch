# Claude Code Brief: Swiss German → Dieth-Orthography ASR

(Timo's original brief, written 2026-08 from a literature scan. Stored verbatim on
2026-09-29 after it was found missing from the repo. Corrections discovered
against reality are logged in DECISIONS.md, not edited here.)

## Context you need before starting

I want to build a speech-to-text model that transcribes spoken Swiss German into **written Swiss German** (Dieth orthography), not into Standard German.

This distinction is the entire point of the project. Almost all existing Swiss German ASR work — the ZHAW corpora, SDS-200, STT4SG-350, the public Hugging Face fine-tunes — maps Swiss German audio to *Standard German* text. That is a speech translation task. It is not what I want.

The only serious prior work on dialect-to-dialect output is Nigmatulina & Kew, "ASR for Non-standardised Languages with Dialectal Variation: the case of Swiss German" (VarDial 2020), which reported roughly 60% WER on ArchiMob. Nobody has revisited this with Whisper-class models. That gap is the opportunity, and the 60% number is the honest prior on difficulty.

**Do not silently substitute Standard German targets to make metrics look better.** If you find yourself doing that, stop and tell me.

## Scope: v1 only

This is a weekend-scale v1, not a research programme. Deliberately out of scope:

- Two-stage training (encoder adaptation on the 343h Standard German corpus first)
- SwissDial (licensing/access adds friction; ArchiMob alone is enough for v1)
- All eight dialect regions
- Any LLM post-processing layer

Build the smallest thing that answers one question: **does a LoRA-fine-tuned Whisper produce output that reads like written Swiss German?**

Character error rate is the proxy. My ear is the actual judge. Optimise the pipeline for fast qualitative inspection, not for leaderboard numbers.

---

## Phase 0 — Repo and environment

1. Initialise a repo `swiss-german-asr` with `uv` (preferred) or a venv. Python 3.11.
2. Dependencies: `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `librosa`, `soundfile`, `jiwer`, `evaluate`, `tensorboard`.
3. Structure:
   ```
   data/raw/          # downloaded corpora, gitignored
   data/processed/    # HF datasets on disk, gitignored
   src/prepare.py     # corpus → HF dataset
   src/normalise.py   # orthography handling
   src/train.py       # LoRA fine-tune
   src/evaluate.py    # metrics + qualitative dump
   scripts/*.slurm    # cluster job scripts
   notes/             # findings, decisions, failures
   ```
4. `.gitignore` must exclude audio and checkpoints. Do not commit corpus data — ArchiMob is CC BY-NC-SA and redistribution has conditions.
5. Maintain `notes/DECISIONS.md` from the start. Every non-obvious choice gets a line. I want to be able to reconstruct why later.

## Phase 1 — Data acquisition

**ArchiMob Release 2 (2019)** is the corpus. Licence is CC BY-NC-SA 4.0 — direct download, no application.

- Landing page: https://www.spur.uzh.ch/en/research/projects-all/lab-projects/ArchiMob0.html
- Zenodo DOI: https://doi.org/10.5281/zenodo.1158572

Release 2 is what we want specifically because it has **speech-to-text alignment at utterance level (4–10 second segments)**. Release 1 does not, and would require forced alignment. Verify you have Release 2 before proceeding.

Also clone the preprocessing work from Iuliia Nigmatulina's and Tannon Kew's public GitHub repos (linked from the ArchiMob page). They built Kaldi systems on exactly this corpus. **Read their segmentation and text normalisation code before writing your own.** Reuse what transfers. Do not reimplement from scratch out of habit — report to me what you reused and what you had to replace.

Deliverable: a written summary in `notes/DATA.md` of what's actually in the download — number of documents, speakers, total audio hours, transcription format (XML? TextGrid?), and whether the Dieth transcription and the normalised Standard German layer are both present per utterance.

## Phase 2 — Orthography (the hard part — do not rush this)

ArchiMob provides two transcription layers: Dieth dialectal transcription, and a normalised layer. We are training on the **Dieth layer**.

Dieth spelling (Eugen Dieth, *Schwyzertütschi Dialäktschrift*, 1986) represents dialect phonetics using Standard-German-like spelling conventions plus limited diacritics. Crucially it is a *convention*, not a strict standard — annotators vary.

Tasks:

1. Extract the Dieth text layer. Build a character-frequency table across the whole corpus. Show it to me.
2. Identify every non-ASCII character and diacritic in use. Decide, and record in `notes/DECISIONS.md`, which are semantically meaningful and which are annotator noise.
3. Decide on and implement a normalisation policy: casing, punctuation, hesitation and disfluency markers, overlapping speech markers, inaudible-segment markers. ArchiMob transcripts contain annotation artefacts — these must not become training targets.
4. **Report the ambiguity rate**: how often does the same spoken word appear with different Dieth spellings across the corpus? This number determines whether plain WER is meaningful at all. I expect it to be high.

Do not proceed to training until you have shown me the character table and the ambiguity estimate.

## Phase 3 — Dataset construction

1. Build a Hugging Face `DatasetDict` with `audio` (16 kHz mono) and `text` (normalised Dieth) columns.
2. **Split by speaker, not by utterance.** ArchiMob has relatively few speakers and long interviews per speaker. Utterance-level splitting will leak speaker identity and give you a flatteringly wrong result. This is the single easiest way to fool ourselves — be explicit in the code about why the split works the way it does.
3. Hold out enough speakers for a meaningful test set but keep train as large as possible. Report the actual hours in each split.
4. Drop utterances shorter than ~1s or longer than 30s (Whisper's window).
5. Cache the processed dataset to `data/processed/` with `save_to_disk`.

## Phase 4 — Baseline before training

Run zero-shot `openai/whisper-large-v3` on the test split with German forced as the language.

It will emit Standard German and score badly against Dieth references. **That is the expected and desired result** — it establishes that the task is genuinely distinct and gives us a floor. Record WER and CER.

Dump 30 side-by-side examples (reference vs hypothesis) to `notes/baseline_examples.md`. I want to read these.

## Phase 5 — LoRA fine-tune

- Base: `openai/whisper-large-v3`. If cluster VRAM constrains you, fall back to `large-v3-turbo` and say so.
- PEFT LoRA on attention projections. Start `r=32`, `alpha=64`, dropout 0.05.
- The decoder is what needs to learn Dieth. If you're rank-constraining anywhere, constrain the encoder first.
- bf16, gradient checkpointing, batch size to fit, gradient accumulation to a sensible effective batch.
- Evaluate on the held-out speakers every N steps. Log to TensorBoard.
- **At every eval step, dump 10 sample transcriptions to a text file.** Metrics alone will not tell me whether the output reads like Swiss German. This is the most important logging in the project.
- Checkpoint often. Cluster jobs get killed.

Run one short smoke job (a few hundred steps) and show me sample output before launching anything long.

## Phase 6 — Evaluation

1. **CER is the primary metric.** WER punishes legitimate spelling variation and will be misleadingly awful.
2. Implement a FlexWER-style metric following the VarDial 2020 paper — scores a hypothesis as correct if it matches any attested spelling variant of the reference token. Build the variant table from the corpus itself.
3. Report per-speaker and, if dialect region is recoverable from ArchiMob metadata, per-region.
4. Produce `notes/RESULTS.md`: baseline vs fine-tuned, CER/WER/FlexWER, plus 50 qualitative examples.
5. Include failure examples, not just good ones. I specifically want to see where it collapses back into Standard German.

---

## Imperial GPU cluster

I'm an MSc Computing student, so the **Department of Computing SLURM cluster** is my route. Note: students cannot directly request the central RCS/CX3 GPU nodes — only a supervisor can apply on a student's behalf — so assume DoC unless I tell you otherwise.

Guide: https://www.imperial.ac.uk/computing/csg/guides/hpcomputing/gpucluster/

**Discover, don't assume.** Cluster configuration changes. Before writing job scripts, SSH in and actually run:

```bash
sinfo                      # partitions, time limits, node states
sinfo -o "%P %G %N"        # GPU types per partition
squeue -u $USER
scontrol show partition
nvidia-smi                 # on an allocated node, not the head node
ls /vol/cuda               # available CUDA versions
quota                      # storage limits — corpora are large
```

Report what you find back to me before writing `scripts/*.slurm`. Write the job scripts against the real configuration, not against a template.

Rules:

- **Never run compute on the head node.** Use `salloc` for interactive debugging, `sbatch` for real runs.
- Ask for the smallest GPU that fits. Grabbing an A100 for a smoke test is antisocial and gets noticed.
- Set `--time` realistically; jobs are killed at the limit with no grace.
- Check the storage quota before downloading ArchiMob. If home directory quota is tight, find the scratch or `/vol/` space and put `data/` there, with symlinks.
- Cache Hugging Face models to a shared/scratch location via `HF_HOME`, not to home. Whisper large-v3 is several GB.
- Write `scripts/train.slurm` with checkpoint resumption so a killed job can be requeued without losing everything.

Set up SSH config for painless reconnection, and use `tmux` on the head node for anything interactive.

---

## How I want you to work

- **Show me things before proceeding.** Character table, ambiguity rate, smoke-test samples, cluster config. Do not run the full pipeline end to end and present a finished result.
- **Flag when reality contradicts this brief.** I wrote it from a literature scan, not from having opened the corpus. If ArchiMob's structure isn't what I described, tell me rather than working around it silently.
- Commit at each phase boundary with a real message.
- When something fails, write it in `notes/DECISIONS.md`. Failed approaches are the most valuable output of a v1.
- Prefer boring, inspectable code. This is an experiment, not a product.

## Definition of done for v1

A fine-tuned checkpoint, a results file with honest metrics, and fifty transcription examples I can read and judge for myself. If CER is bad but the output looks recognisably like Züridütsch, that is a success. If CER is good but it reads like Standard German with typos, that is a failure regardless of the number.
