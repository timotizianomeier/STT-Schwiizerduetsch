# Decisions log

Format: date — decision — why.

- 2026-08-25 — Build inside the existing `STT-Schwiizerduetsch` repo instead of
  initialising a new `swiss-german-asr` repo. The repo already existed with an
  initial commit; a nested repo would be pointless friction. Package name in
  `pyproject.toml` is `stt-schwiizerduetsch`.
- 2026-08-25 — **Brief correction:** the Zenodo DOI in the brief
  (10.5281/zenodo.1158572) resolves to ArchiMob **Release 1** (2016), not
  Release 2. Release 2 (2019) is distributed via SwissUbase:
  transcriptions/XML at doi:10.48656/496p-3w34 (`Archimob_Release_2.zip`),
  full audio at doi:10.48656/brdm-ht43 (`archimob_r2_audio_share.zip`).
  We use the SwissUbase Release 2 artefacts.
- 2026-08-25 — Clone Nigmatulina (yunigma/Kaldi-for-ASR-of-Swiss-German) and
  Kew (tannonk/two-headed-master) into `third_party/` (gitignored, reference
  only) per brief: read their segmentation/normalisation code before writing
  ours.
- 2026-08-25 — **Reuse Kew's XML→CSV conversion output** (`data/archimob.csv`
  in two-headed-master) as the text-layer source instead of re-parsing the
  Release 2 XML. Token count matches the official corpus stats; the CSV
  already maps annotation artefacts to `<SPOKEN_NOISE>/<SIL_WORD>/<NOISE>`
  and carries per-utterance quality flags. Their Python-2 Kaldi pipeline
  itself is NOT reused. Re-verify against the XML once
  `Archimob_Release_2.zip` is downloaded.
- 2026-08-25 — Reuse Kew's `flexwer_mapping.json` for the Phase 6 FlexWER
  metric (it is the VarDial 2020 variant table built from this corpus).
- 2026-08-25 — Do NOT reuse Kew's train/dev/test splits as-is: they are not
  speaker-disjoint (doc 1007 utterances appear in train, dev and test). We
  split by document (Phase 3). Caveat to record there: interviewer voices may
  recur across documents under different doc-scoped IDs.
- 2026-08-25 — Normalisation policy (Phase 2, pending Timo's sign-off):
  NFC-normalise Unicode (759 decomposed combining graves found); keep umlauts,
  Dieth grave (open-vowel) and tilde (nasal) diacritics as-is; drop the 479
  anonymity-flagged utterances (contain `*` redaction masks); strip stray
  `()`; drop utterances flagged missing_audio or speech_in_speech or
  no_relevant_speech; decide `<SPOKEN_NOISE>`/`<SIL_WORD>`/`<NOISE>` handling
  at dataset build (v1 plan: strip tags, keep utterance if text remains).
- 2026-08-25 — Cluster facts (gpucluster2.doc.ic.ac.uk, discovered not
  assumed): partitions a40/a30/t4/a16*/a100/long/training; 3-day MaxTime
  except long (30 d). A100-80GB×4 on linnet/merlin/vm-he-a100. Home quota
  10.5/12 GB — nearly full → all data, HF cache, checkpoints go to
  /vol/bitbucket/ttm25. shell4 has no SLURM tools; submit from gpucluster2.
- 2026-08-25 — `bitsandbytes` declared with `sys_platform == 'linux'` marker
  (no macOS wheels); local Mac env is for data prep/analysis only.
- 2026-09-29 — Brief recovered and stored verbatim at `notes/BRIEF.md` (it had
  never been committed). Any future divergence from it is logged here, not
  edited into the brief.
- 2026-09-29 — **Normalisation policy signed off by Timo** as proposed on
  2026-08-25 (NFC; keep umlauts, Dieth graves and tildes; drop anonymity,
  missing_audio, speech_in_speech, no_relevant_speech utterances; strip `()`;
  strip `<SPOKEN_NOISE>/<SIL_WORD>/<NOISE>` tags, keep utterance if text
  remains). Implementation goes in `src/normalise.py`.
- 2026-09-29 — Imperial DoC cluster access confirmed still active post-MSc;
  cluster plan from 2026-08-25 stands.
- 2026-09-29 — **Brief correction (licence):** the audio is NOT a plain
  CC BY-NC-SA direct download. SWISSUbase requires a per-user download
  contract with approval: research/teaching use only, no redistribution in
  original *or edited* form, keep from third parties, delete after the
  project, cite the dataset. Timo's contract: 3 months, education/training use, publication
  planned (per the pending-download page), submitted 2026-09-29. Consequences: the
  processed HF dataset stays on `/vol/bitbucket` only; no dataset or audio
  ever leaves the cluster or lands in a public bucket; a trained checkpoint
  is not covered explicitly by the contract wording, so publishing weights
  on the Hub is deferred until Timo asks LaRS. The "daily-use app" idea is a
  personal-use v2 question and does not conflict with research use, but
  redistribution of the model would need clarification.
- 2026-09-29 — Contract duration 3 months → audio must be deleted from the
  cluster by ~2026-12-29 unless renewed. Put a reminder in RESULTS.md when
  v1 closes.
- 2026-09-29 — README carries the LaRS citations for both datasets (contract
  clause 3) and states that no data is redistributed.
- 2026-09-29 — **Parse the Release 2 XML ourselves** (`src/parse_archimob.py`)
  instead of Kew's `archimob.csv`. Verified byte-identical on 82,432/82,437
  utterances; the 4 real differences are 2019 corrections that post-date
  Kew's CSV. Our parse also keeps per-utterance element counts that the CSV
  flattens away. Reused from prior work: the element→token mapping idea
  (Nigmatulina's `process_exmaralda_xml.py`), utterance indexing scheme,
  overlap-via-shared-pointer definition. Replaced: everything else (Py2,
  Kaldi-specific).
- 2026-09-29 — **Drop utterances containing `<gap reason="unintelligible">`**
  (910 usable-otherwise). Kew's CSV silently deleted the marker, leaving
  audio with speech the text omits — a target that teaches the model to
  skip words. Not in the signed-off policy; added, flagged to Timo.
- 2026-09-29 — Hesitations (`<vocal>`), truncated fragments (`<del>`),
  pauses and non-speech events are stripped, utterance kept. Consequence:
  the model learns to omit fillers and fragments, which is what a daily-use
  transcript wants. `<unclear>` words are kept (transcriber's best guess).
- 2026-09-29 — Orthography detail: keep `à` (8 tokens, grave-on-a); drop the
  single combining acute (`kapä́l`); keep acute letters as-is rather than
  folding (DATA.md had said "fold" — reversed, too rare to justify a rule).
  Policy pinned by `tests/test_normalise.py`. The tests caught a real bug on
  first run (lowercasing before stripping upper-case meta tokens).
- 2026-09-29 — Interviewer utterances are 16% of the corpus and interviewer
  identity is not recorded → **Phase 3 rule: interviewer utterances go to
  train only, never dev/test.** Same for `otherPerson`.
