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
