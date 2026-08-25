# ArchiMob data notes

## Provenance — deviation from the brief

The brief's Zenodo DOI (10.5281/zenodo.1158572) is **Release 1** (2016, no
utterance alignment). Release 2 (2019) is on SwissUbase:

- Transcriptions/XML + docs: doi:10.48656/496p-3w34 → `Archimob_Release_2.zip`
- Full audio: doi:10.48656/brdm-ht43 → `archimob_r2_audio_share.zip`
- Audio samples only: doi:10.48656/kdap-cy77

SwissUbase downloads sit behind a SWITCH edu-ID login (browser SPA, no
anonymous endpoint found). **Audio acquisition is therefore a manual step for
Timo** — see BLOCKERS below.

However: `third_party/two-headed-master/data/` (Kew's public repo, linked from
the official ArchiMob page) ships the **complete Release 2 text layer** as
`archimob.csv`, produced from the Release 2 XML by their
`process_exmaralda_xml.py`. Token count (581,974 vs officially stated 581,976)
confirms it is the full Release 2 corpus. So Phases 2–3 (text side) are
unblocked; only audio-dependent steps (dataset build, training) wait on the
SwissUbase download.

## What is in archimob.csv

- 82,439 utterances, 52 audio documents (43 interviews; some split into
  parts), 145 doc-scoped speaker IDs (~3 per doc: interviewee + interviewer +
  occasional third voice).
- Columns: `utt_id`, `transcription` (**Dieth**), `normalized` (Standard-
  German-like normalisation layer), `speaker_id`, `audio_id` (chunk ID,
  `d<doc>-T<n>`), plus flags `anonymity`, `speech_in_speech`,
  `missing_audio`, `no_relevant_speech`.
- Both layers are present for every utterance. Token-aligned 1:1 in the
  overwhelming majority of utterances.
- Flags set: speech_in_speech 10,509 · missing_audio 6,318 · anonymity 479 ·
  no_relevant_speech 268. These will be filtered in dataset construction
  (missing_audio and anonymity are hard drops; speech_in_speech = overlapping
  speech, drop for v1).
- Annotation artefacts already mapped by Kew's pipeline to `<SPOKEN_NOISE>`
  (21,210), `<SIL_WORD>` (10,796), `<NOISE>` (969).
- `*` appears only as anonymisation mask (`e***` = redacted name), exactly in
  the 479 `anonymity`-flagged utterances. `()` are 5 stray artefacts.
- Dialect region metadata: `meta/fileId_dialect.json` maps doc → region.
  14 regions; ZH dominates (12 docs), then AG 6, LU 5, BE 5, BS 4, rest 1–2.
- Kew's repo also ships train/dev/test splits (`data/splits/`) and the
  VarDial-2020 **FlexWER variant mapping** (`data/flexwer_mapping.json`,
  31,755 entries) — directly reusable for Phase 6.
  ⚠ Their dev/test splits are NOT speaker-disjoint from train (e.g. doc 1007
  utterances appear in all three). We will build our own document-disjoint
  splits (Phase 3) and only use theirs for comparability if needed.

## Character table (Dieth layer)

Text is already lowercased and punctuation-free. Full table in
`notes/char_table.txt`. Summary:

| class | chars | verdict |
|---|---|---|
| ASCII letters | a–z | keep |
| Umlauts | ä (2.87%), ü (1.18%), ö (0.80%) | semantically essential, keep |
| Grave = open vowel quality (Dieth) | ì (0.47%), è (0.35%), ò (0.27%), ù (0.17%), ǜ (0.07%) | meaningful in Dieth, keep for v1 |
| Tilde = nasal vowel | õ (489), ã (104), ẽ (70), ĩ (1) | meaningful but rare; keep |
| Acute | é (114), ó (8), á (5), í (3), ú (2) | mostly loanwords/annotator noise; keep é, fold others → NFC + review |
| Combining marks | U+0300 (759), U+0301 (3) | **decomposed Unicode — must NFC-normalise** |
| `*` | 1,536 | anonymisation mask → drop utterances |
| `()` | 5 | artefacts → strip/drop |

## Ambiguity of Dieth spellings (the number that matters)

Using the normalised layer as word identity (same operationalisation as
FlexWER, VarDial 2020), over 581,974 token-aligned pairs:

- 31,782 normalised word types; **25.8% of types** have >1 attested Dieth
  spelling.
- **94.0% of running tokens** belong to a type with >1 attested spelling.
- Extreme cases: *haben* → 147 variants, *eigentlich* → 93, *können* → 82.
  (Some of this is genuine dialect variation across regions, some is
  morphology hidden by normalisation (clitics: *hemmer* = "haben wir"), some
  is annotator inconsistency (*hät/het/hèt*).)

Conclusion: plain WER is close to meaningless as a primary metric, exactly as
the brief suspected. CER primary, FlexWER secondary (mapping table already
available), WER reported for comparability only.

## Audio (pending)

Original audio: wav, 48 kHz→16 kHz conversion needed, mono, chunked
per-utterance by Kew's `split_audio.py` naming scheme keyed on `utt_id`.
Segment durations unverifiable until download. Full-set zip size unknown
(SwissUbase doesn't show it anonymously); plan for tens of GB on
`/vol/bitbucket/ttm25`.

## BLOCKERS

1. **Audio download needs a human**: create/log into SWITCH edu-ID at
   swissubase.ch, accept the CC BY-NC-SA usage licence, download
   `archimob_r2_audio_share.zip` (and ideally `Archimob_Release_2.zip` for
   provenance/XML) — or generate download links and hand them over.
