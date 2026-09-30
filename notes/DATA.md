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

## Release 2 XML (downloaded 2026-09-29) — verified against Kew's CSV

Format is **TEI XML** (`Archimob_Release_2/<doc>[_part].xml`, 52 files for 43
interviews) plus `person_file.xml` and `Metadata.txt` (per doc: speaker,
birth year, sex, profession, dialect area, transcriber, tool, transcription
phase, manual/automatic normalisation). Not EXMARaLDA, not TextGrid.

Per utterance `<u start="media_pointers#d1007-T31" who="...">` with one
`<w normalised="..." tag="POS">dieth</w>` per word — **both layers present
for every one of the 581,974 words** (0 empty normalisations). Timestamps
are NOT in the XML: `start` points into an external media-pointer file that
is not shipped, so segment durations wait on the audio zip.

Annotation elements (corpus totals): `<pause/>` 10,796 · `<vocal><desc>`
12,928 (eh/ää/ehm/[lacht]...) · `<del type="truncation">` 8,836 (word
fragments like `zw/`) · `<unclear>` 3,348 (wraps words the transcriber was
unsure of; words kept) · `<gap reason="unintelligible"/>` 1,549 (speech with
no text) · `<incident>`/`<kinesic>`/`<other>` 416 (non-speech events).

`src/parse_archimob.py` parses this directly. `src/verify_against_kew.py`
cross-checked it against `archimob.csv`: 82,432 utterances byte-identical
after stripping Kew's meta tokens, 3 differ only by stray `()`, 2 differ by
one word, and Kew has 2 extra empty rows. All 4 real differences match the
"second correction phase" items in `release_2_notes.pdf` → Kew's CSV was
built from the pre-2019-correction XML. **Our parse is the source of truth
from now on**; the CSV is no longer used.

Speakers: `who` is `interviewer` (13,409 utts = 16%), `otherPerson`, or a
`person_db` id. Interviewer voices are not identified and may recur across
documents — Phase 3 must keep interviewer utterances out of dev/test.

Overlap: `audio_pointer` is shared by two utterances when speech overlaps
(77,159 unique pointers for 82,437 utterances). This reproduces Kew's
`speech_in_speech` flag except for 1 row.

### v1 target filter (`normalise.usable`)

| reason | utterances |
|---|---|
| overlap (shared audio chunk) | 10,482 |
| empty after stripping markers | 1,671 |
| contains `<gap>` (audible speech without text) | 910 |
| anonymised (`***`) | 479 |
| **usable** | **68,895** (513,927 words) |

`missing_audio` (6,318 in Kew's CSV) cannot be derived from the XML; it is
re-derived from the actual chunk files once the audio is on the cluster, so
the usable count will drop further.

## Splits (Phase 3, text side; hours pending audio)

| split | docs | utterances | words | share of words | regions |
|---|---|---|---|---|---|
| train | 36 | 60,833 (incl. interviewer) | 452,402 | 89.3% | all 15 |
| dev | 3 (1055 ZH, 1142 BE, 1261 LU) | 2,725 | 22,826 | 4.5% | ZH, BE, LU |
| test | 4 (1225 ZH, 1121 BE, 1195 LU, 1263 BS) | 4,291 | 31,223 | 6.2% | ZH, BE, LU, BS |
| dropped | — | 14,535 | — | — | unusable + non-interviewee rows of held-out docs |

Rationale and rules in `src/splits.py`. Train still holds ZH 16.8k, AG
13.6k, BS 5.9k, BE 5.3k, LU 3.5k interviewee utterances.

## FlexWER variant table

`src/metrics.py` builds `dieth_to_norm.json` from all usable utterances:
43,815 Dieth spellings, 28,713 normalised forms, 3,914 spellings that map
to more than one normalised form (homographs like *me* = "man"/"wenn").
Largest variant sets: *haben* 136, *eigentlich* 81, *können* 76, *nachher*
73, *wenn* 70. Slightly smaller than Kew's table (147/93/82) because we
drop unusable utterances first.

## Character table (Dieth layer)

Text is already lowercased and punctuation-free. Full table in
`notes/char_table.txt`. Summary:

| class | chars | verdict |
|---|---|---|
| ASCII letters | a–z | keep |
| Umlauts | ä (2.87%), ü (1.18%), ö (0.80%) | semantically essential, keep |
| Grave = open vowel quality (Dieth) | ì (0.47%), è (0.35%), ò (0.27%), ù (0.17%), ǜ (0.07%) | meaningful in Dieth, keep for v1 |
| Tilde = nasal vowel | õ (489), ã (104), ẽ (70), ĩ (1) | meaningful but rare; keep |
| Acute | é (114), ó (8), á (5), í (3), ú (2) | mostly loanwords; keep all as-is (too rare to matter) |
| à | 8 | grave on a, same convention as ì/è/ò/ù; keep |
| Combining marks | U+0300 (759), U+0301 (1 after NFC) | NFC-normalise; ö̀ stays 2 codepoints (no precomposed form); U+0301 dropped |
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

## Audio (downloaded 2026-09-30, MD5 verified)

`swissubase_2277_1_0.zip` (19.4 GB) wraps `archimob_r2_audio_share.zip`
(20.8 GB, stored uncompressed → readable in place; `scripts/extract_audio.py`
extracts without materialising it). Inside: `audio_segmented_anonymized/
<doc_part>/` — **78,156 wavs already cut per utterance**, 23.5 GB, 52
folders. Mono, 16-bit, **sample rate varies by recording** (48 kHz and
44.1 kHz seen) → per-file resampling to 16 kHz in `src/prepare.py`.

File name = XML media pointer with `-`→`_` (`d1209-T821` → `d1209_T821.wav`;
EXMARaLDA docs use `TLI_n`); 1,390 files in 1082_3 carry a doubled
`1082_3d1082_3_…` prefix. Sizes: median 281 KB ≈ 3 s at 48 kHz; 2,248
files < 1 s; max 15 MB ≈ 160 s (dropped by the 30 s Whisper limit).

Coverage against the XML: 77,158 pointers, 76,120 with a wav (1,038
missing, no zero-byte files); 386 wavs have no XML utterance. Of the
split-assigned utterances 430 lack audio, **403 of them in doc 1235** whose
audio stops at chunk 519 of 988 → 1235 removed from dev (replaced by 1261).
Kew's `missing_audio` count (6,318) was from an older distribution; ours is
the reinstated one described in the release notes.

Local dev copy: docs 1225 and 1055 extracted to `data/raw/audio_dev/`
(659 MB) so `prepare.py` could be developed against real files. Result on
those two: 842 dev + 850 test utterances kept, 0.75 h + 0.79 h, 5 dropped
as < 1 s. Full-corpus hours per split are filled in after the cluster run.

## BLOCKERS

1. ~~Audio download needs a human~~ — done 2026-09-30 (contract approved,
   MD5 verified). Next: transfer to `/vol/bitbucket/ttm25/stt/data/raw`,
   extract, run `prepare.py` there.
