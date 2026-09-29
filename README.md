# stt-schwiizerduetsch

Speech-to-text for spoken Swiss German with **written Swiss German (Dieth
orthography)** as the output, not Standard German. A LoRA fine-tune of Whisper
on the ArchiMob corpus. See `notes/BRIEF.md` for the goal and
`notes/DECISIONS.md` for every non-obvious choice made along the way.

Status: v1 in progress. Phases 0–2 done (repo, cluster recon, orthography
analysis); dataset construction and training pending audio access.

## Layout

- `src/` — data preparation, normalisation, training, evaluation
- `scripts/` — SLURM job scripts for the Imperial DoC cluster
- `notes/` — brief, data findings, decisions, results
- `data/raw`, `data/processed` — corpus and derived datasets (gitignored, never committed)
- `third_party/` — reference clones of prior work (gitignored)

## Data

This project uses **ArchiMob Release 2 (2019)**, obtained from SWISSUbase /
LaRS under a research-and-teaching download contract. The data is not
redistributed here in any form, original or derived. Please cite the datasets
if you build on this work:

> Scherrer, Y., Samardzic, T., & Glaser, E. (2023). *ArchiMob Release 2 (2019)*
> (Version 1.0.0) [Data set]. LaRS – Language Repository of Switzerland.
> https://doi.org/10.48656/496p-3w34

> Scherrer, Y., Samardzic, T., & Glaser, E. (2023). *ArchiMob Release 2 (2019) –
> Full set of audio files* (Version 1.0.0) [Data set]. LaRS – Language
> Repository of Switzerland. https://doi.org/10.48656/brdm-ht43

Canonical (version-independent) DOIs: transcriptions
https://doi.org/10.48656/6g4e-8k73, audio https://doi.org/10.48656/990x-pb34.

Corpus paper:

> Samardžić, T., Scherrer, Y., & Glaser, E. (2016). ArchiMob – A Corpus of
> Spoken Swiss German. *Proceedings of LREC 2016*.

## Prior work reused

- Kew, T. et al. — `tannonk/two-headed-master`: Release 2 text layer as CSV
  (`archimob.csv`) and the VarDial 2020 FlexWER variant table.
- Nigmatulina, I. — `yunigma/Kaldi-for-ASR-of-Swiss-German`: reference for
  segmentation and normalisation.
- Nigmatulina, I., Kew, T., & Samardžić, T. (2020). ASR for Non-Standardised
  Languages with Dialectal Variation: the Case of Swiss German. *Proceedings
  of VarDial 2020*.

## Setup

```bash
uv sync
python src/analyse_text.py third_party/two-headed-master/data/archimob.csv
```
