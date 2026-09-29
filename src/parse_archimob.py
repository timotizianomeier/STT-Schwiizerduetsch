"""ArchiMob Release 2 TEI-XML -> one row per utterance (Phase 1/2 groundwork).

Why parse the XML ourselves instead of using Kew's archimob.csv (which we
verified token-for-token)? Because the CSV flattens the annotation elements
into three Kaldi tokens and loses information we need for target selection:

  <w normalised=".." tag="..">dieth</w>   the word itself (kept)
  <pause/>                                silence            (count only)
  <vocal><desc>eh</desc></vocal>          hesitation/laughter (count only)
  <del type="truncation">zw/</del>        word fragment       (count only)
  <unclear><w>..</w></unclear>            transcriber unsure  (words kept, counted)
  <gap reason="unintelligible"/>          speech with NO text (count -> drop utt)
  <incident>/<kinesic>/<other>            non-speech events   (count only)

Utterance-level facts we derive:
  * speaker: "interviewer", "otherPerson", or the person_db id (e.g. EJos1007)
  * audio_pointer: the media pointer (e.g. d1007-T31) that names the audio
    chunk. NOT unique: overlapping speech shares one chunk between two
    utterances, which is how Kew's speech_in_speech flag was derived.
  * anonymised: dieth text contains "***" (redacted name)

Run: python src/parse_archimob.py data/raw/archimob_r2_text/Archimob_Release_2 \
         data/processed/archimob_utterances.jsonl
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from normalise import normalise_dieth, unexpected_chars, usable

NS = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
COUNTED = ("pause", "vocal", "del", "unclear", "gap", "incident", "kinesic", "other")


def load_metadata(xml_dir: Path) -> dict[str, dict]:
    """Metadata.txt: one row per interview (doc) — dialect area, transcriber,
    tool, transcription phase, manual/automatic normalisation. The dialect
    area is what Phase 6 reports per-region results on; transcriber/tool is
    the likely source of spelling inconsistency (release notes describe
    per-phase corrections), so the split should mix them."""
    meta = {}
    with open(xml_dir / "Metadata.txt", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            area = r["Dialect area"]
            meta[r["DocID"]] = {
                "region": area.split()[0],          # canton code, e.g. "ZH"
                "region_detail": area,
                "transcriber": r["Transcriptor"],
                "tool": r["Tool"],
                "phase": r["Transcription phase"],
                "normalisation": r["Normalisation"],
            }
    return meta


def speaker_of(who: str) -> str:
    # who="interviewer" | "otherPerson" | "person_db#EJos1007"
    return who.split("#", 1)[-1]


def parse_doc(path: Path, meta: dict[str, dict]) -> list[dict]:
    root = ET.parse(path).getroot()
    doc_part = path.stem            # "1082_2"
    doc = doc_part.split("_")[0]    # "1082" — the interview (= main speaker)
    doc_meta = meta[doc]
    rows = []
    for idx, u in enumerate(root.iter(NS + "u"), start=1):
        dieth, norm, counts = [], [], Counter()
        n_eq = 0
        for el in u.iter():
            tag = el.tag.replace(NS, "")
            if tag == "w":
                w = (el.text or "").strip()
                if not w:
                    continue
                n = el.get("normalised", "")
                if n == "==":       # annotator placeholder for "no normalisation"
                    n_eq += 1
                dieth.append(w)
                norm.append(n)
            elif tag in COUNTED:
                counts[tag] += 1
        raw = " ".join(dieth)
        rows.append({
            "utt_id": u.get(XML_ID),                    # d1007-u18
            "doc": doc,
            "doc_part": doc_part,
            "idx_in_part": idx,                         # matches Kew's utt_id suffix
            "audio_pointer": u.get("start", "").split("#", 1)[-1],
            "speaker": speaker_of(u.get("who", "")),
            **doc_meta,
            "raw": raw,
            "text": normalise_dieth(raw),
            "normalised": " ".join(norm),
            "n_words": len(dieth),
            "n_norm_eq": n_eq,
            "anonymised": "***" in raw,
            **{f"n_{t}": counts[t] for t in COUNTED},
        })
    return rows


def main(xml_dir: str, out_path: str) -> None:
    rows: list[dict] = []
    meta = load_metadata(Path(xml_dir))
    for path in sorted(Path(xml_dir).glob("*.xml")):
        if path.name == "person_file.xml":
            continue
        rows.extend(parse_doc(path, meta))

    # Overlap: two utterances in the same file pointing at the same chunk.
    seen = Counter((r["doc_part"], r["audio_pointer"]) for r in rows)
    for r in rows:
        r["overlap"] = seen[(r["doc_part"], r["audio_pointer"])] > 1
        r["usable"], r["drop_reason"] = usable(r)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- report -----------------------------------------------------------
    print(f"files: {len(set(r['doc_part'] for r in rows))}  "
          f"docs: {len(set(r['doc'] for r in rows))}  utterances: {len(rows)}")
    print(f"words: {sum(r['n_words'] for r in rows)}")
    print("element counts:", {t: sum(r[f'n_{t}'] for r in rows) for t in COUNTED})
    print("utterances with >=1:", {t: sum(r[f'n_{t}'] > 0 for r in rows) for t in COUNTED})
    print("speakers:", Counter(r["speaker"] for r in rows).most_common(5), "...")
    print(f"interviewer utterances: {sum(r['speaker']=='interviewer' for r in rows)}")
    print(f"normalised=='==' tokens: {sum(r['n_norm_eq'] for r in rows)}")
    print("drop reasons:", Counter(r["drop_reason"] for r in rows))
    kept = [r for r in rows if r["usable"]]
    print(f"usable: {len(kept)} utterances, {sum(r['n_words'] for r in kept)} words")
    odd = Counter()
    for r in kept:
        for c in unexpected_chars(r["text"]):
            odd[c] += 1
    print("unexpected chars in usable text:", dict(odd) or "none")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
