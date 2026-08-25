"""Phase 2 reconnaissance on the ArchiMob Dieth text layer.

Reads the utterance CSV shipped with tannonk/two-headed-master (derived from
ArchiMob Release 2 XML via their process_exmaralda_xml.py) and reports:

  1. corpus shape: documents, speakers, utterances, token counts, flag counts
  2. full character frequency table for the Dieth layer
  3. spelling ambiguity: how many Dieth surface forms map to the same
     normalised form, and what share of tokens are ambiguous

The normalised layer gives us word identity across dialects/spellings, so
"variants per normalised form" is our operational definition of ambiguity —
the same definition the VarDial 2020 FlexWER metric uses.

Run: python src/analyse_text.py third_party/two-headed-master/data/archimob.csv
"""

import csv
import sys
import unicodedata
from collections import Counter, defaultdict

META_TOKENS = {"<SPOKEN_NOISE>", "<NOISE>", "<SIL_WORD>"}


def main(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    print(f"utterances: {len(rows)}")

    docs = Counter()
    speakers = Counter()
    flags = Counter()
    char_freq = Counter()
    dieth_tokens = 0
    meta_token_count = Counter()
    # normalised form -> Counter of dieth spellings
    variants = defaultdict(Counter)

    for r in rows:
        doc = r["audio_id"].split("-")[0]
        docs[doc] += 1
        speakers[r["speaker_id"]] += 1
        for f in ("anonymity", "speech_in_speech", "missing_audio", "no_relevant_speech"):
            if r[f] == "1":
                flags[f] += 1

        dieth = r["transcription"].split()
        norm = r["normalized"].split()
        for tok in dieth:
            if tok in META_TOKENS:
                meta_token_count[tok] += 1
                continue
            dieth_tokens += 1
            char_freq.update(tok)

        # token-aligned pairs only where both layers have same length,
        # otherwise alignment is unsafe (splits/merges in normalisation)
        if len(dieth) == len(norm):
            for d, n in zip(dieth, norm):
                if d in META_TOKENS or n in META_TOKENS:
                    continue
                variants[n.lower()][d.lower()] += 1

    print(f"documents: {len(docs)}")
    print(f"speakers: {len(speakers)}")
    print(f"dieth tokens (excl. meta): {dieth_tokens}")
    print(f"meta tokens: {dict(meta_token_count)}")
    print(f"utterance flags set: {dict(flags)}")
    print(f"utterances per doc: min={min(docs.values())} max={max(docs.values())}")

    print("\n== character frequency table (Dieth layer, all chars) ==")
    total_chars = sum(char_freq.values())
    for ch, n in char_freq.most_common():
        name = unicodedata.name(ch, "?")
        ascii_flag = "" if ord(ch) < 128 else "  NON-ASCII"
        print(f"{ch!r}\t{n}\t{n/total_chars:.4%}\t{name}{ascii_flag}")

    print("\n== ambiguity of Dieth spellings ==")
    # restrict to normalised forms seen reasonably often, else hapax noise
    aligned_tokens = sum(sum(c.values()) for c in variants.values())
    ambiguous_types = 0
    ambiguous_tokens = 0
    n_variants_hist = Counter()
    for n_form, spellings in variants.items():
        n_variants_hist[min(len(spellings), 10)] += 1
        if len(spellings) > 1:
            ambiguous_types += 1
            ambiguous_tokens += sum(spellings.values())
    print(f"token-aligned pairs used: {aligned_tokens}")
    print(f"normalised types: {len(variants)}")
    print(f"types with >1 Dieth spelling: {ambiguous_types} "
          f"({ambiguous_types/len(variants):.1%})")
    print(f"tokens whose normalised form has >1 Dieth spelling: "
          f"{ambiguous_tokens} ({ambiguous_tokens/aligned_tokens:.1%})")
    print("variants-per-type histogram (10 = '>=10'):")
    for k in sorted(n_variants_hist):
        print(f"  {k}: {n_variants_hist[k]}")

    print("\n== examples of high-variant words ==")
    top = sorted(variants.items(), key=lambda kv: -len(kv[1]))[:15]
    for n_form, spellings in top:
        common = ", ".join(f"{s}({c})" for s, c in spellings.most_common(8))
        print(f"{n_form}: {len(spellings)} variants -> {common}")


if __name__ == "__main__":
    main(sys.argv[1])
