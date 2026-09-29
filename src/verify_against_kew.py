"""Cross-check our XML parse against Kew's archimob.csv (DECISIONS 2026-08-25
said: re-verify once the XML is downloaded). Join key: (doc_part, index in
file) — Kew's utt_id suffix is the 1-based utterance index within the file.

Run: python src/verify_against_kew.py data/processed/archimob_utterances.jsonl \
         third_party/two-headed-master/data/archimob.csv
"""
import csv, json, re, sys
from collections import Counter

META = re.compile(r"<[A-Z_]+>")

ours = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    ours[(r["doc_part"], r["idx_in_part"])] = r

kew = list(csv.DictReader(open(sys.argv[2], encoding="utf-8")))
print(f"ours: {len(ours)}  kew: {len(kew)}")

stats = Counter()
examples = []
for k in kew:
    m = re.match(r"(.+?)_[A-Z]-\1-(\d+)$", k["utt_id"])
    if not m:
        stats["unparseable_kew_id"] += 1
        continue
    key = (m.group(1), int(m.group(2)))
    o = ours.get(key)
    if o is None:
        stats["missing_in_ours"] += 1
        examples.append(("missing", k["utt_id"], k["transcription"]))
        continue
    stats["matched"] += 1
    if k["audio_id"] != o["audio_pointer"]:
        stats["audio_id_mismatch"] += 1
    kew_text = " ".join(META.sub(" ", k["transcription"]).split())
    if kew_text == o["raw"]:
        stats["text_equal"] += 1
    else:
        stats["text_differs"] += 1
        if len(examples) < 12:
            examples.append(("text", k["utt_id"], kew_text, o["raw"]))
    if (k["anonymity"] == "1") != o["anonymised"]:
        stats["anonymity_flag_differs"] += 1
    if (k["speech_in_speech"] == "1") != o["overlap"]:
        stats["overlap_flag_differs"] += 1
    if (k["no_relevant_speech"] == "1") != (o["n_words"] == 0):
        stats["empty_flag_differs"] += 1

print(dict(stats))
for e in examples[:12]:
    print(e)
