"""Phase 3: document-disjoint train/dev/test assignment.

Why by document and not by utterance or by speaker id?
  * Utterance-level splitting leaks: the same voice, room and microphone
    would be in train and test, and Whisper would look better than it is.
    (Kew's splits do this — doc 1007 appears in all three — so we don't
    reuse them.)
  * Speaker ids are document-scoped ("EJos1007"), so "by speaker" and "by
    document" are the same thing for the interviewees. Interviewers are NOT
    identified (`who="interviewer"`) and may recur across recordings; hence
    the rule below.

Rules:
  1. Every utterance of a document goes to that document's split.
  2. In dev/test, only the interviewee's utterances are evaluated
     (`eval_ok`). Interviewer / otherPerson utterances of held-out documents
     are dropped entirely: they share the recording conditions of the test
     speaker and their voice may also be in train.
  3. Held-out documents were chosen by hand (not randomly) to
       - cover the four biggest regions (ZH, BE, LU/AG, BS) so per-region
         numbers exist for the regions that matter, while leaving most of
         each region in train;
       - mix transcribers/tools (Peters+Nisus, Mächler+FOLKER,
         Aepli+EXMARaLDA), so the test set is not one annotator's habits;
       - be medium-sized single-file recordings without known anomalies
         (excluded: 1188 "ZH/BS ??", 1240 tiny, 1163 once misaligned).
     ZH appears in both dev and test because Züridütsch is the judge
     criterion in the brief.

Sizes are reported in utterances and words here; hours are filled in by
prepare.py once the audio is available.

Run: python src/splits.py data/processed/archimob_utterances.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

TEST_DOCS = {
    "1225": "ZH Zürich · Mächler/FOLKER · ~850 utts",
    "1121": "BE Köniz · Mächler/FOLKER · ~1000 utts",
    "1195": "LU Sempach · Peters/Nisus · ~1040 utts",
    "1263": "BS Basel · Aepli/EXMARaLDA · ~1390 utts",
}
DEV_DOCS = {
    "1055": "ZH Zürich · Aepli/EXMARaLDA phase 3 · ~840 utts",
    "1142": "BE Ittigen · Peters/Nisus, manual normalisation · ~660 utts",
    "1235": "LU Wolhusen · Aepli/EXMARaLDA phase 4 · ~800 utts",
}
MAIN_SPEAKER_EXCLUDED = {"interviewer", "otherPerson"}


def assign(row: dict) -> tuple[str | None, bool]:
    """Return (split, eval_ok). split is None when the row is dropped."""
    if not row["usable"]:
        return None, False
    doc = row["doc"]
    is_main = row["speaker"] not in MAIN_SPEAKER_EXCLUDED
    if doc in TEST_DOCS:
        return ("test", True) if is_main else (None, False)
    if doc in DEV_DOCS:
        return ("dev", True) if is_main else (None, False)
    return "train", is_main


def main(path: str) -> None:
    assert not (TEST_DOCS.keys() & DEV_DOCS.keys())
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        r["split"], r["eval_ok"] = assign(r)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = defaultdict(Counter)
    region = defaultdict(Counter)
    docs = defaultdict(set)
    for r in rows:
        s = r["split"] or "dropped"
        by[s]["utts"] += 1
        if r["split"]:
            by[s]["words"] += r["n_words"]
            docs[s].add(r["doc"])
            region[s][r["region"]] += 1
    for s in docs:
        by[s]["docs"] = len(docs[s])
    print(f"{'split':8}{'docs':>5}{'utts':>7}{'words':>8}")
    for s in ("train", "dev", "test", "dropped"):
        print(f"{s:8}{by[s]['docs']:5}{by[s]['utts']:7}{by[s]['words']:8}")
    total = sum(by[s]["words"] for s in ("train", "dev", "test"))
    for s in ("dev", "test"):
        print(f"{s}: {by[s]['words']/total:.1%} of words; regions {dict(region[s])}")
    print("train regions:", dict(region["train"].most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
