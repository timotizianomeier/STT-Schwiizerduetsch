"""Phase 6 metrics: CER, WER and FlexWER.

CER and WER come from `jiwer`. FlexWER is ours, following Kew's
`compute_flexwer.py` (VarDial 2020): a word-level edit distance in which a
substitution costs 0 when the hypothesis token is an *attested Dieth
spelling of the same word* as the reference token. "Same word" is defined
through the normalised layer: two Dieth tokens match if they share at least
one normalised form in the variant table.

    ref:  ich  hät   gsäit
    hyp:  ich  het   gsait      -> WER 2/3, FlexWER 0/3 if het∈V(hat), gsait∈V(gesagt)

The variant table is built from the corpus itself (every usable utterance,
all splits). That is a lexicon resource, not model training data, so using
test-set spellings in it does not leak anything into the model; it just makes
the metric tolerate every spelling a transcriber actually used. Kew built it
the same way. Words never seen in the table fall back to exact match.

Both edit distances are plain O(n·m) dynamic programming; utterances are
short, so this is fast enough and easy to read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import jiwer


# --- variant table ---------------------------------------------------------

def build_dieth_to_norm(rows: list[dict]) -> dict[str, set[str]]:
    """dieth spelling -> set of normalised forms it was aligned with.

    Only utterances where both layers have the same token count are used, so
    the zip is a true 1:1 alignment (our parser guarantees this for <w>
    elements, but we check anyway)."""
    d2n: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        d = r["text"].split()
        n = r["normalised"].lower().split()
        if len(d) != len(n):
            continue
        for dw, nw in zip(d, n):
            d2n[dw].add(nw)
    return d2n


def save_table(d2n: dict[str, set[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in d2n.items()}, f, ensure_ascii=False)


def load_table(path: str | Path) -> dict[str, set[str]]:
    with open(path, encoding="utf-8") as f:
        return {k: set(v) for k, v in json.load(f).items()}


# --- edit distance with a pluggable match ----------------------------------

def _edit_distance(ref: list[str], hyp: list[str], same) -> int:
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0 if same(ref[i - 1], hyp[j - 1]) else 1)
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, sub)
        prev = cur
    return prev[m]


class FlexWER:
    def __init__(self, d2n: dict[str, set[str]]):
        self.d2n = d2n

    def same(self, a: str, b: str) -> bool:
        if a == b:
            return True
        na, nb = self.d2n.get(a), self.d2n.get(b)
        return bool(na and nb and (na & nb))

    def score(self, refs: list[str], hyps: list[str]) -> float:
        errors = total = 0
        for r, h in zip(refs, hyps):
            rt, ht = r.split(), h.split()
            errors += _edit_distance(rt, ht, self.same)
            total += len(rt)
        return errors / total if total else 0.0


# --- the three numbers -------------------------------------------------------

def all_metrics(refs: list[str], hyps: list[str], d2n: dict[str, set[str]]) -> dict[str, float]:
    return {
        "cer": jiwer.cer(refs, hyps),
        "wer": jiwer.wer(refs, hyps),
        "flexwer": FlexWER(d2n).score(refs, hyps),
    }


if __name__ == "__main__":
    import sys
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    rows = [r for r in rows if r["usable"]]
    d2n = build_dieth_to_norm(rows)
    save_table(d2n, sys.argv[2])
    n2d = defaultdict(set)
    for d, ns in d2n.items():
        for n in ns:
            n2d[n].add(d)
    print(f"dieth spellings: {len(d2n)}  normalised forms: {len(n2d)}")
    print(f"dieth spellings mapping to >1 normalised form: {sum(len(v) > 1 for v in d2n.values())}")
    top = sorted(n2d.items(), key=lambda kv: -len(kv[1]))[:5]
    for n, ds in top:
        print(f"  {n}: {len(ds)} spellings, e.g. {sorted(ds)[:8]}")
