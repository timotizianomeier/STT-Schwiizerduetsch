"""Re-score saved predictions against the current references, without decoding.

When the normalisation policy changes (e.g. 2026-10-01: Dieth diacritics
folded), the references change but the model outputs do not. Decoding the
test set again costs 30 GPU-minutes; re-scoring costs a second.

Run: python src/rescore.py runs/x/predictions.jsonl \
         data/processed/archimob_utterances.jsonl data/processed/dieth_to_norm.json
"""
import json
import sys
from collections import defaultdict

from metrics import all_metrics, load_table
from normalise import normalise_hyp

preds = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
ref_of = {}
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l)
    ref_of[r["utt_id"]] = r["text"]
table = load_table(sys.argv[3])

refs = [ref_of[p["utt_id"]] for p in preds]
hyps = [normalise_hyp(p["hyp_raw"]) for p in preds]
out = {"utterances": len(preds), "overall": all_metrics(refs, hyps, table), "per_region": {}}
idx = defaultdict(list)
for i, p in enumerate(preds):
    idx[p["region"]].append(i)
for k, v in sorted(idx.items()):
    out["per_region"][k] = {**all_metrics([refs[i] for i in v], [hyps[i] for i in v], table), "n": len(v)}
print(json.dumps(out, indent=2))
