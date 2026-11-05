# v2 wishlist

Ideas, not commitments. v1 is closed (see RESULTS.md, DECISIONS 2026-11-05).
Order is not priority yet; Timo decides.

## From Timo

- **Zürich-specific STT.** ZH is the dialect that matters most to Timo. v1
  is already best on ZH (test CER 11.2 %, review quality 3.90/5, no
  Standard German drift), but the test set holds a single ZH speaker
  (doc 1225). ArchiMob Release 2 has 12 ZH documents (DATA.md); with one in
  dev and one in test, about 10 ZH interviews are in train. Open questions
  for a ZH-focused model: is a ZH-only (or ZH-weighted) fine-tune better
  than the four-region model on ZH, is ~10 interviews enough to train on
  at all, and how to evaluate on more than one held-out ZH speaker (a
  leave-one-document-out over the 12 ZH docs would do it, at the cost of
  several runs).

## From the v1 results

- **Word-level drift to Standard German forms in BS and LU** (*dr*→*de*/
  *der*, *sin*→*sind*, inserted *ja*). Options: more dialect-matched
  training data, a penalty or constrained decoding on Standard German
  forms, or accept it.
- **Misheard or dropped words** change meaning (Timo's main complaint).
  Options: longer training or higher encoder rank (the schedule was flat
  after step 1000, so more of the same steps probably won't help), more
  speakers, the Standard German encoder-adaptation stage that the brief
  put out of scope.
- **More than one speaker per region in the test set**, so region effects
  can be separated from speaker effects. Needs a cross-validation over
  speakers, not a single split.
- **Short-clip hallucinations** (6 utterances above 100 % CER): length
  filter or a no-speech threshold at inference.

(Add yours here.)
