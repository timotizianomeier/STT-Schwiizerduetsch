"""Orthography handling for the ArchiMob Dieth layer (Phase 2 policy).

Policy signed off 2026-09-29 (see notes/DECISIONS.md):

  * Unicode NFC. The corpus contains 759 decomposed combining graves
    (U+0300), e.g. "schpö̀ö̀ter". NFC composes what it can (e.g. "u" + grave
    -> "ù") but there is NO precomposed "ö with grave" in Unicode, so "ö̀"
    stays a two-codepoint sequence after NFC. That is fine for a byte-level
    BPE tokenizer; it just means CER counts it as two characters.
  * Keep umlauts (ä ö ü), Dieth graves (ì è ò ù ǜ = open vowel quality) and
    tildes (õ ã ẽ ĩ = nasal vowel). They carry meaning in Dieth.
  * Keep the ~130 acute-accented letters (é ó á í ú) as they are. They are
    mostly loanwords; too rare to matter, and folding would be a guess.
  * Drop the single stray COMBINING ACUTE (U+0301, "kapä́l" in d1207-u231):
    annotator noise, no Dieth meaning.
  * "à" (8 tokens, e.g. "jà", "wàr") is a grave on a: same open-vowel
    convention as ì/è/ò/ù, just rare. Kept.
  * Lowercase (the corpus already is; enforced anyway).
  * Strip the 5 stray parentheses.
  * Strip Kaldi-style meta tokens (<SPOKEN_NOISE>, <SIL_WORD>, <NOISE>) if
    the input comes from Kew's CSV rather than our own XML parse.
  * Collapse whitespace.

What this module deliberately does NOT do: decide whether an utterance is
usable. That is `usable()` below, which looks at the structural flags the
parser produced (anonymisation, overlap, gaps ...), not at the string.
"""

from __future__ import annotations

import re
import unicodedata

META_TOKEN_RE = re.compile(r"<[A-Z_]+>")
WS_RE = re.compile(r"\s+")

# Every character we expect to see in a normalised Dieth string. Anything
# outside this set is reported by `unexpected_chars` so it can be looked at
# instead of silently becoming a training target.
BASE = set("abcdefghijklmnopqrstuvwxyz")
UMLAUT = set("äöü")
GRAVE = set("àìèòùǜ")          # ǜ = U+01DC, precomposed ü-with-grave
TILDE = set("õãẽĩ")
ACUTE = set("éóáíú")
COMBINING_GRAVE = "\u0300"     # survives NFC only on ö (no precomposed form)
COMBINING_ACUTE = "\u0301"     # 1 occurrence in the corpus, removed
ALLOWED = BASE | UMLAUT | GRAVE | TILDE | ACUTE | {COMBINING_GRAVE, " "}


def normalise_dieth(text: str) -> str:
    """Normalise one Dieth utterance string according to the v1 policy."""
    text = META_TOKEN_RE.sub(" ", text)   # before lower(): the tokens are upper-case
    text = unicodedata.normalize("NFC", text)
    text = text.replace(COMBINING_ACUTE, "")
    text = text.lower()
    text = text.replace("(", "").replace(")", "")
    text = WS_RE.sub(" ", text).strip()
    return text


def unexpected_chars(text: str) -> set[str]:
    """Characters in `text` that the policy does not account for."""
    return {c for c in text if c not in ALLOWED}


def usable(row: dict) -> tuple[bool, str]:
    """Decide whether a parsed utterance row is a valid training/eval target.

    Returns (ok, reason). Order matters only for the reason reported; an
    utterance failing several checks is counted once, under the first.
    Reasons are stable strings so they can be tallied.
    """
    if row["anonymised"]:
        return False, "anonymised"        # contains *** redaction masks
    if row["overlap"]:
        return False, "overlap"           # shares an audio chunk with another utterance
    if row["n_gap"] > 0:
        return False, "gap"               # unintelligible speech with no text
    if not row["text"]:
        return False, "empty"             # nothing left after stripping markers
    return True, "ok"
