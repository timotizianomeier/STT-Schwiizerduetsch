"""Orthography handling for the ArchiMob Dieth layer.

Policy v2, decided by Timo on 2026-10-01 after reading the smoke samples
(v1, signed off 2026-09-29, kept all Dieth diacritics; see DECISIONS.md):

  * **Fold the Dieth phonetic diacritics to plain letters.** Grave (open
    vowel: à ì è ò ù ǜ, and ö + combining grave), tilde (nasal: õ ã ẽ ĩ) and
    the rare acutes (é ó á í ú) all lose their mark. Umlauts ä ö ü stay:
    they are different letters, not annotations. The target alphabet is
    therefore exactly a-z + ä ö ü + space — what a Swiss German speaker
    actually types. Why: nobody writes the graves in practice, transcribers
    apply them inconsistently, and the smoke model dropped most of them,
    paying CER for a distinction the end use does not want.
    How: decompose (NFD), drop the combining grave/acute/tilde, recompose
    (NFC). The diaeresis is a different combining mark and survives.
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

# The whole target alphabet after folding. `unexpected_chars` reports anything
# else so a new character can never silently become a training target.
ALLOWED = set("abcdefghijklmnopqrstuvwxyzäöü ")

# Combining marks removed by the fold: grave, acute, tilde. NOT U+0308
# (diaeresis), which is what makes ä ö ü.
_FOLD = {"\u0300", "\u0301", "\u0303"}


def fold_diacritics(text: str) -> str:
    """ì -> i, ǜ -> ü, õ -> o, é -> e ... while keeping ä ö ü intact."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _FOLD))


def normalise_dieth(text: str) -> str:
    """Normalise one Dieth utterance string according to the v1 policy."""
    text = META_TOKEN_RE.sub(" ", text)   # before lower(): the tokens are upper-case
    text = fold_diacritics(text.lower())
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


# --- hypothesis side -------------------------------------------------------

_NOT_LETTER_RE = re.compile(r"[^\w\s]|[\d_]")


def normalise_hyp(text: str) -> str:
    """Normalise a model output so it is scored on the same footing as the
    references (which are lowercase, punctuation-free Dieth).

    Whisper emits capitals, punctuation and ß; none of those exist in the
    references, and counting them as errors would say nothing about dialect.
    So: lowercase, ß -> ss (Swiss usage), the same diacritic fold as the
    references, drop every character that is not a letter or whitespace,
    collapse whitespace.

    Deliberately NOT done: spelling out digits ("1917"). The references spell
    numbers as words; a model that emits digits is penalised, and that is a
    real difference in output style worth seeing in the error rate.
    """
    text = fold_diacritics(text.lower().replace("ß", "ss"))   # same fold as the references
    text = _NOT_LETTER_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()
