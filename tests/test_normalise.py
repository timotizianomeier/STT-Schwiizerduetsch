"""Tests pin the Phase 2 orthography policy so a later 'small cleanup' cannot
silently change training targets. Each test names the rule it guards."""
import sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalise import normalise_dieth, unexpected_chars, usable  # noqa: E402


def test_nfc_composes_decomposed_grave():
    decomposed = "ùnd"                      # u + combining grave
    assert normalise_dieth(decomposed) == "ùnd"
    assert len(normalise_dieth(decomposed)) == 3


def test_o_umlaut_with_grave_stays_two_codepoints():
    # There is no precomposed ö-with-grave; NFC leaves ö + U+0300.
    s = normalise_dieth("chö̀ne")
    assert unicodedata.normalize("NFC", s) == s
    assert "̀" in s
    assert not unexpected_chars(s)


def test_keeps_dieth_diacritics():
    for w in ["ìch", "hèt", "òòbe", "ùnd", "mǜse", "õ", "ẽ", "jänner", "über", "öppis"]:
        assert normalise_dieth(w) == w
        assert not unexpected_chars(w), w


def test_strips_meta_tokens_and_parens():
    assert normalise_dieth("<SPOKEN_NOISE> s ìch (bin) <SIL_WORD> gibore") == "s ìch bin gibore"


def test_lowercases_and_collapses_whitespace():
    assert normalise_dieth("  Jaa   Ìch\tbin ") == "jaa ìch bin"


def _row(**kw):
    base = dict(anonymised=False, overlap=False, n_gap=0, text="jaa")
    base.update(kw)
    return base


def test_usable_order_of_reasons():
    assert usable(_row()) == (True, "ok")
    assert usable(_row(anonymised=True))[1] == "anonymised"
    assert usable(_row(overlap=True))[1] == "overlap"
    assert usable(_row(n_gap=1))[1] == "gap"
    assert usable(_row(text=""))[1] == "empty"


def test_drops_combining_acute_but_keeps_a_grave():
    assert normalise_dieth("kapä́l") == "kapäl"     # ä + combining acute
    assert normalise_dieth("jà") == "jà" and not unexpected_chars("jà")
