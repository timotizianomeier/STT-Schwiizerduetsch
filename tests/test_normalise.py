"""Tests pin the orthography policy (v2: Dieth diacritics folded) so a later
'small cleanup' cannot silently change training targets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalise import fold_diacritics, normalise_dieth, normalise_hyp, unexpected_chars, usable  # noqa: E402


def test_graves_tildes_acutes_fold_to_plain_letters():
    assert normalise_dieth("ìch hèt ùnd òòbe jà") == "ich het und oobe ja"
    assert normalise_dieth("õ ã ẽ ĩ") == "o a e i"
    assert normalise_dieth("kafé") == "kafe"


def test_umlauts_survive_the_fold():
    assert normalise_dieth("jänner über öppis") == "jänner über öppis"


def test_grave_on_umlaut_keeps_the_umlaut():
    assert normalise_dieth("mǜse") == "müse"                    # precomposed U+01DC
    assert normalise_dieth("chö̀ne") == "chöne"       # o + diaeresis + grave
    assert normalise_dieth("chö̀ne") == "chöne"             # ö + combining grave
    assert normalise_dieth("kapä́l") == "kapäl"       # ä + combining acute


def test_decomposed_input_gives_the_same_result_as_composed():
    assert normalise_dieth("ùnd") == normalise_dieth("ùnd") == "und"
    assert fold_diacritics("ä") == "ä"                     # recomposed, one codepoint


def test_output_alphabet_is_closed():
    out = normalise_dieth("Ìch hä̀t ǜber õ é (ja) <SIL_WORD>")
    assert not unexpected_chars(out), out


def test_strips_meta_tokens_and_parens():
    assert normalise_dieth("<SPOKEN_NOISE> s ìch (bin) <SIL_WORD> gibore") == "s ich bin gibore"


def test_lowercases_and_collapses_whitespace():
    assert normalise_dieth("  Jaa   Ìch\tbin ") == "jaa ich bin"


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


def test_normalise_hyp_strips_whisper_formatting_and_folds():
    assert normalise_hyp("Ja, ich weiß — das Haus!") == "ja ich weiss das haus"
    assert normalise_hyp("chö̀ne  ÌCH") == "chöne ich"
    assert normalise_hyp("Im Jahr 1917.") == "im jahr"
