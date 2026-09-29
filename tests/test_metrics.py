import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import FlexWER, all_metrics, build_dieth_to_norm  # noqa: E402

ROWS = [
    {"text": "ich hät gsäit", "normalised": "ich hat gesagt"},
    {"text": "er het gsait", "normalised": "er hat gesagt"},
    {"text": "si hèt", "normalised": "sie hat"},
]


def test_table_groups_spellings_by_normalised_form():
    d2n = build_dieth_to_norm(ROWS)
    assert d2n["hät"] == d2n["het"] == d2n["hèt"] == {"hat"}
    assert d2n["gsäit"] == d2n["gsait"] == {"gesagt"}


def test_variant_spelling_is_free_but_real_error_is_not():
    d2n = build_dieth_to_norm(ROWS)
    m = all_metrics(["ich hät gsäit"], ["ich het gsait"], d2n)
    assert m["wer"] > 0.6 and m["flexwer"] == 0.0
    m = all_metrics(["ich hät gsäit"], ["ich het nüt"], d2n)
    assert m["flexwer"] == 1 / 3


def test_flexwer_counts_insertions_and_deletions():
    f = FlexWER(build_dieth_to_norm(ROWS))
    assert f.score(["ich hät gsäit"], ["ich gsäit"]) == 1 / 3
    assert f.score(["ich hät gsäit"], ["ich hät gsäit jaa"]) == 1 / 3


def test_unknown_words_fall_back_to_exact_match():
    f = FlexWER(build_dieth_to_norm(ROWS))
    assert f.same("zürich", "zürich") and not f.same("zürich", "züri")
