"""The wav file name -> XML media pointer mapping is the one place where a
silent mismatch would drop thousands of utterances without an error."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare import wav_index  # noqa: E402


def test_wav_index_handles_all_four_name_patterns(tmp_path):
    root = tmp_path / "audio_segmented_anonymized"
    files = {
        "1209/d1209_T821.wav": "d1209-T821",                      # plain
        "1082_2/d1082_2_T5.wav": "d1082_2-T5",                    # multi-part doc
        "1083_1/d1083_1_TLI_507.wav": "d1083_1-TLI_507",          # EXMARaLDA pointer
        "1082_3/1082_3d1082_3_TLI_12.wav": "d1082_3-TLI_12",      # doubled prefix
    }
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    index = wav_index(tmp_path)
    assert set(index) == set(files.values())
    assert index["d1082_3-TLI_12"].name == "1082_3d1082_3_TLI_12.wav"
