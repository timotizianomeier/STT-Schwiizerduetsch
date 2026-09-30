"""Extract the ArchiMob audio from the SwissUbase wrapper without writing the
20.8 GB inner zip to disk first.

swissubase_2277_1_0.zip stores archimob_r2_audio_share.zip *uncompressed*
(compress_type 0), so the inner archive is seekable in place and its
members can be read straight through. Peak disk = wrapper (19.4 GB) +
extracted wavs (23.5 GB) instead of + another 20.8 GB.

Run on the cluster (a login-node job this small is acceptable; it is I/O,
not compute, but use `nice`):
    nice python scripts/extract_audio.py \
        /vol/bitbucket/ttm25/stt/data/raw/swissubase_2277_1_0.zip \
        /vol/bitbucket/ttm25/stt/data/raw/audio
"""
import sys
import zipfile
from pathlib import Path


def main(wrapper: str, out_dir: str) -> None:
    outer = zipfile.ZipFile(wrapper)
    (entry,) = outer.infolist()
    assert entry.compress_type == 0, "inner zip is compressed; extract normally"
    with outer.open(entry) as f:
        inner = zipfile.ZipFile(f)
        members = inner.infolist()
        print(f"{len(members)} entries, {sum(m.file_size for m in members)/1e9:.1f} GB")
        for i, m in enumerate(members, 1):
            inner.extract(m, out_dir)
            if i % 5000 == 0:
                print(f"  {i}/{len(members)}", flush=True)
    print("done ->", Path(out_dir) / "audio_segmented_anonymized")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
