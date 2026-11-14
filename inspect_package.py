"""Zeigt Aufbau und Kopfzeilen einer heruntergeladenen Datei aus dem Download-Assistenten (zip oder Text), ohne sie ganz zu laden.

  python inspect_package.py <datei> [anzahl_dateien_im_zip=3] [zeilen=6]

Die Ausgabe (Namen, Größen, erste Zeilen, Trenner, Zeichensatz) genügt, um den Importer anzupassen. Es wird nichts verändert.
"""
import sys
import zipfile
from pathlib import Path


def decode(b: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "iso-8859-1"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1", "replace"), "latin-1?"


def head(raw: bytes, n: int) -> None:
    text, enc = decode(raw)
    lines = text.splitlines()[:n]
    sep = max(";,\t|", key=lambda c: sum(l.count(c) for l in lines))
    print(f"  Zeichensatz {enc}, vermuteter Trenner {sep!r}")
    for l in lines:
        print("  |", l[:200])


def main() -> None:
    p = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
            print(f"zip mit {len(infos)} Einträgen, unkomprimiert {sum(i.file_size for i in infos) / 1e6:.1f} MB")
            for i in infos[:25]:
                print(f"  {i.filename}  {i.file_size / 1e6:.2f} MB")
            if len(infos) > 25:
                print(f"  ... und {len(infos) - 25} weitere")
            for i in [i for i in infos if not i.is_dir()][:k]:
                print(f"\n== {i.filename}")
                with z.open(i) as f:
                    head(f.read(20000), n)
    else:
        with p.open("rb") as f:
            head(f.read(20000), n)


if __name__ == "__main__":
    main()
