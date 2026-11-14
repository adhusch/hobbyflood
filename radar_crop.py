#!/usr/bin/env python3
"""RADKLIM / RADOLAN RW: Ausschnitt Prüm/Enz/Nims aus einem Monatsarchiv schneiden.

Läuft auf dem eigenen Rechner nur mit der Python-Standardbibliothek. Das Archiv kommt als Datei oder per Pipe:

    curl -sf https://opendata.dwd.de/.../RW2017.002_202107.tar.gz | python3 radar_crop.py - 202107 AUSGABEORDNER

Je Stunde wird nur das Rechteck um die Einzugsgebiete behalten (rohe 2-Byte-Werte, Kennbits unverändert).
Ausgabe: crop_JJJJMM.bin.gz (je Stunde 10 Zeichen Zeitstempel JJMMTThhmm + Rohwerte) und crop_JJJJMM.json
(Raster, Ausschnitt, Kopfzeile der ersten Datei, Anzahl Stunden). Das Entschlüsseln (Kennbits, Genauigkeit)
passiert später in radar.py.
"""
import gzip
import json
import os
import re
import sys
import tarfile
import time

# Ausschnitt (Zeilen r0..r1-1 von Süden, Spalten c0..c1-1 von Westen), 49,80-50,40 N und 6,05-6,75 O
BOX = {(1100, 900): (416, 490, 144, 201), (900, 900): (316, 390, 224, 281)}
STAMP = re.compile(r"-(\d{10})-")


def parse(raw: bytes):
    i = raw.index(b"\x03")
    head = raw[:i].decode("latin-1")
    m = re.search(r"GP\s*(\d+)\s*x\s*(\d+)", head)
    nr, nc = int(m.group(1)), int(m.group(2))
    return head, nr, nc, raw[i + 1:]


def crop(data: bytes, nr: int, nc: int) -> bytes:
    r0, r1, c0, c1 = BOX[(nr, nc)]
    out = bytearray()
    for r in range(r0, r1):
        a = (r * nc + c0) * 2
        out += data[a:a + (c1 - c0) * 2]
    return bytes(out)


def members(tf):
    """Alle Binärdateien eines Archivs, auch in verschachtelten Tages-Archiven."""
    for m in tf:
        if not m.isfile():
            continue
        f = tf.extractfile(m)
        name = os.path.basename(m.name)
        if name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(fileobj=f, mode="r|*") as inner:
                yield from members(inner)
        elif name.endswith(".gz"):
            yield name[:-3], gzip.decompress(f.read())
        else:
            yield name, f.read()


def main(src: str, month: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    fin = sys.stdin.buffer if src == "-" else open(src, "rb")
    tmp = os.path.join(outdir, f"crop_{month}.bin.gz.part")
    n, first, grid, names, bad = 0, None, None, [], []
    with tarfile.open(fileobj=fin, mode="r|*") as tf, gzip.open(tmp, "wb", compresslevel=6) as out:
        for name, raw in members(tf):
            s = STAMP.search(name)
            if not s:
                bad.append(name)
                continue
            try:
                head, nr, nc, data = parse(raw)
                if len(data) < nr * nc * 2:
                    raise ValueError(f"zu kurz: {len(data)}")
                c = crop(data, nr, nc)
            except Exception as e:  # kaputte Einzeldatei: notieren, weiter
                bad.append(f"{name}: {e}")
                continue
            if first is None:
                first, grid = head, (nr, nc)
            elif (nr, nc) != grid:
                bad.append(f"{name}: Raster {nr}x{nc} statt {grid}")
                continue
            out.write(s.group(1).encode() + c)
            names.append(s.group(1))
            n += 1
    os.replace(tmp, os.path.join(outdir, f"crop_{month}.bin.gz"))
    r0, r1, c0, c1 = BOX[grid] if grid else (0, 0, 0, 0)
    meta = dict(month=month, hours=n, grid=grid, box=[r0, r1, c0, c1], header_first=first, first=min(names) if names else None,
                last=max(names) if names else None, sorted=names == sorted(names), problems=bad[:50], n_problems=len(bad),
                seconds=round(time.time() - t0, 1))
    with open(os.path.join(outdir, f"crop_{month}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("month", "hours", "grid", "first", "last", "n_problems", "seconds")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
