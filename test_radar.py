"""Unit-Test (synthetische Daten, nur zur Prüfung der Dateiverarbeitung): radar_crop.py schneidet aus verschachtelten
Archiven richtig aus, radar.read_crop entschlüsselt Kennbits und Zeitstempel richtig."""
import gzip
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

import radar as RA
import radar_crop as RC

HERE = Path(__file__).parent


def fake_file(stamp: str, grid=(1100, 900), seed=0) -> tuple[bytes, np.ndarray]:
    nr, nc = grid
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 300, size=(nr, nc)).astype("<u2")
    v[0, 0] |= 0x2000
    r0, r1, c0, c1 = RC.BOX[grid]
    v[r0 + 3, c0 + 4] = 0x2000 | 0x0FFF          # Fehlkennung im Ausschnitt
    v[r0 + 5, c0 + 6] = 0x8000 | 12              # Clutter
    v[r0 + 7, c0 + 8] = 0x1000 | 25              # interpoliert -> gültig, 2,5 mm
    head = f"RW{stamp[4:10]}10000{stamp[2:4]}{stamp[0:2]}BY{nr * nc * 2 + 100:7d}VS 3SW   2.28.1PR E-01INT  60GP{nr:4d}x{nc:4d}MS 10<x>".encode()
    return head + b"\x03" + v.tobytes(), v


def tar_bytes(files: dict[str, bytes], gz: bool) -> bytes:
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz" if gz else "w") as t:
        for n, d in files.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(d)
            t.addfile(ti, io.BytesIO(d))
    return b.getvalue()


def test_crop_roundtrip():
    stamps = ["2107140050", "2107140150", "2107150050"]
    raws, vals = {}, {}
    for i, s in enumerate(stamps):
        raws[s], vals[s] = fake_file(s, seed=i)
    day1 = tar_bytes({f"raa01-rw2017.002_10000-{s}-dwd---bin": raws[s] for s in stamps[:2]}, gz=True)
    day2 = tar_bytes({f"raa01-rw2017.002_10000-{stamps[2]}-dwd---bin.gz": gzip.compress(raws[stamps[2]])}, gz=True)
    month = tar_bytes({"RW2017.002_20210715.tar.gz": day2, "RW2017.002_20210714.tar.gz": day1}, gz=True)
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([sys.executable, str(HERE / "radar_crop.py"), "-", "202107", d], input=month, capture_output=True, check=True)
        info = json.loads(p.stdout)
        assert info["hours"] == 3 and info["n_problems"] == 0, info
        t, val, meta = RA.read_crop("202107", Path(d))
    r0, r1, c0, c1 = RC.BOX[(1100, 900)]
    assert [str(x) for x in t] == ["2021-07-14 02:00:00", "2021-07-14 03:00:00", "2021-07-15 02:00:00"], t
    for k, s in enumerate(stamps):
        exp = (vals[s][r0:r1, c0:c1] & 0x0FFF).astype(float) * 0.1
        exp[3, 4] = np.nan
        exp[5, 6] = np.nan
        got = val[k]
        assert np.allclose(np.nan_to_num(got, nan=-1), np.nan_to_num(exp, nan=-1), atol=1e-5), s
        assert abs(got[7, 8] - 2.5) < 1e-6
    # Gebietsmittel mit Umgewichtung um fehlende Pixel
    W = {"A": np.zeros((r1 - r0, c1 - c0))}
    W["A"][3, 4] = 1.0
    W["A"][3, 5] = 3.0
    df = RA.areal(t, val, W, min_cover=0.5)
    assert np.allclose(df["A"], val[:, 3, 5])
    df = RA.areal(t, val, W, min_cover=0.9)
    assert df["A"].isna().all()
    print("radar ok")


if __name__ == "__main__":
    test_crop_roundtrip()
