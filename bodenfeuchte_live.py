"""Bodenfeuchte im Live-Betrieb: die letzten Tage aus der aktuellen DWD-Jahresdatei, nur die nötigen Kacheln über HTTP-Teilabrufe.

    python bodenfeuchte_live.py            -> ergänzt data/live/bodenfeuchte.csv (Datum x Teilgebiet, % nFK, 0-60 cm unter Gras)

Die Jahresdatei (grids_germany_daily_soil_moisture_grass_<Jahr>_0-60_v1.nc, rund 125 MB, HDF5) ist in Kacheln zu 107 Tagen x 289 x 218 Zellen
gespeichert; unser Gebiet liegt in einer räumlichen Kachel. h5py liest über RangeFile nur Kopf, Index und die Kachel(n) der gewünschten Tage
(einige MB). Zellen und Flächenanteile je Teilgebiet aus data/bodenfeuchte/gewichte.json (bodenfeuchte.weights), also ohne pyproj/shapely.
Ausfallsicher: Jeder Fehler lässt die Tabelle unverändert; live.py schreibt dann den letzten Wert höchstens 3 Tage fort, danach Ersatzwert.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture/grass/{y}/grids_germany_daily_soil_moisture_grass_{y}_0-60_v1.nc"
OUT = HERE / "data" / "live" / "bodenfeuchte.csv"
DAYS = 14                     # so viele Tage am Ende der Datei lesen
BLOCK = 1 << 20               # Teilabrufe in 1-MB-Blöcken


class RangeFile:
    """Nur-Lese-Datei über HTTP-Teilabrufe (Range), mit Blockspeicher; für h5py.File(fileobj)."""

    def __init__(self, url: str, timeout: float = 30.0, ua: str = "hobbyflood"):
        self.url, self.timeout, self.ua, self.pos, self.cache, self.bytes = url, timeout, ua, 0, {}, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Length"])

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            a, b = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"User-Agent": self.ua, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if r.status != 206:
                    raise RuntimeError(f"Server liefert keine Teilabrufe (HTTP {r.status})")
                self.cache[i] = r.read()
            self.bytes += len(self.cache[i])
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out, p, end = bytearray(), self.pos, self.pos + n
        while p < end:
            i = p // BLOCK
            blk = self._block(i)
            a = p - i * BLOCK
            take = min(end - p, len(blk) - a)
            out += blk[a:a + take]
            p += take
        self.pos = end
        return bytes(out)

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def area_means(f, days: int = DAYS) -> pd.DataFrame:
    """Tagesmittel je Teilgebiet für die letzten `days` Tage aus einer geöffneten Datei (h5py.File)."""
    G = json.loads((HERE / "data" / "bodenfeuchte" / "gewichte.json").read_text(encoding="utf-8"))["teilgebiete"]
    rmin = min(min(v["zeile"]) for v in G.values()); rmax = max(max(v["zeile"]) for v in G.values())
    cmin = min(min(v["spalte"]) for v in G.values()); cmax = max(max(v["spalte"]) for v in G.values())
    d = f["paws"]
    nt = d.shape[0]
    t0 = max(0, nt - days)
    units = f["time"].attrs["units"]
    units = units.decode() if isinstance(units, bytes) else str(units)
    start = pd.Timestamp(units.split("since")[1].strip()[:10])
    days_idx = start + pd.to_timedelta(f["time"][t0:nt], unit="D")
    block = d[t0:nt, rmin:rmax + 1, cmin:cmax + 1].astype(float)
    fill = d.attrs.get("_FillValue", [-9999])[0]
    block[block == fill] = np.nan
    block *= float(d.attrs.get("scale_factor", [0.1])[0])
    res = {}
    for k, v in G.items():
        r, c, w = np.array(v["zeile"]) - rmin, np.array(v["spalte"]) - cmin, np.array(v["gewicht"])
        x = block[:, r, c]
        ok = np.isfinite(x)
        val = (np.where(ok, x, 0) * w).sum(1) / np.maximum((ok * w).sum(1), 1e-9)
        val[(ok * w).sum(1) < 0.5 * w.sum()] = np.nan
        res[k] = val
    return pd.DataFrame(res, index=pd.DatetimeIndex(days_idx).normalize()).round(2)


def update(now: pd.Timestamp | None = None, url_fmt: str = BASE) -> str:
    """Neue Tage holen, wenn der letzte Tag in der Tabelle älter als gestern ist. Rückgabe: kurze Meldung fürs Protokoll."""
    import h5py
    now = now or pd.Timestamp.now()
    old = pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else None
    yesterday = now.normalize() - pd.Timedelta(days=1)
    if old is not None and len(old) and old.index.max() >= yesterday:
        return f"aktuell bis {old.index.max().date()}"
    t = time.time()
    frames, info = [], []
    years = [now.year] if now.dayofyear > 7 else [now.year - 1, now.year]  # Jahreswechsel: die neue Datei gibt es erst nach einigen Tagen
    for y in years:
        try:
            rf = RangeFile(url_fmt.format(y=y))
            with h5py.File(rf, "r") as f:
                frames.append(area_means(f))
            info.append(f"{y}: {rf.bytes / 1e6:.1f} MB")
        except Exception as e:                                     # noqa: BLE001
            info.append(f"{y}: {e!r}"[:120])
    if not frames:
        return "nicht verfügbar (" + "; ".join(info) + ")"
    new = pd.concat(frames)
    new = new[new.notna().all(axis=1)]
    tab = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT)
    return f"bis {new.index.max().date()} ({', '.join(info)}, {time.time() - t:.0f} s)"


if __name__ == "__main__":
    print("Bodenfeuchte:", update())
