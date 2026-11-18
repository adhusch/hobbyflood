"""Bodenfeuchte (DWD, AMBAV 2.0, Raster 1 km, täglich, % nFK unter Gras) als Tagesmittel je Teilgebiet.

    python bodenfeuchte.py <datei.nc> [...]       -> data/bodenfeuchte/bf_<schicht>.csv (Datum x Teilgebiet), vorhandene Tage werden ersetzt

Quelle: opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture/grass/<Jahr>/
grids_germany_daily_soil_moisture_grass_<Jahr>_<Schicht>_v1.nc (HDF5/NetCDF4, Variable paws in 0,1 % nFK, Gauß-Krüger Zone 3, EPSG:31467).
Teilgebiete aus data/radar/catchments.geojson (WGS84); Gewicht je Rasterzelle = Anteil der Zelle im Teilgebiet (10 x 10 Teilpunkte).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely import contains_xy

HERE = Path(__file__).parent
OUT = HERE / "data" / "bodenfeuchte"
SUB = 10


def weights(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Je Teilgebiet: Zeilen- und Spaltenindex (y, x) der Zellen und Gewicht (Flächenanteil)."""
    g = json.loads((HERE / "data" / "radar" / "catchments.geojson").read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", "EPSG:31467", always_xy=True)
    dx, dy = float(np.median(np.diff(x))), float(np.median(np.diff(y)))
    out = {}
    for f in g["features"]:
        geom = shape(f["geometry"])
        from shapely.ops import transform
        gk = transform(lambda a, b, z=None: tr.transform(a, b), geom)
        x0, y0, x1, y1 = gk.bounds
        ix = np.where((x >= x0 - abs(dx)) & (x <= x1 + abs(dx)))[0]
        iy = np.where((y >= y0 - abs(dy)) & (y <= y1 + abs(dy)))[0]
        off = (np.arange(SUB) + 0.5) / SUB - 0.5
        rows, cols, w = [], [], []
        for j in iy:
            for i in ix:
                px = x[i] + off[:, None] * dx + 0 * off[None, :]
                py = y[j] + 0 * off[:, None] + off[None, :] * dy
                frac = contains_xy(gk, px.ravel(), py.ravel()).mean()
                if frac > 0:
                    rows.append(j); cols.append(i); w.append(frac)
        out[f["properties"]["key"]] = (np.array(rows), np.array(cols), np.array(w))
    return out


def read(path: Path) -> tuple[str, pd.DataFrame]:
    m = re.search(r"_(\d{4})_(\d+-\d+)_v\d+\.nc$", path.name)
    year, layer = int(m.group(1)), m.group(2)
    with h5py.File(path, "r") as f:
        x, y = f["x"][:], f["y"][:]
        t = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(f["time"][:], unit="D")
        W = weights(x, y)
        rmin = min(int(r.min()) for r, _, _ in W.values()); rmax = max(int(r.max()) for r, _, _ in W.values())
        cmin = min(int(c.min()) for _, c, _ in W.values()); cmax = max(int(c.max()) for _, c, _ in W.values())
        d = f["paws"]
        block = d[:, rmin:rmax + 1, cmin:cmax + 1].astype(float)          # nur der Ausschnitt (Teilstücke der HDF5-Datei)
        fill = d.attrs.get("_FillValue", [-9999])[0]
        block[block == fill] = np.nan
        block *= float(d.attrs.get("scale_factor", [0.1])[0])
    res = {}
    for k, (r, c, w) in W.items():
        v = block[:, r - rmin, c - cmin]
        ok = np.isfinite(v)
        res[k] = (np.where(ok, v, 0) * w).sum(1) / np.maximum((ok * w).sum(1), 1e-9)
        res[k][(ok * w).sum(1) < 0.5 * w.sum()] = np.nan
    return layer, pd.DataFrame(res, index=t.normalize()).round(2)


def main(paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        layer, df = read(Path(p))
        f = OUT / f"bf_{layer}.csv"
        old = pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else None
        new = df if old is None else pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        new.to_csv(f)
        print(f"{Path(p).name}: {len(df)} Tage {df.index.min().date()} bis {df.index.max().date()} -> {f.name} ({len(new)} Tage)")


if __name__ == "__main__":
    main(sys.argv[1:])
