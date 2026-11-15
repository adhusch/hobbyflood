"""Flächiger Regen auf dem 1-km-Radarraster aus dem Stationsregen, zwei Produkte:

  A "raster"   Stationsregen interpoliert (Abstandsgewichtung, Höhenabhängigkeit aus den Stationen), über die Einzugsgebiete gemittelt
  B "merge"    Menge von den Stationen, Muster vom Radar: je Stunde der Faktor (Station + c) / (Radar + c) an den Stationsorten, interpoliert,
               auf das Radarfeld angewendet (c = 0,5 mm dämpft das Rauschen bei wenig Regen); wo das Radar fehlt, Produkt A

Ausgabe: data/radar/areal_raster.csv, data/radar/areal_merge.csv (mm je Stunde und Teilgebiet, Stempel wie areal_hourly.csv).
Bisher (rain_fit.py) ist der Gebietsregen der abstandsgewichtete Stationswert am Gebietsmittelpunkt, kein Flächenmittel.

  python grid_rain.py [Jahr ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import catchments as CA
import radar as RA
import rain_fit as RF

HERE = Path(__file__).parent
RD = HERE / "data" / "radar"
POWER, MIN_KM, C_MM = 2.0, 1.0, 0.5
F_CLIP = (0.25, 4.0)


def proj(lat, lon):
    """Polarstereographie des Radarrasters in km (wie catchments.radolan_index, ohne Abrunden)."""
    R = 6370.04
    phi0, lam0 = np.radians(60.0), np.radians(10.0)
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    m = (1 + np.sin(phi0)) / (1 + np.sin(phi))
    return R * m * np.cos(phi) * np.sin(lam - lam0) - CA.X0, -R * m * np.cos(phi) * np.cos(lam - lam0) - CA.Y0      # Spalte, Zeile (gebrochen)


def pixel_elevation() -> np.ndarray:
    """Mittlere Geländehöhe je Radarpixel des Ausschnitts (Copernicus GLO-30)."""
    p = RD / "pixel_elev.npy"
    if p.exists():
        return np.load(p)
    import glob
    dem, tr = CA.load_dem(sorted(glob.glob(str(RD / "dem" / "*.tif"))))
    r0, r1, c0, c1 = CA.BOX
    s, n = np.zeros((r1 - r0, c1 - c0)), np.zeros((r1 - r0, c1 - c0))
    for i in range(0, dem.shape[0], 200):
        rows = np.arange(i, min(i + 200, dem.shape[0]))
        lon = tr.c + (np.arange(dem.shape[1]) + 0.5) * tr.a
        lat = tr.f + (rows + 0.5) * tr.e
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        rr, cc = CA.radolan_index(LA, LO)
        ok = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1) & np.isfinite(dem[rows])
        np.add.at(s, (rr[ok] - r0, cc[ok] - c0), dem[rows][ok])
        np.add.at(n, (rr[ok] - r0, cc[ok] - c0), 1)
    el = s / np.maximum(n, 1)
    np.save(p, el)
    return el


def setup():
    st = pd.read_csv(HERE / "data" / "dwd_rr" / "stationen.csv", dtype={"id": str}).set_index("id")
    S = RF.load_stations(1.0)
    ids = [i for i in st.index if i in S]
    r0, r1, c0, c1 = CA.BOX
    sx, sy = proj(st.loc[ids, "lat"], st.loc[ids, "lon"])
    sx, sy = sx - c0, sy - r0                                        # Stationen in Pixelkoordinaten des Ausschnitts
    px, py = np.meshgrid(np.arange(c1 - c0) + 0.5, np.arange(r1 - r0) + 0.5)
    d = np.maximum(np.hypot(px.ravel()[:, None] - sx[None, :], py.ravel()[:, None] - sy[None, :]), MIN_KM)
    w = 1.0 / d ** POWER                                             # [Pixel, Stationen]
    # Höhenabhängigkeit: mittlerer Stundenregen je Station über die gemeinsamen Stunden gegen die Stationshöhe (lineare Regression)
    df = pd.concat({i: S[i] for i in ids}, axis=1)
    com = df.dropna()
    mean = com.mean()
    elev = st.loc[ids, "hoehe_m"].to_numpy(float)
    b, a = np.polyfit(elev, mean.to_numpy(), 1)
    el = pixel_elevation()
    clim_pix = (a + b * np.clip(el, elev.min(), elev.max())).ravel()
    clim_st = a + b * elev
    info = dict(stationen=ids, gemeinsame_stunden=int(len(com)), regen_je_100m=round(float(b * 100 / mean.mean()), 4),
                station_mittel={i: round(float(m * 8766), 0) for i, m in mean.items()}, hoehe={i: float(e) for i, e in zip(ids, elev)})
    return dict(S=df, ids=ids, w=w, clim_pix=clim_pix, clim_st=clim_st, sx=sx, sy=sy, shape=(r1 - r0, c1 - c0), info=info)


def year(key: str, G: dict, W: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t, val, _ = RA.read_crop(key)
    n = len(t)
    P = G["shape"][0] * G["shape"][1]
    st = G["S"].reindex(t).to_numpy(float)                            # [n, Stationen]
    ok = np.isfinite(st)
    norm = np.where(ok, st / G["clim_st"][None, :], 0.0)
    den = ok.astype(float) @ G["w"].T
    A = (norm @ G["w"].T) / np.maximum(den, 1e-12) * G["clim_pix"][None, :]
    A[den <= 0] = np.nan
    # B: Faktor an den Stationsorten (Radar im 3x3-Umfeld gemittelt)
    rad = val.reshape(n, P).astype(float)
    ix, iy = np.clip(G["sx"].astype(int), 1, G["shape"][1] - 2), np.clip(G["sy"].astype(int), 1, G["shape"][0] - 2)
    rs = np.stack([np.nanmean(val[:, y - 1:y + 2, x - 1:x + 2].reshape(n, 9), axis=1) for x, y in zip(ix, iy)], axis=1)   # [n, Stationen]
    okf = ok & np.isfinite(rs)
    f = np.where(okf, np.clip((st + C_MM) / (np.where(okf, rs, 0.0) + C_MM), *F_CLIP), 0.0)
    denf = okf.astype(float) @ G["w"].T
    F = (f @ G["w"].T) / np.maximum(denf, 1e-12)
    B = np.maximum((rad + C_MM) * F - C_MM, 0.0)
    B = np.where(np.isfinite(rad) & (denf > 0), B, A)
    out = []
    for X in (A, B):
        d = RA.areal(t, X.reshape(n, *G["shape"]).astype("float32"), W)
        out.append(d[[c for c in d.columns if not c.endswith("_max")]])
    return out[0], out[1]


def main(keys: list[str]) -> None:
    G = setup()
    W = RA.weights(json.load(open(RD / "catchments.json")))
    print(json.dumps(G["info"], ensure_ascii=False))
    a, b = [], []
    for k in keys or [k for k in RA.keys() if len(k) == 4 and 2004 <= int(k) <= 2022]:
        x, y = year(k, G, W)
        a.append(x); b.append(y)
        print(k, "A", x.sum().round(0).to_dict(), flush=True)
    for name, parts in (("raster", a), ("merge", b)):
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "time_mez"
        df.round(3).to_csv(RD / f"areal_{name}.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
