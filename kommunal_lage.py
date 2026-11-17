"""Lage der kommunalen Messstellen im Gewässernetz: Abstand zur nächsten Mündung flussab (Rückstaugefahr).

    python kommunal_lage.py data/radar/dem/*N49*.tif data/radar/dem/*N50*.tif

Fließrichtung und Fläche wie catchments.py (Copernicus GLO-30). Jede Messstelle wird auf die Rasterzelle im Umkreis gesetzt,
deren Einzugsgebiet zwischen Unter- und Obergrenze liegt (Fläche des nächsten Landespegels flussab bzw. des Nebenbachs an
der Mündung), damit sie nicht auf den größeren Fluss nebenan springt. Dann flussab gehen und festhalten:
  - erste Mündung eines Zuflusses mit mindestens 30 % der eigenen Fläche (Rückstau möglich, wenn nah),
  - Mündung in einen größeren Fluss (eigene Fläche kleiner als die des anderen Asts; dort staut der größere zurück).
Dazu das Gefälle auf 2 km flussab und als grobes Maß der Rückstaulänge 2 m Wassertiefe geteilt durch das Gefälle.
Ausgabe: data/kommunal_lage.csv. Abstände sind Rasterwege (30 m), die Koordinaten teils nur Ortsmitten: Schätzung.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import catchments as C

# Obergrenze der Fläche je Messstelle (km2) und Suchradius (m)
UPPER = {"Hermespand": 53, "Niederprüm": 326, "Niedermehlen": 34, "Pittenbach": 326, "Habscheid": 58, "Kinzenburg": 326,
         "Mauel": 326, "Merkeshausen": 326, "Brecht": 490, "Bettingen": 490, "Holsthum Prüm": 490,
         "Neurath / Arzfeld": 101, "Neuerburg": 101, "Sinspelt Gemeindehaus": 101.5, "Enzen": 150, "Holsthum Schloss": 150,
         "Rommersheim": 17.7, "Nimsreuland": 137, "Rittersdorf": 265, "Dockendorf": 265}
# Untergrenze: Fläche des Landespegels flussauf (sonst 30 % der Obergrenze)
LOWER = {"Neurath / Arzfeld": 5, "Niederprüm": 53.4, "Pittenbach": 53.4, "Kinzenburg": 53.4, "Mauel": 53.4, "Merkeshausen": 53.4, "Brecht": 340,
         "Bettingen": 340, "Holsthum Prüm": 340, "Enzen": 101.5, "Holsthum Schloss": 101.5, "Nimsreuland": 17.8,
         "Rittersdorf": 137, "Dockendorf": 137}


def main(paths):
    N = pd.read_csv(C.HERE / "pegel_netz.csv", dtype=str).fillna("")
    dem, tr = C.load_dem(paths)
    parent, order, acc, area, filled = C.flow(dem, tr)
    fz = np.asarray(filled).ravel()
    H, W = dem.shape
    lat_r = tr.f + (np.arange(H) + 0.5) * tr.e
    dy = abs(tr.e) * np.pi / 180 * C.R_EARTH * 1000
    dxr = tr.a * np.pi / 180 * C.R_EARTH * 1000 * np.cos(np.radians(lat_r))
    accf = acc.ravel()
    par = np.asarray(parent).ravel()
    rows = []
    for _, r in N[N.art.isin(["kommunal", "LfU"])].iterrows():
        if r.art == "LfU":                                         # Landespegel zum Vergleich: amtliche Fläche ±15 %
            UPPER[r["name"]] = float(r.catchment_km2) * 1.15
            LOWER[r["name"]] = float(r.catchment_km2) * 0.85
        name = r["name"]
        if name not in UPPER:
            continue
        lat, lon = C.utm32_to_wgs84(float(r.easting), float(r.northing))
        rad = 1500 if "Ortskoordinate" in r.hinweis else 400
        inv = ~tr
        c0, r0 = inv * (lon, lat)
        r0, c0 = int(r0), int(c0)
        dr, dc = int(rad / dy) + 1, int(rad / dxr[min(max(r0, 0), H - 1)]) + 1
        best = None
        for i in range(r0 - dr, r0 + dr + 1):
            for j in range(c0 - dc, c0 + dc + 1):
                d = np.hypot((i - r0) * dy, (j - c0) * dxr[i])
                a = acc[i, j]
                if d <= rad and LOWER.get(name, 0.3 * UPPER[name]) <= a <= UPPER[name] and (best is None or d < best[3]):
                    best = (a, i, j, d)
        if best is None:
            print(f"{name}: keine passende Zelle im Umkreis {rad} m")
            continue
        a0, i, j, d0 = best
        k = i * W + j
        dist, first_trib, mouth = 0.0, None, None
        z0, z2k = fz[k], None
        while dist < 15000:
            p = par[k]
            if p < 0 or p == k:
                break
            pi, pj = divmod(p, W)
            ci, cj = divmod(k, W)
            dist += np.hypot((pi - ci) * dy, (pj - cj) * dxr[ci])
            if z2k is None and dist >= 2000:
                z2k = (z0 - fz[p]) / dist
            side = accf[p] - accf[k] - area.ravel()[p]
            if side >= 0.3 * accf[k] and first_trib is None:
                first_trib = (round(dist), round(side, 1), round(accf[k], 1))
            if side > accf[k]:
                mouth = (round(dist), round(side, 1), round(accf[k], 1))
                break
            k = p
        rows.append(dict(name=name, art=r.art, gewaesser=r.gewaesser, flaeche_km2=round(a0, 1), versatz_m=round(d0),
                         ortsmitte="Ortskoordinate" in r.hinweis,
                         zufluss_m=first_trib[0] if first_trib else None, zufluss_km2=first_trib[1] if first_trib else None,
                         eigen_dort_km2=first_trib[2] if first_trib else None,
                         muendung_m=mouth[0] if mouth else None, groesser_km2=mouth[1] if mouth else None,
                         gefaelle_promille=round(z2k * 1000, 1) if z2k is not None else None,
                         rueckstau_m=round(2.0 / z2k) if z2k else None))   # grob: 2 m Hochwassertiefe / Gefälle
    df = pd.DataFrame(rows)
    df.to_csv(C.HERE / "data" / "kommunal_lage.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
