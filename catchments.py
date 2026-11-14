"""Einzugsgebiete der acht Pegel aus dem Copernicus-Höhenmodell (GLO-30) und ihre Anteile an den Radarpixeln.

    python catchments.py DEM_N49.tif DEM_N50.tif

1. Kacheln zusammensetzen (gemeinsames Raster 1,5" x 1", Ausschnitt 49,80-50,40 N, 6,05-6,75 O).
2. Senken vom Rand her füllen (Priority-Flood, eigene Umsetzung), D8-Fließrichtung, Fließakkumulation als Fläche in km2.
3. Pegel auf die Rasterzelle im Umkreis von 400 m setzen, deren Einzugsgebiet am besten zur amtlichen Fläche
   (pegel_netz.csv) passt, Einzugsgebiet abgrenzen.
4. Teilgebiete der Regenrechnung (Oberläufe P2, SI, GI und Zwischengebiete EC-P2, PZ-WI-SI, SE-GI, AL-SE, dazu WI-EC)
   und je Teilgebiet die Fläche in jedem 1-km-Radarpixel (RADOLAN-Raster 1100x900, Ausschnitt wie radar_crop.py).

Ausgabe: data/radar/catchments.json (Flächen, Pixelanteile) und data/radar/catchments.geojson (Umrisse für die Karte).
"""
from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin

HERE = Path(__file__).parent
OUT = HERE / "data" / "radar"
LON0, LON1, LAT0, LAT1 = 6.05, 6.75, 49.80, 50.40
RES = (1.5 / 3600, 1.0 / 3600)            # Grad (lon, lat)
R_EARTH = 6371.0088
BOX = (416, 490, 144, 201)               # wie radar_crop.py, Raster 1100x900
X0, Y0 = -443.46216692, -4758.64472427   # linke untere Ecke des 1100x900-Rasters in km

GAUGES = {"P2": "Prüm 2", "EC": "Echtershausen", "WI": "Wiersdorf", "PZ": "Prümzurlay", "SI": "Sinspelt",
          "GI": "Giesdorf", "SE": "Seffern", "AL": "Alsdorf-Oberecken"}
# Teilgebiete der Regenrechnung: Schlüssel -> (Pegel, abzuziehende Pegel)
PARTS = {"P2": ("P2", ()), "SI": ("SI", ()), "GI": ("GI", ()), "EC": ("EC", ("P2",)), "WI": ("WI", ("EC",)),
         "PZ": ("PZ", ("WI", "SI")), "SE": ("SE", ("GI",)), "AL": ("AL", ("SE",))}


def utm32_to_wgs84(e: float, n: float) -> tuple[float, float]:
    from pyproj import Transformer
    t = Transformer.from_crs(25832, 4326, always_xy=True)
    lon, lat = t.transform(e, n)
    return lat, lon


def load_dem(paths: list[str]) -> tuple[np.ndarray, rasterio.Affine]:
    srcs = [rasterio.open(p) for p in paths]
    arr, tr = merge(srcs, bounds=(LON0, LAT0, LON1, LAT1), res=RES, resampling=rasterio.enums.Resampling.bilinear, nodata=-9999)
    return arr[0].astype("float64"), tr


def cell_area_km2(tr, shape) -> np.ndarray:
    lat = tr.f + (np.arange(shape[0]) + 0.5) * tr.e
    dy = abs(tr.e) * np.pi / 180 * R_EARTH
    dx = tr.a * np.pi / 180 * R_EARTH * np.cos(np.radians(lat))
    return np.repeat((dx * dy)[:, None], shape[1], axis=1)


@numba.njit(cache=True)
def _priority_flood(dem, dx, dy):
    """Priority-Flood (Barnes et al. 2014): füllt Senken vom Rand her und liefert für jede Zelle die Unterliegerzelle.
    Rückgabe: Unterlieger (flacher Index, -1 am Rand), Reihenfolge (Unterlieger stets vor Oberlieger), gefüllte Höhe.
    Richtung: steilstes Gefälle zu einer tieferen gefüllten Nachbarzelle; in Ebenen die Zelle, von der aus geflutet wurde."""
    nr, nc = dem.shape
    n = nr * nc
    filled = dem.copy().ravel()
    parent = np.full(n, -1, np.int64)
    order = np.empty(n, np.int64)
    done = np.zeros(n, np.bool_)
    heap = [(0.0, 0)]
    heap.pop()
    for r in range(nr):
        for c in range(nc):
            if r == 0 or c == 0 or r == nr - 1 or c == nc - 1:
                i = r * nc + c
                heapq.heappush(heap, (filled[i], i))
                done[i] = True
    k = 0
    while len(heap) > 0:
        z, i = heapq.heappop(heap)
        order[k] = i
        k += 1
        r, c = i // nc, i % nc
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                    continue
                j = rr * nc + cc
                if done[j]:
                    continue
                done[j] = True
                parent[j] = i
                if filled[j] < z:
                    filled[j] = z
                heapq.heappush(heap, (filled[j], j))
    # steilstes Gefälle, wo es eine tiefere Nachbarzelle gibt
    for i in range(n):
        if parent[i] < 0:
            continue
        r, c = i // nc, i % nc
        best, bj = 0.0, -1
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                    continue
                j = rr * nc + cc
                d = np.sqrt((dr * dy) ** 2 + (dc * dx[r]) ** 2)
                s = (filled[i] - filled[j]) / d
                if s > best:
                    best, bj = s, j
        if bj >= 0:
            parent[i] = bj
    return parent, order, filled


@numba.njit(cache=True)
def _accumulate(parent, order, w):
    acc = w.copy()
    for k in range(len(order) - 1, -1, -1):
        i = order[k]
        p = parent[i]
        if p >= 0:
            acc[p] += acc[i]
    return acc


@numba.njit(cache=True)
def _upstream(parent, order, outlet):
    m = np.zeros(len(parent), np.bool_)
    m[outlet] = True
    for k in range(len(order)):
        i = order[k]
        p = parent[i]
        if p >= 0 and m[p]:
            m[i] = True
    return m


def flow(dem: np.ndarray, tr) -> tuple:
    area = cell_area_km2(tr, dem.shape)
    lat = tr.f + (np.arange(dem.shape[0]) + 0.5) * tr.e
    dy = abs(tr.e) * np.pi / 180 * R_EARTH * 1000
    dx = tr.a * np.pi / 180 * R_EARTH * 1000 * np.cos(np.radians(lat))
    parent, order, filled = _priority_flood(dem, dx, dy)
    acc = _accumulate(parent, order, area.ravel()).reshape(dem.shape)
    return parent, order, acc, area, filled.reshape(dem.shape)


def snap(acc: np.ndarray, tr, lat: float, lon: float, target: float, radius_m: float = 400) -> tuple[int, int, float]:
    inv = ~tr
    c, r = inv * (lon, lat)
    r, c = int(r), int(c)
    dr = int(np.ceil(radius_m / (abs(tr.e) * np.pi / 180 * R_EARTH * 1000)))
    dc = int(np.ceil(radius_m / (tr.a * np.pi / 180 * R_EARTH * 1000 * np.cos(np.radians(lat)))))
    best = None
    for i in range(r - dr, r + dr + 1):
        for j in range(c - dc, c + dc + 1):
            y = tr.f + (i + 0.5) * tr.e
            x = tr.c + (j + 0.5) * tr.a
            d = np.hypot((y - lat) * 111195, (x - lon) * 111195 * np.cos(np.radians(lat)))
            if d > radius_m:
                continue
            err = abs(acc[i, j] / target - 1)
            score = err + d / radius_m * 0.05
            if best is None or score < best[0]:
                best = (score, i, j, acc[i, j], d)
    return best[1], best[2], best[3], best[4]


def radolan_index(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zeile/Spalte (von Süden/Westen) im 1100x900-Raster; Polarstereographie, wahr bei 60 N, Mittelmeridian 10 O."""
    R = 6370.04
    phi0 = np.radians(60.0)
    lam0 = np.radians(10.0)
    phi, lam = np.radians(lat), np.radians(lon)
    m = (1 + np.sin(phi0)) / (1 + np.sin(phi))
    x = R * m * np.cos(phi) * np.sin(lam - lam0)
    y = -R * m * np.cos(phi) * np.cos(lam - lam0)
    return np.floor(y - Y0).astype(int), np.floor(x - X0).astype(int)


def polygons(mask: np.ndarray, tr) -> list:
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    geoms = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=tr) if v == 1]
    u = unary_union(geoms).simplify(0.0008, preserve_topology=True)
    return mapping(u)


def main(paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    N = pd.read_csv(HERE / "pegel_netz.csv", dtype=str).fillna("").set_index("name")
    dem, tr = load_dem(paths)
    print("DEM", dem.shape, "fehlend", int((dem == -9999).sum()))
    parent, order, acc, area, filled = flow(dem, tr)
    print("gefüllt:", int((filled > dem + 0.5).sum()), "Zellen, größte Fülltiefe", round(float((filled - dem).max()), 1), "m")
    masks, info = {}, {}
    for k, name in GAUGES.items():
        lat, lon = utm32_to_wgs84(float(N.loc[name, "easting"]), float(N.loc[name, "northing"]))
        target = float(N.loc[name, "catchment_km2"])
        i, j, a, d = snap(acc, tr, lat, lon, target)
        cm = _upstream(parent, order, i * dem.shape[1] + j).reshape(dem.shape)
        masks[k] = cm
        A = float((np.asarray(area) * cm).sum())
        info[k] = dict(name=name, amtlich_km2=target, dem_km2=round(A, 2), abw_proz=round((A / target - 1) * 100, 1),
                       versatz_m=round(d), lat=round(lat, 5), lon=round(lon, 5))
        print(f"{k:3s} {name:18s} amtlich {target:7.2f}  DEM {A:7.2f} km2 ({(A / target - 1) * 100:+.1f} %), Versatz {d:.0f} m")
    # Radarpixel je DEM-Zelle
    rows, cols = np.indices(dem.shape)
    lat = tr.f + (rows + 0.5) * tr.e
    lon = tr.c + (cols + 0.5) * tr.a
    rr, cc = radolan_index(lat, lon)
    r0, r1, c0, c1 = BOX
    inside = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    assert all(inside[m].all() for m in masks.values()), "Ausschnitt zu klein"
    rr, cc = np.clip(rr, r0, r1 - 1), np.clip(cc, c0, c1 - 1)
    lin = (rr - r0) * (c1 - c0) + (cc - c0)
    A = np.asarray(area)
    parts, feats = {}, []
    for k, (up, minus) in PARTS.items():
        m = masks[up].copy()
        for mm in minus:
            m &= ~masks[mm]
        w = np.bincount(lin[m], weights=A[m], minlength=(r1 - r0) * (c1 - c0))
        nz = np.nonzero(w)[0]
        parts[k] = dict(area_km2=round(float(w.sum()), 3), pixels=[[int(p // (c1 - c0)), int(p % (c1 - c0)), round(float(w[p]), 4)] for p in nz])
        feats.append(dict(type="Feature", properties=dict(key=k, area_km2=round(float(w.sum()), 1)), geometry=polygons(m, tr)))
        print(f"Teilgebiet {k:3s} {w.sum():7.2f} km2 in {len(nz)} Radarpixeln")
    json.dump(dict(box=BOX, grid=[1100, 900], gauges=info, parts=parts), open(OUT / "catchments.json", "w"), indent=1)
    json.dump(dict(type="FeatureCollection", features=feats), open(OUT / "catchments.geojson", "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
