"""Radarniederschlag (RADKLIM RW 2017.002) als Gebietsniederschlag der Teilgebiete.

Eingang: data/radar/crop/crop_JJJJMM.bin.gz + .json aus radar_crop.py, data/radar/catchments.json aus catchments.py.
Ausgang: data/radar/areal_hourly.csv (mm je Stunde, Stempel = Ende des Intervalls, MEZ wie die Pegel und der
Stationsregen in rain_fit.py) und je Teilgebiet das stärkste Pixel (für Starkregen-Auswertungen).

RADOLAN-Rohwerte: 12 Bit Wert (Einheit 10^PR mm), Bit 12 interpoliert, Bit 13 Fehlkennung, Bit 14 negativ,
Bit 15 Clutter. Fehlkennung und Clutter werden als fehlend behandelt.
RW-Stempel ist das Ende der Stundensumme (meist hh:50 UTC). Zuordnung: Stempel + 1 h (MEZ), dann auf die nächste
volle Stunde aufgerundet. Dieser Versatz wird in check() gegen den Stationsregen geprüft.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RD = HERE / "data" / "radar"
SHIFT_H = 1.0


def crop_meta(key: str, crop_dir: Path) -> dict:
    """Metadaten zu crop_<key>.bin.gz: eigene JSON-Datei (radar_crop.py) oder Eintrag in crop_meta.json (Browser-Lauf, je Jahr)."""
    p = crop_dir / f"crop_{key}.json"
    if p.exists():
        return json.load(open(p))
    return json.load(open(crop_dir / "crop_meta.json"))[key]


def read_crop(key: str, crop_dir: Path = RD / "crop") -> tuple[pd.DatetimeIndex, np.ndarray, dict]:
    """key: Monat JJJJMM (radar_crop.py) oder Jahr JJJJ (radar_crop_browser.js)."""
    meta = crop_meta(key, crop_dir)
    r0, r1, c0, c1 = meta["box"]
    nr, nc = r1 - r0, c1 - c0
    raw = gzip.open(crop_dir / f"crop_{key}.bin.gz").read()
    rec = 10 + nr * nc * 2
    if len(raw) % rec:
        raise ValueError(f"{key}: Länge {len(raw)} kein Vielfaches von {rec}")
    n = len(raw) // rec
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, rec)
    stamps = [bytes(x).decode() for x in a[:, :10]]
    v = a[:, 10:].copy().view("<u2").reshape(n, nr, nc)
    pr = meta.get("pr")
    if pr and set(pr) != {"E-01"}:
        raise ValueError(f"{key}: unerwartete Genauigkeit {pr}")
    m = re.search(r"PR\s*E([-+]?\d+)", meta["header_first"] or "")
    prec = 10.0 ** int(m.group(1)) if m else 0.1
    val = (v & 0x0FFF).astype("float32") * prec
    val[(v & 0x4000) > 0] *= -1
    val[(v & 0x2000) > 0] = np.nan
    val[(v & 0x8000) > 0] = np.nan
    t = pd.to_datetime(stamps, format="%y%m%d%H%M") + pd.Timedelta(hours=SHIFT_H)
    t = t.ceil("1h")
    o = np.argsort(t.values, kind="stable")
    return t[o], val[o], meta


def weights(cat: dict) -> dict[str, np.ndarray]:
    r0, r1, c0, c1 = cat["box"]
    out = {}
    for k, p in cat["parts"].items():
        w = np.zeros((r1 - r0, c1 - c0))
        for r, c, a in p["pixels"]:
            w[r, c] = a
        out[k] = w
    return out


def areal(t: pd.DatetimeIndex, val: np.ndarray, W: dict[str, np.ndarray], min_cover: float = 0.9) -> pd.DataFrame:
    """Flächengewichtetes Mittel (mm/h) je Teilgebiet; fehlende Pixel werden umgewichtet, unter 90 % Abdeckung NaN.
    Dazu je Teilgebiet das größte Pixel (Spalte <key>_max)."""
    ok = np.isfinite(val)
    v0 = np.where(ok, val, 0.0)
    cols = {}
    for k, w in W.items():
        num = np.tensordot(v0, w, axes=([1, 2], [0, 1]))
        den = np.tensordot(ok.astype(float), w, axes=([1, 2], [0, 1]))
        mean = np.where(den >= min_cover * w.sum(), num / np.maximum(den, 1e-9), np.nan)
        cols[k] = mean
        mx = np.where(w > 0.05 * w.max(), np.where(ok, val, -np.inf), -np.inf).max(axis=(1, 2))
        cols[k + "_max"] = np.where(np.isfinite(mx), mx, np.nan)
    df = pd.DataFrame(cols, index=t)
    return df[~df.index.duplicated(keep="first")]


EXCESS_MM_H = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)


def excess(t: pd.DatetimeIndex, val: np.ndarray, W: dict[str, np.ndarray], min_cover: float = 0.9) -> pd.DataFrame:
    """Starkregenanteil je Teilgebiet: Flächenmittel von max(Pixelregen - f, 0) in mm/h für die Schwellen f in EXCESS_MM_H
    (Spalten <key>_x<f>). Das Gebietsmittel allein verdünnt Schauerzellen; diese Größe behält, wie viel Regen über der Schwelle fiel."""
    ok = np.isfinite(val)
    cols = {}
    for k, w in W.items():
        den = np.tensordot(ok.astype(float), w, axes=([1, 2], [0, 1]))
        good = den >= min_cover * w.sum()
        for f in EXCESS_MM_H:
            ex = np.where(ok, np.maximum(val - f, 0.0), 0.0)
            num = np.tensordot(ex, w, axes=([1, 2], [0, 1]))
            cols[f"{k}_x{f:g}"] = np.where(good, num / np.maximum(den, 1e-9), np.nan)
    df = pd.DataFrame(cols, index=t)
    return df[~df.index.duplicated(keep="first")]


def keys(crop_dir: Path = RD / "crop") -> list[str]:
    ks = sorted(p.name[5:-7] for p in crop_dir.glob("crop_*.bin.gz"))
    years = {k for k in ks if len(k) == 4}
    return [k for k in ks if not (len(k) == 6 and k[:4] in years)]      # Monatsdatei nur, wenn das Jahr fehlt


def build(ks: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = json.load(open(RD / "catchments.json"))
    W = weights(cat)
    parts = []
    xparts = []
    for k in ks or keys():
        t, val, _ = read_crop(k)
        parts.append(areal(t, val, W))
        xparts.append(excess(t, val, W))
    out = []
    for p in (parts, xparts):
        df = pd.concat(p).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "time_mez"
        out.append(df)
    return out[0], out[1]


def load() -> pd.DataFrame:
    return pd.read_csv(RD / "areal_hourly.csv", index_col=0, parse_dates=True)


if __name__ == "__main__":
    df, dx = build(sys.argv[1:] or None)
    df.round(3).to_csv(RD / "areal_hourly.csv")
    dx.round(4).to_csv(RD / "areal_excess.csv")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    miss = df.reindex(full)[[c for c in df.columns if not c.endswith("_max")]].isna()
    print("Stunden", len(full), "| fehlend je Gebiet (%):", (miss.mean() * 100).round(2).to_dict())
    print((miss.any(axis=1).groupby(full.year).mean() * 100).round(1).to_string())
