"""Regendaten: Einlesen, Gebietsniederschlag je Zwischengebiet, Vorregenindex.

Einlesen
  read_dwd_hourly(zip_oder_txt)   DWD-Stundenwerte RR (produkt_rr_stunde_*.txt, ';' getrennt, R1 in mm, -999 fehlt, MESS_DATUM YYYYMMDDHH)
                                  Zeitbasis laut DWD-Beschreibung nicht eindeutig (Intervallende oder -beginn): `shift_h` erlaubt eine Verschiebung,
                                  der passende Wert wird bei der Anpassung über den Zielwert gewählt (rain_select_shift in rain_fit.py).
Gebietsniederschlag
  idw_weights(stationen, ziel)    inverse Abstandsgewichte (Potenz 2) aus Länge und Breite der Stationen und dem Schwerpunkt des Zwischengebiets
  areal_series(reihen, gewichte)  gewichtetes Mittel mit Lückenbehandlung (Gewichte werden je Zeitpunkt auf die vorhandenen Stationen renormiert,
                                  fehlen alle, bleibt der Wert fehlend)
  areal_frame(...)                15-min-Raster (mm/h) und Vorregenindex (mm), bereit für routing.make_data(rain=...)
Der Vorregenindex API(t) = API(t - dt) * exp(-dt / tau) + P(t) * dt mit fester Abklingzeit tau = 168 h (7 Tage).
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TAU_API_H = 168.0
TAU_API_S_H = float(__import__("os").environ.get("PRUEM_TAU_S", "72.0"))   # Ereignisregen für den Sättigungsüberschuss (bis 22.11.2026: 24 h; PRUEM_TAU_S)


def read_dwd_hourly(path, shift_h: float = 0.0) -> pd.Series:
    """DWD-Stundenwerte RR einer Station als Reihe in mm je Stunde, Index UTC (naiv). Fehlwerte (-999) werden NaN."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.startswith("produkt_") and n.endswith(".txt")]
            if len(names) != 1:
                raise ValueError(f"{path}: erwartet genau eine produkt_*.txt, gefunden {names}")
            text = z.read(names[0]).decode("latin-1")
    else:
        text = path.read_bytes().decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=";", skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "MESS_DATUM" not in df or "R1" not in df:
        raise ValueError(f"{path}: Spalten MESS_DATUM und R1 erwartet, gefunden {list(df.columns)}")
    t = pd.to_datetime(df["MESS_DATUM"].str.strip(), format="%Y%m%d%H") + pd.Timedelta(hours=shift_h)
    v = pd.to_numeric(df["R1"].str.strip(), errors="coerce")
    v[v < 0] = np.nan                                                  # -999 und negative Kennungen
    s = pd.Series(v.to_numpy(), index=t, name=path.stem).sort_index()
    return s[~s.index.duplicated(keep="first")]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def idw_weights(stations: dict[str, tuple[float, float]], target: tuple[float, float], power: float = 2.0, min_km: float = 1.0) -> dict[str, float]:
    """stations: Name -> (Breite, Länge); target: (Breite, Länge) des Gebietsschwerpunkts. Gewichte summieren zu 1."""
    d = {k: max(haversine_km(la, lo, target[0], target[1]), min_km) for k, (la, lo) in stations.items()}
    w = {k: 1.0 / v ** power for k, v in d.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def areal_series(series: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Gewichtetes Mittel der Stationsreihen (gleicher Zeitraster), Gewichte je Zeitpunkt auf die vorhandenen Stationen renormiert."""
    df = pd.concat({k: series[k] for k in weights}, axis=1)
    w = pd.Series(weights)[df.columns]
    ok = df.notna()
    num = (df.fillna(0.0) * w).sum(axis=1)
    den = (ok * w).sum(axis=1)
    out = num / den
    out[den <= 0] = np.nan
    return out


def api_series(p_mm_h: np.ndarray, dt_h: float = 0.25, tau_h: float = TAU_API_H) -> np.ndarray:
    """Vorregenindex in mm aus Intensität (mm/h) auf gleichmäßigem Raster; fehlende Werte zählen als 0 (Lücken werden vom Aufrufer über p maskiert)."""
    k = math.exp(-dt_h / tau_h)
    import routing                                          # Rechenkern in C, falls übersetzt (spät importiert)
    lib = routing._load_lib()
    if lib:
        x = np.ascontiguousarray(p_mm_h, dtype=np.float64)
        out = np.empty_like(x)
        lib.api_series(x, out, len(x), k, float(dt_h))
        return out
    out = np.empty(len(p_mm_h))
    acc = 0.0
    for i, v in enumerate(np.nan_to_num(p_mm_h)):
        acc = acc * k + v * dt_h
        out[i] = acc
    return out


def areal_frame(hourly_mm: pd.Series, index: pd.DatetimeIndex, dt_h: float = 0.25) -> pd.DataFrame:
    """Stundenreihe (mm je Stunde, Stempel = Ende des Intervalls) auf das 15-min-Raster von `index` (UTC, naiv): Intensität in mm/h, konstant im
    Intervall. Rückgabe Spalten p (mm/h), api (mm, 7 Tage) und api_s (mm, TAU_API_S_H, seit 22.11.2026 72 h). Zeiten ohne Stundenwert bleiben NaN (auch im api, damit Ereignisse mit Lücken entfallen)."""
    h = hourly_mm.reindex(pd.date_range(hourly_mm.index.min(), hourly_mm.index.max(), freq="1h"))
    # Stundenwert gilt für die vier Viertelstunden vor dem Stempel: Zeitpunkt t (Ende des 15-min-Intervalls) -> nächster voller Stundenstempel
    stamp = index.ceil("1h")
    p = h.reindex(stamp).to_numpy()
    out = pd.DataFrame({"p": p}, index=index)
    # Vorregenindex: auch vor und in Lücken fortschreiben (Lücken als 0), aber nur dort gültig, wo davor 7 Tage Regenwerte vorhanden waren
    api = api_series(out["p"].to_numpy(), dt_h)
    valid = out["p"].notna().astype(float).rolling(int(TAU_API_H / dt_h), min_periods=1).mean() > 0.95
    out["api"] = np.where(valid, api, np.nan)
    out["api_s"] = np.where(valid, api_series(out["p"].to_numpy(), dt_h, TAU_API_S_H), np.nan)
    return out
