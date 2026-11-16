"""Stundenregen aus Tagessummen: Menge aus den Tageswerten der DWD-Stationen, Stundenverlauf aus der Reanalyse ERA5.

Für die Jahre ohne Stundenregen (vor 2001 kein Radar, vor 2004 keine Stundenstationen). Je Teilgebiet:
  Tagessumme   abstandsgewichtetes Mittel der Tagesstationen am Gebietsmittelpunkt (Tageswert mit Datum D = 07 Uhr MEZ an D bis 07 Uhr am Folgetag,
               an den Stundenstationen 2010-2020 nachgeprüft: Korrelation 1,00)
  Verlauf      ERA5-Stundenwerte (Open-Meteo-Archiv, 0,25 Grad), auf den Gebietsmittelpunkt gewichtet; innerhalb des Tagesfensters anteilig,
               ohne ERA5-Regen im Fenster gleichmäßig
Eingang: data/tag/dwd_tageswerte_rr.json, data/tag/era5_precip_hourly.json.gz (mit dem Browser geladen). Ausgabe: data/tag/areal_tagera.csv
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rain as RN
import rain_fit as RF

HERE = Path(__file__).parent
TD = HERE / "data" / "tag"
DAY_START_H = 7          # Tageswert D: (D 07:00, D+1 07:00] MEZ
ERA_SHIFT_H = 1          # ERA5-Stempel GMT (Summe der Stunde davor) -> MEZ


def load() -> tuple[pd.DataFrame, dict, pd.DataFrame, list]:
    D = json.loads((TD / "dwd_tageswerte_rr.json").read_text(encoding="utf-8"))
    day = pd.DataFrame({sid: pd.Series(v["rs"], index=pd.to_datetime(v["d"]), dtype=float) for sid, v in D["daten"].items()})
    xy = {s["id"]: (s["lat"], s["lon"]) for s in D["stationen"]}
    E = json.loads(gzip.open(TD / "era5_precip_hourly.json.gz").read())
    t = pd.date_range(E["t0"], periods=len(E["p"][0]), freq="1h") + pd.Timedelta(hours=ERA_SHIFT_H)
    cells = {}
    for g, p in zip(E["grid"], E["p"]):
        cells.setdefault((g[0], g[1]), p)
    era = pd.DataFrame({f"{la}_{lo}": pd.Series(p, index=t, dtype=float) for (la, lo), p in cells.items()})
    return day, xy, era, list(cells)


def build() -> pd.DataFrame:
    day, xy, era, cells = load()
    out = {}
    for k, (area, cen) in _defs().items():
        w = RN.idw_weights({s: xy[s] for s in day.columns}, cen)
        dsum = RN.areal_series({s: day[s] for s in day.columns}, w)                      # mm je Tag, Datum D
        we = RN.idw_weights({c: tuple(map(float, c.split("_"))) for c in era.columns}, cen, min_km=5.0)
        e = sum(we[c] * era[c] for c in era.columns)
        # Tagesfenster je Stunde: Stempel t gehört zum Tag floor((t - 1 h - 7 h))
        dkey = (e.index - pd.Timedelta(hours=DAY_START_H) - pd.Timedelta(minutes=1)).floor("D")
        es = e.groupby(dkey).transform("sum")
        n = e.groupby(dkey).transform("size")
        share = np.where(es > 0.05, e / es.where(es > 0, 1.0), 1.0 / n)
        tot = dsum.reindex(dkey).to_numpy()
        out[k] = pd.Series(np.where(n >= 24, tot * share, np.nan), index=e.index)
    df = pd.DataFrame(out)
    df.index.name = "time_mez"
    return df


def _defs() -> dict:
    """Teilgebiete wie in rain_fit.rain_frames (Zwischengebiete und Oberläufe): Schlüssel -> (Fläche, Mittelpunkt)."""
    import chain_fc as C
    N = RF.net()
    defs = RF.lateral_defs(N)
    for k, a in C.HW.items():
        defs[k] = (float(N.loc[a["dn"], "A"]), (float(N.loc[a["nodes"], "lat"].mean()), float(N.loc[a["nodes"], "lon"].mean())))
    return defs


if __name__ == "__main__":
    df = build()
    df.round(3).to_csv(TD / "areal_tagera.csv")
    print(df.resample("YS").sum().round(0).loc["1989":"2022"].iloc[::4].to_string())
