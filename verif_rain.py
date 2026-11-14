"""Regenvorhersage ICON-D2 gegen die DWD-Stationen (python verif_rain.py): Wie gut ist die Vorhersage für die Gebiete, die das Modell nutzt?

Daten (beide echt, keine Synthetik):
  - DWD-Stundenwerte der neun Stationen (data/dwd_rr/*_hist.zip bis 31.12.2025, data/live/raw_*/dwd/*_akt.zip bis gestern), UTC.
  - ICON-D2 an den Stationsorten über Open-Meteo (data/verif/icon_d2_2022_2026.json, Abruf 12.11.2026):
      hist  Historical Forecast API ab 24.11.2022: die ersten Stunden jedes 3-stündlichen Laufs aneinandergehängt (Vorlauf 0 bis 3 h)
      day1  Previous Runs API ab 1.1.2024: Wert aus dem Lauf etwa 24 h vor der Zielzeit (Vorlauf 24 bis 27 h)
Gebietsregen wie im Modell: Stationsgewichte je Gebiet (rain_fit.rain_frames), je Stunde auf die vorhandenen Stationen renormiert; für ICON dieselben
Stationen und Gewichte (nur Stunden, in denen die Station gemessen hat). Ausgaben data/verif/rain_verif.csv und rain_factors.json (Faktor Station / ICON
aus den Summen bei 24 bis 27 h Vorlauf, je Station und je Gebiet; ICON-D2 liegt an den höheren Stationen 12 bis 19 % unter der Messung, live.py
gleicht das je Station aus).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rain as RN

HERE = Path(__file__).parent
VERIF = HERE / "data" / "verif"
AREAS = ("P2", "SI", "GI", "EC", "PZ", "SE", "AL")


def station_obs() -> dict[str, pd.Series]:
    """Stundenwerte je Station in UTC (Stempel = Ende der Stunde), historisch und aktuell zusammengesetzt."""
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / "dwd_rr" / "stundenwerte_RR_*_hist.zip"))):
        out[Path(f).name.split("_")[2]] = RN.read_dwd_hourly(f, 0.0)
    live = sorted(glob.glob(str(HERE / "data" / "live" / "raw_*")))
    if live:
        for f in sorted(glob.glob(str(Path(live[-1]) / "dwd" / "stundenwerte_RR_*_akt.zip"))):
            k = Path(f).name.split("_")[2]
            s = RN.read_dwd_hourly(f, 0.0)
            out[k] = s if k not in out else out[k].combine_first(s)
    return out


def icon_series(blocks: list, key: str, ids: list[str]) -> dict[str, pd.Series]:
    """Open-Meteo-Antworten (je Abschnitt eine Liste über die Orte) zu Reihen je Station, Index UTC = Ende der Stunde."""
    out = {k: [] for k in ids}
    for blk in blocks:
        for sid, loc in zip(ids, blk):
            t = pd.to_datetime(loc["hourly"]["time"])
            v = pd.to_numeric(pd.Series(loc["hourly"][key]), errors="coerce").to_numpy()
            out[sid].append(pd.Series(v, index=t))
    return {k: pd.concat(v).sort_index().pipe(lambda s: s[~s.index.duplicated()]) for k, v in out.items()}


def areal(series: dict[str, pd.Series], weights: dict[str, float], mask: dict[str, pd.Series] | None = None) -> pd.Series:
    s = {k: (series[k].where(mask[k].reindex(series[k].index).notna()) if mask is not None else series[k]) for k in weights}
    return RN.areal_series(s, weights)


def scores(f: pd.Series, o: pd.Series, agg_h: int) -> dict:
    """Summen über agg_h Stunden (nur vollständige Fenster), Kennzahlen Vorhersage f gegen Messung o."""
    d = pd.concat({"f": f, "o": o}, axis=1).dropna()
    d = d.resample(f"{agg_h}h", label="right", closed="right").agg(["sum", "count"])
    d = d[(d[("f", "count")] == agg_h) & (d[("o", "count")] == agg_h)]
    F, O = d[("f", "sum")].to_numpy(), d[("o", "sum")].to_numpy()
    out = dict(n=len(F), sum_ratio=float(F.sum() / max(O.sum(), 1e-9)), r=float(np.corrcoef(F, O)[0, 1]) if len(F) > 2 else np.nan,
               mae=float(np.mean(np.abs(F - O))))
    for thr in ((2.0, 10.0) if agg_h <= 6 else (10.0, 25.0)):
        yo, yf = O >= thr, F >= thr
        out[f"ereignisse>={thr:g}mm"] = int(yo.sum())
        out[f"treffer>={thr:g}"] = float(yf[yo].mean()) if yo.any() else np.nan
        out[f"fehlalarm>={thr:g}"] = float(1 - yo[yf].mean()) if yf.any() else np.nan
        out[f"haeufigkeit>={thr:g}"] = float(yf.sum() / max(yo.sum(), 1))
    return out


def main() -> None:
    import rain_fit as RF2
    import pkgdata
    import chain_fc as C
    obs = station_obs()
    raw = json.loads((VERIF / "icon_d2_2022_2026.json").read_text(encoding="utf-8"))
    ids = [s[0] for s in raw["stationen"]]
    fc = {"0-3 h": icon_series(raw["hist"], "precipitation", ids), "24-27 h": icon_series(raw["prev"], "precipitation_previous_day1", ids)}
    _, info = RF2.rain_frames(pkgdata.load_all("Q").index[:10], 1.0, extra=C.HW)
    rows, fac = [], {}
    for lead, F in fc.items():
        # Stationen einzeln
        for sid in ids:
            if sid not in obs:
                continue
            o = obs[sid]
            for agg in (1, 6, 24):
                rows.append(dict(vorlauf=lead, ort=f"Station {sid}", summe_h=agg, **scores(F[sid], o, agg)))
        for k in AREAS:
            w = info[k]["weights"]
            ao = areal(obs, w)
            af = areal(F, w, mask=obs)
            for agg in (1, 6, 24):
                rows.append(dict(vorlauf=lead, ort=f"Gebiet {k}", summe_h=agg, **scores(af, ao, agg)))
            if lead == "24-27 h":
                d = pd.concat({"f": af, "o": ao}, axis=1).dropna()
                fac.setdefault("gebiete", {})[k] = dict(faktor=round(float(d.o.sum() / d.f.sum()), 3), stunden=int(len(d)), von=str(d.index.min()),
                                                       bis=str(d.index.max()))
        if lead == "24-27 h":                                  # je Station: Summe Station / Summe ICON (gleiche Stunden); live.py rechnet damit
            for sid in ids:
                if sid in obs:
                    d = pd.concat({"f": F[sid], "o": obs[sid]}, axis=1).dropna()
                    if len(d) > 1000:
                        fac.setdefault("stationen", {})[sid] = dict(faktor=round(float(d.o.sum() / d.f.sum()), 3), stunden=int(len(d)))
    df = pd.DataFrame(rows)
    df.to_csv(VERIF / "rain_verif.csv", index=False, float_format="%.4g")
    (VERIF / "rain_factors.json").write_text(json.dumps(fac, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df[df.ort.str.startswith("Gebiet")].to_string(index=False))
    print(json.dumps(fac, indent=1))


if __name__ == "__main__":
    main()
