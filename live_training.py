"""Live-Daten (Download der Live-Seite) als zusätzliche Trainingsdaten: Pegel, Stationsregen, Bodenfeuchte.

    python live_training.py ZIP_ODER_ORDNER [...]     -> data/live_pkg/{Q,W}.csv.gz, data/dwd_rr/akt/*.zip, ergänzt data/bodenfeuchte/bf_0-60.csv

Pegel: Die Rohabrufe enthalten je Abruf die letzten Tage der HVZ-Schnittstelle (W und Q in 15 min); sie werden zusammengesetzt, der neueste Abruf
gewinnt. Q kommt von der HVZ mit der heute gültigen Abflusskurve; das Modell ist an den Abflüssen bis 2022 angepasst und rechnet live mit einem
Niveaufaktor auf die Simulation (data/niveau.json: Prümzurlay 1,16, Alsdorf-Oberecken 0,76, Seffern 1,3). Für das Training wird Q deshalb
durch diesen Faktor geteilt, damit alle Jahre auf derselben Skala liegen. W bleibt, wie gemessen.
Stationsregen: die jeweils neuesten stundenwerte_RR_*_akt.zip (DWD, rund 500 Tage) nach data/dwd_rr/akt/; rain_fit.load_stations hängt sie an die
historischen Dateien an (dort, wo diese enden).
Bodenfeuchte: Tageswerte je Gebiet aus den archivierten Seiten (meta.bodenfeuchte, dieselben DWD-Werte wie bodenfeuchte_live.py), nur Tage, die in
bf_0-60.csv noch fehlen.

pkgdata.load_all(..., live=True) hängt die Pegel an die LfU-Reihen an (nur chain_fc und uncert); die LfU-Reihen haben Vorrang, dazwischen bleibt
eine Lücke (NaN). Die Dateien unter data/live_pkg/ und data/dwd_rr/akt/ kommen nicht ins Repository.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

import live

HERE = Path(__file__).parent
PKG = HERE / "data" / "live_pkg"
AKT = HERE / "data" / "dwd_rr" / "akt"
NAMES = {"P2": "Prüm 2", "EC": "Echtershausen", "WI": "Wiersdorf", "PZ": "Prümzurlay", "SI": "Sinspelt", "GI": "Giesdorf", "SE": "Seffern",
         "AL": "Alsdorf-Oberecken"}


def _roots(args: list[str], tmp: Path) -> list[Path]:
    out = []
    for i, a in enumerate(args):
        p = Path(a)
        if p.suffix == ".zip":
            d = tmp / f"z{i}"
            with zipfile.ZipFile(p) as z:
                z.extractall(d)
            p = d
        out.append(p)
    return out


def main(args: list[str]) -> None:
    with tempfile.TemporaryDirectory() as t:
        roots = _roots(args, Path(t))
        raws = sorted((r for root in roots for r in (root / "roh").glob("raw_*")), key=lambda r: r.name)
        parts = {"W": {}, "Q": {}}
        for r in raws:
            try:
                hv = live.read_hvz(r)
            except Exception as e:                                         # noqa: BLE001
                print("übersprungen", r.name, e)
                continue
            for k, v in hv.items():
                for typ in ("W", "Q"):
                    s = v.get(typ)
                    if s is not None and len(s):
                        parts[typ].setdefault(k, []).append(s)
        niveau = {k: v["faktor"] for k, v in json.loads((HERE / "data" / "niveau.json").read_text(encoding="utf-8"))["pegel"].items() if v.get("angewendet")}
        PKG.mkdir(parents=True, exist_ok=True)
        for typ, d in parts.items():
            cols = {}
            for k, ss in d.items():
                s = pd.concat(ss)
                s = s[~s.index.duplicated(keep="last")].sort_index()
                cols[NAMES[k]] = s / niveau.get(k, 1.0) if typ == "Q" else s
            df = pd.DataFrame(cols)
            df = df.reindex(pd.date_range(df.index.min().ceil("15min"), df.index.max().floor("15min"), freq="15min"))
            old = PKG / f"{typ}.csv.gz"
            if old.exists():                                                # frühere Downloads behalten, neuere Werte gewinnen
                prev = pd.read_csv(old, index_col=0, parse_dates=True)
                df = pd.concat([prev[~prev.index.isin(df.index)], df]).sort_index()
            df.index.name = "time_mez"
            df.to_csv(old, float_format="%.4g")
            print(f"{typ}: {df.index.min()} bis {df.index.max()}, {len(df)} Zeitschritte, fehlend {int(df.isna().sum().sum())}")
        # Stationsregen: neueste akt-Datei je Station
        AKT.mkdir(parents=True, exist_ok=True)
        newest = {}
        for r in raws:
            for f in (r / "dwd").glob("stundenwerte_RR_*_akt.zip"):
                newest[f.name] = f
        for n, f in newest.items():
            shutil.copy(f, AKT / n)
        # die letzten Stunden (nach dem Ende der akt-Dateien, die nur bis gestern reichen) aus dem neuesten Abruf: Stundensummen wie live
        # aus den Abrufen der letzten 3 Tage (je Abruf deckt der 10-min-Teil nur die Stunden seit dem Ende der akt-Datei ab), neuester gewinnt
        last_t = pd.Timestamp(raws[-1].name[4:].replace("Z", ""))
        obs = {}
        for r in raws:
            if pd.Timestamp(r.name[4:].replace("Z", "")) < last_t - pd.Timedelta(days=3):
                continue
            try:
                o, _ = live.station_obs(r)
            except Exception:                                              # noqa: BLE001
                continue
            for sid, ser in o.items():
                obs[sid] = ser if sid not in obs else pd.concat([obs[sid][~obs[sid].index.isin(ser.dropna().index)], ser.dropna()]).sort_index()
        for sid, ser in obs.items():
            ser.rename("rr").to_csv(AKT / f"jetzt_{sid}.csv")
        print("Stationsregen:", len(newest), "akt-Dateien und", len(obs), "Reihen bis", max(s.index.max() for s in obs.values()), "nach", AKT)
        # Bodenfeuchte aus den Seiten
        days = {}
        for root in roots:
            for f in sorted((root / "seiten" / "archiv").glob("live_*.html")):
                s = f.read_text(encoding="utf-8")
                i = s.find("const D = ")
                if i < 0:
                    continue
                d, _ = json.JSONDecoder().raw_decode(s[i + 10:])
                b = (d.get("meta") or {}).get("bodenfeuchte") or {}
                if b.get("stand") and b.get("werte"):
                    days[b["stand"]] = b["werte"]
        sm_f = HERE / "data" / "bodenfeuchte" / "bf_0-60.csv"
        sm = pd.read_csv(sm_f, index_col=0, parse_dates=True)
        new = {pd.Timestamp(k): v for k, v in days.items() if pd.Timestamp(k) not in sm.index}
        if new:
            add = pd.DataFrame(new).T.reindex(columns=sm.columns)
            sm = pd.concat([sm, add]).sort_index()
            sm.to_csv(sm_f)
        print("Bodenfeuchte: neu", sorted(str(k.date()) for k in new))


if __name__ == "__main__":
    main(sys.argv[1:])
