"""Eigenständige HTML-Übersicht zu den großen Hochwassern (Prümzurlay > 400 cm): eine Datei ohne Bibliotheken, Ganglinien als SVG.

  python floods_html.py [--out gui/hochwasser_400cm.html]     (oder über floods.py)

Inhalt: Auswahl des Hochwassers, die fünf Prüm-/Enz-Pegel übereinander, alle acht Pegel einzeln (72 h um den Scheitel in Prümzurlay),
gemeinsames Fadenkreuz, Tabellen (Scheitel je Pegel, Laufzeiten je Strecke, Vergleich aller Hochwasser), Stundenwerte.
Die Zahlen stammen aus denselben Funktionen wie data/floods_400cm_*.csv (floods.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import floods as F
import floods_plot as FP
import pkgdata

HERE = Path(__file__).parent
CSS_VAR = {"Prümzurlay": "--c-pz", "Wiersdorf": "--c-wi", "Echtershausen": "--c-ec", "Prüm 2": "--c-p2", "Sinspelt": "--c-si",
           "Giesdorf": "--c-gi", "Seffern": "--c-se", "Alsdorf-Oberecken": "--c-al"}


def clean(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def flood_payload(Q: pd.DataFrame, W: pd.DataFrame, fl: dict) -> dict:
    t0 = fl["t_peak"]
    win = F.window(Q, fl)
    rows = {r["gauge"]: r for r in F.gauge_rows(Q, W, fl)}
    series, peaks = {}, {}
    for g in F.GAUGES:
        v = win[g].to_numpy()
        series[g] = [round(float(x), 2 if v.max() < 10 else 1) for x in v]
        r = rows[g]
        peaks[g] = dict(i=int((r["t_peak"] - win.index[0]) / pd.Timedelta(minutes=15)), q=clean(r["q_peak"]), w=clean(r["w_peak"]),
                        lag=clean(r["lag_to_pz_h"]), clip=FP.clipped(g, r["w_peak"]))
    clip_w = peaks["Wiersdorf"]["clip"]
    pairs = []
    for r in F.pair_rows(Q, fl):
        keep = ("up", "dn", "km", "q_up", "q_dn", "share", "peak", "mid", "fit", "fit_nse", "dc_lag", "dc_r", "r50")
        d = {k: clean(r[k]) for k in keep}
        d["clip"] = bool(clip_w and "Wiersdorf" in (r["up"], r["dn"]))
        pairs.append(d)
    return dict(id=f"{t0:%Y-%m-%d}", t_peak=f"{t0:%Y-%m-%d %H:%M}", t0=f"{win.index[0]:%Y-%m-%d %H:%M}", n=len(win),
                w_peak=clean(fl["w_peak"]), hours_over=clean(fl["hours_over"]), q_pz=peaks["Prümzurlay"]["q"], series=series, peaks=peaks, pairs=pairs)


def build(Q: pd.DataFrame, W: pd.DataFrame, floods: list[dict], out: Path) -> None:
    data = dict(
        gauges=[dict(id=g, river=F.RIVER[g], css=CSS_VAR[g]) for g in F.GAUGES],
        overlay=FP.OVERLAY, left=FP.GRID[0], right=FP.GRID[1],
        reaches=[dict(up=u, dn=d, km=k) for u, d, k in F.PAIRS],
        floods=[flood_payload(Q, W, fl) for fl in floods])
    tpl = (HERE / "floods_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1024:.0f} kB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "gui" / "hochwasser_400cm.html"))
    ap.add_argument("--thr", type=float, default=400)
    a = ap.parse_args()
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    build(Q, W, F.find_floods(W["Prümzurlay"], a.thr), Path(a.out))


if __name__ == "__main__":
    main()
