"""Große Hochwasser 1989-2022: Prümzurlay über 400 cm Wasserstand. Je Hochwasser Scheitel aller Pegel, Lags der Prüm- und Enz-Strecken und
72-h-Abbildungen des Abflusses (36 h vor bis 36 h nach dem Scheitel in Prümzurlay) für alle acht Pegel.

  python floods.py [--thr 400] [--plots figures] [--html gui/hochwasser_400cm.html]

Ausgaben: data/floods_400cm_gauges.csv (je Hochwasser und Pegel), data/floods_400cm_pairs.csv (je Hochwasser und Strecke),
figures/hochwasser_<datum>.png, gui/hochwasser_400cm.html (eine Datei, Ganglinien als SVG).
Hochwasser = zusammenhängende Zeit mit W Prümzurlay > Schwelle; Zeiträume mit weniger als 48 h Abstand gelten als ein Ereignis.
Zeitbasis: wie im Paket (angenommen MEZ, UTC+1, ohne Sommerzeit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lags_event as le
import lags_history as lh
import pkgdata

HERE = Path(__file__).parent
GAUGES = ["Prüm 2", "Echtershausen", "Wiersdorf", "Prümzurlay", "Sinspelt", "Giesdorf", "Seffern", "Alsdorf-Oberecken"]
RIVER = {"Prüm 2": "Prüm", "Echtershausen": "Prüm", "Wiersdorf": "Prüm", "Prümzurlay": "Prüm", "Sinspelt": "Enz",
         "Giesdorf": "Nims", "Seffern": "Nims", "Alsdorf-Oberecken": "Nims"}
# (Oberlieger, Unterlieger, Fließstrecke km): Prüm-km aus den Fluss-km der Pegel, Enz bis zur Mündung und weiter bis Prümzurlay geschätzt (15,1 + 4,8)
PAIRS = [("Prüm 2", "Echtershausen", 41.1), ("Echtershausen", "Wiersdorf", 6.3), ("Wiersdorf", "Prümzurlay", 24.3),
         ("Echtershausen", "Prümzurlay", 30.6), ("Sinspelt", "Prümzurlay", 19.9)]
STEP_H = 0.25
HALF_H = 36


def find_floods(W: pd.Series, thr: float = 400, merge_h: float = 48) -> list[dict]:
    m = W > thr
    grp = (m != m.shift()).cumsum()
    ev = [(s.index[0], s.index[-1], s.idxmax(), float(s.max())) for _, s in W[m].groupby(grp[m])]
    out: list[tuple] = []
    for e in ev:
        if out and (e[0] - out[-1][1]) < pd.Timedelta(hours=merge_h):
            a = out[-1]
            out[-1] = (a[0], e[1], e[2] if e[3] > a[3] else a[2], max(a[3], e[3]))
        else:
            out.append(e)
    return [dict(start=a, end=b, t_peak=t, w_peak=w, hours_over=float((W.loc[a:b] > thr).sum()) * 0.25) for a, b, t, w in out]


def gauge_rows(Q: pd.DataFrame, W: pd.DataFrame, fl: dict) -> list[dict]:
    t0 = fl["t_peak"]
    rows = []
    for g in GAUGES:
        s = Q[g].loc[t0 - pd.Timedelta(hours=30): t0 + pd.Timedelta(hours=6)]
        i = s.to_numpy().argmax()
        v = s.to_numpy()
        mid, plat = lh.plateau_mid(v, i)
        ws = W[g].loc[s.index[0]:s.index[-1]]
        full = Q[g].loc[t0 - pd.Timedelta(hours=HALF_H): t0 + pd.Timedelta(hours=HALF_H)]
        rows.append(dict(flood=t0, gauge=g, river=RIVER[g], t_peak=s.index[i], q_peak=v[i], w_peak=float(ws.max()),
                         lag_to_pz_h=(s.index[i] - t0).total_seconds() / 3600, plateau_h=plat,
                         q_start=float(full.iloc[0]), q_min_window=float(full.min()), q_max_window=float(full.max())))
    return rows


def pair_rows(Q: pd.DataFrame, fl: dict) -> list[dict]:
    t0 = fl["t_peak"]
    ipz = Q.index.get_loc(t0)
    out = []
    for up, dn, km in PAIRS:
        if dn == "Prümzurlay":
            ip = ipz
        else:
            seg = Q[dn].iloc[ipz - int(24 / STEP_H): ipz + int(6 / STEP_H) + 1]
            ip = ipz - int(24 / STEP_H) + int(seg.to_numpy().argmax())
        r = lh.pair_row(Q, up, dn, ip)
        if r is None:
            continue
        r.update(flood=t0, up=up, dn=dn, km=km)
        out.append(r)
    return out


def joint_row(Q: pd.DataFrame, fl: dict) -> dict | None:
    ip = Q.index.get_loc(fl["t_peak"])
    w0, w1 = ip - int(36 / STEP_H), ip + int(12 / STEP_H)
    sl = slice(w0 - int(12 / STEP_H), w1 + 1)
    sel = np.r_[np.zeros(int(12 / STEP_H), bool), np.ones(w1 - w0 + 1, bool)]
    ft = le.fit_lag(Q["Wiersdorf"].iloc[sl], Q["Prümzurlay"].iloc[sl], sel, extra=Q["Sinspelt"].iloc[sl], maxl_h=12)
    return dict(flood=fl["t_peak"], L1_wiersdorf=ft["lag_h"], L2_sinspelt=ft["lag2_h"], a=ft["a"], b=ft["b"], c=ft["c"], nse=ft["nse"])


def peak_time(s: pd.Series, t0: pd.Timestamp) -> pd.Timestamp:
    """Scheitelzeit eines Pegels: Maximum im Fenster [t0 - 30 h, t0 + 6 h] um den Scheitel in Prümzurlay (bei Gleichstand der erste Wert)."""
    return s.loc[t0 - pd.Timedelta(hours=30): t0 + pd.Timedelta(hours=6)].idxmax()


def window(Q: pd.DataFrame, fl: dict) -> pd.DataFrame:
    t0 = fl["t_peak"]
    return Q.loc[t0 - pd.Timedelta(hours=HALF_H): t0 + pd.Timedelta(hours=HALF_H), GAUGES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=400)
    ap.add_argument("--plots", default=str(HERE / "figures"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--html", default=str(HERE / "gui" / "hochwasser_400cm.html"))
    ap.add_argument("--no-html", action="store_true")
    a = ap.parse_args()
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    floods = find_floods(W["Prümzurlay"], a.thr)
    g, p, j = [], [], []
    for fl in floods:
        g += gauge_rows(Q, W, fl)
        p += pair_rows(Q, fl)
        j.append(joint_row(Q, fl))
    pd.DataFrame(floods).to_csv(HERE / "data" / "floods_400cm_events.csv", index=False)
    pd.DataFrame(g).to_csv(HERE / "data" / "floods_400cm_gauges.csv", index=False, float_format="%.3f")
    pd.DataFrame(p).to_csv(HERE / "data" / "floods_400cm_pairs.csv", index=False, float_format="%.3f")
    pd.DataFrame(j).to_csv(HERE / "data" / "floods_400cm_joint.csv", index=False, float_format="%.3f")
    print(len(floods), "Hochwasser")
    if not a.no_plots:
        import floods_plot
        floods_plot.make_all(Q, floods, a.plots)
    if not a.no_html:
        import floods_html
        floods_html.build(Q, W, floods, Path(a.html))


if __name__ == "__main__":
    main()
