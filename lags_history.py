"""Laufzeiten je Hochwasser-Ereignis aus den 15-min-Reihen 2019-11 bis 2022-12 (LfU-Paket PKG04, geprüfte Werte): hängt der Lag vom Abfluss ab?

  python lags_history.py [--pkg data/lfu_pkg/PKG04_Q.csv.gz] [--out data/lags_history_events.csv]

Ereignisse: Scheitel des Unterliegers über seinem 98-%-Quantil (Glättung 1 h, mindestens 3 Tage Abstand, Prominenz mindestens 40 % der
Schwelle). Je Ereignis und Strecke (Oberlieger -> Unterlieger):
  peak     Scheitelzeit des Oberliegers (größtes Maximum im Fenster [Scheitel unten - 24 h, + 6 h], parabolisch verfeinert) bis Scheitel unten
  mid      dasselbe mit der Mitte des Bereichs >= 99 % des Maximums (Plateaus)
  r25..r90 Zeit, zu der jeder Pegel den Anteil f seines eigenen Anstiegs (über dem Minimum im Fenster) zuletzt vor seinem Scheitel erreicht
  fit      y(t) = a * x(t - L) + c über [Scheitel unten - 36 h, + 12 h], L = 0 bis 24 h, Gütemaß NSE
  dc       Verschiebung mit der höchsten Korrelation der 1-h-Änderungen auf dem steigenden Ast (36 h bis Scheitel unten), dc_r = Korrelation;
           reagiert auf steile Fronten (Speicherabgaben) und weniger auf glatten Zwischenzufluss
  share    Scheitel oben / Scheitel unten (wie viel des unteren Scheitels der Oberlieger erklären kann)
Lags sind Reaktionsverzögerungen zwischen zwei Ganglinien einschließlich Zwischenzufluss, bei Strecken mit kleinem share kaum Wellenlaufzeit.
Zusätzlich für Prümzurlay: gemeinsamer Fit mit Wiersdorf und Sinspelt (Enz) je Ereignis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import lags_event as le

HERE = Path(__file__).parent
PAIRS = [("Prüm 2", "Echtershausen"), ("Echtershausen", "Wiersdorf"), ("Wiersdorf", "Prümzurlay"), ("Echtershausen", "Prümzurlay"),
         ("Giesdorf", "Seffern"), ("Seffern", "Alsdorf-Oberecken")]
STEP_H = 0.25
FR = (0.25, 0.5, 0.75, 0.9)


def parab(v: np.ndarray, i: int) -> float:
    """Verfeinerter Index des Maximums bei i (Parabel durch drei Punkte)."""
    if 0 < i < len(v) - 1:
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def plateau_mid(v: np.ndarray, i: int) -> tuple[float, float]:
    lo = hi = i
    while lo > 0 and v[lo - 1] >= 0.99 * v[i]:
        lo -= 1
    while hi < len(v) - 1 and v[hi + 1] >= 0.99 * v[i]:
        hi += 1
    return (lo + hi) / 2, (hi - lo + 1) * STEP_H


def rise_time(v: np.ndarray, ip: int, base: float, f: float) -> float | None:
    """Index (fraktional), an dem v den Anteil f des Anstiegs base -> v[ip] zuletzt vor ip erreicht."""
    lvl = base + f * (v[ip] - base)
    j = ip
    while j > 0 and v[j - 1] >= lvl:
        j -= 1
    if j == 0:
        return None
    a, b = v[j - 1], v[j]
    return j - 1 + ((lvl - a) / (b - a) if b != a else 0.0)


def dcorr_lag(Q: pd.DataFrame, up: str, dn: str, ip: int, pre_h: float = 36, k: int = 4, maxl_h: float = 12) -> tuple[float, float]:
    """Verschiebung (h) mit der höchsten Korrelation der 1-h-Änderungen auf dem steigenden Ast [ip - pre_h, ip] und diese Korrelation."""
    a, n, mx = ip - int(pre_h / STEP_H), int(pre_h / STEP_H) + 1, int(maxl_h / STEP_H)
    x = Q[up].diff(k).to_numpy()
    y = Q[dn].diff(k).to_numpy()[a:a + n]
    best = (0.0, -2.0)
    for L in range(mx + 1):
        if a - L < 0:
            break
        r = np.corrcoef(x[a - L:a - L + n], y)[0, 1]
        if r > best[1]:
            best = (L * STEP_H, float(r))
    return best


def events_of(s: pd.Series) -> list[int]:
    sm = s.rolling(4, center=True, min_periods=1).mean()
    thr = float(s.quantile(0.98))
    pk, _ = find_peaks(sm.to_numpy(), height=thr, distance=int(3 * 24 / STEP_H), prominence=0.4 * thr)
    return [int(i) for i in pk]


def pair_row(Q: pd.DataFrame, up: str, dn: str, ip: int) -> dict | None:
    """Alle Lag-Maße für ein Ereignis, dessen Scheitel am Unterlieger bei Index ip liegt (±6 h wird das Maximum gesucht)."""
    u, d = Q[up].to_numpy(), Q[dn].to_numpy()
    a, b = ip - int(24 / STEP_H), ip + int(6 / STEP_H)
    if a < 0 or b >= len(Q) or ip + int(12 / STEP_H) >= len(Q) or ip - int(36 / STEP_H) - int(24 / STEP_H) < 0:
        return None
    iu = a + int(np.argmax(u[a:b + 1]))
    ipd = ip - int(6 / STEP_H) + int(np.argmax(d[ip - int(6 / STEP_H):ip + int(6 / STEP_H) + 1]))  # Maximum am Ereignispunkt
    pu, pdn = parab(u, iu), parab(d, ipd)
    mu, plu = plateau_mid(u, iu)
    md, pld = plateau_mid(d, ipd)
    w0, w1 = ipd - int(36 / STEP_H), ipd + int(12 / STEP_H)
    r = dict(t_up=Q.index[iu], t_dn=Q.index[ipd], q_up=u[iu], q_dn=d[ipd], share=u[iu] / d[ipd],
             peak=(pdn - pu) * STEP_H, mid=(md - mu) * STEP_H, plat_up=plu, plat_dn=pld)
    bu, bd = float(u[w0:iu + 1].min()), float(d[w0:ipd + 1].min())
    for f in FR:
        tu, td = rise_time(u, iu, bu, f), rise_time(d, ipd, bd, f)
        r[f"r{int(f * 100)}"] = None if tu is None or td is None else (td - tu) * STEP_H
    x, y = Q[up].iloc[w0 - int(24 / STEP_H):w1 + 1], Q[dn].iloc[w0 - int(24 / STEP_H):w1 + 1]
    sel = np.r_[np.zeros(len(y) - (w1 - w0 + 1), bool), np.ones(w1 - w0 + 1, bool)]
    ft = le.fit_lag(x, y, sel, maxl_h=24)
    r.update(fit=ft["lag_h"], fit_nse=ft["nse"], fit_a=ft["a"])
    r.update(zip(("dc_lag", "dc_r"), dcorr_lag(Q, up, dn, ipd)))
    return r


def pair_events(Q: pd.DataFrame, up: str, dn: str) -> list[dict]:
    return [r for r in (pair_row(Q, up, dn, ip) for ip in events_of(Q[dn])) if r]


def joint_pz(Q: pd.DataFrame) -> list[dict]:
    """Prümzurlay = a * Wiersdorf(t-L1) + b * Sinspelt(t-L2) + c je Ereignis (L1 0..12 h, L2 0..12 h)."""
    rows = []
    for ip in events_of(Q["Prümzurlay"]):
        w0, w1 = ip - int(36 / STEP_H), ip + int(12 / STEP_H)
        if w0 - int(12 / STEP_H) < 0 or w1 >= len(Q):
            continue
        sl = slice(w0 - int(12 / STEP_H), w1 + 1)
        sel = np.r_[np.zeros(int(12 / STEP_H), bool), np.ones(w1 - w0 + 1, bool)]
        ft = le.fit_lag(Q["Wiersdorf"].iloc[sl], Q["Prümzurlay"].iloc[sl], sel, extra=Q["Sinspelt"].iloc[sl], maxl_h=12)
        rows.append(dict(t_dn=Q.index[ip], q_dn=float(Q["Prümzurlay"].iloc[ip]), q_wiers=float(Q["Wiersdorf"].iloc[ip - 48:ip + 8].max()),
                         q_sinsp=float(Q["Sinspelt"].iloc[ip - 48:ip + 8].max()), L1=ft["lag_h"], L2=ft["lag2_h"], a=ft["a"], b=ft["b"],
                         c=ft["c"], nse=ft["nse"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=str(HERE / "data" / "lfu_pkg" / "PKG04_Q.csv.gz"))
    ap.add_argument("--out", default=str(HERE / "data" / "lags_history_events.csv"))
    a = ap.parse_args()
    Q = pd.read_csv(a.pkg, index_col=0, parse_dates=True)
    assert (Q.index[1:] - Q.index[:-1] == pd.Timedelta(minutes=15)).all(), "Raster nicht lückenlos"
    out = []
    for up, dn in PAIRS:
        for r in pair_events(Q, up, dn):
            out.append(dict(up=up, dn=dn, **r))
    df = pd.DataFrame(out)
    df.to_csv(a.out, index=False, float_format="%.3f")
    pd.DataFrame(joint_pz(Q)).to_csv(str(Path(a.out).with_name("lags_history_joint_pz.csv")), index=False, float_format="%.3f")
    print(df.groupby(["up", "dn"]).size())


if __name__ == "__main__":
    main()
