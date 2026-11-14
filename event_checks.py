"""Zusatzprüfungen zur Welle vom 24.–26.10.2026: CSV gegen JSON-Schnittstelle, Geschwindigkeiten aus Scheitel-Lags,
Zwei-Eingangs-Fit für Prümzurlay, Rückhalt im Stausee Bitburg.

  python event_checks.py [data/wasserportal_event_2026-10.csv] [data/real_5d.csv]
"""
import sys

import numpy as np
import pandas as pd

from lags_event import NAMES, PAIRS, fit_lag, load_q, pair_table

EV = sys.argv[1] if len(sys.argv) > 1 else "data/wasserportal_event_2026-10.csv"
JS = sys.argv[2] if len(sys.argv) > 2 else "data/real_5d.csv"
KM = {"Prüm 2": 79.53, "Echtershausen": 38.44, "Wiersdorf": 32.14, "Prümzurlay": 7.86, "Giesdorf": 53.49, "Seffern": 35.04,
      "Alsdorf-Oberecken": 6.23}
W = load_q(EV)
J = load_q(JS)

print("== 1) Wasserportal-CSV gegen JSON-Schnittstelle im gemeinsamen Zeitraum")
idx = W.index.intersection(J.index)
print(f"gemeinsam: {idx[0]:%d.%m. %H:%M} bis {idx[-1]:%d.%m. %H:%M} UTC, {len(idx)} Schritte")
print(f"{'Pegel':<18}{'Max CSV':>9}{'Max JSON':>9}{'Δ Max %':>9}{'Median Verh.':>13}{'Verh. bei Q<Median':>20}{'Verh. bei Q>Median':>20}")
for c in W.columns:
    if c not in J.columns:
        continue
    a, b = W.loc[idx, c], J.loc[idx, c]
    r = (a / b).replace([np.inf, -np.inf], np.nan)
    med = b.median()
    print(f"{c:<18}{a.max():9.2f}{b.max():9.2f}{(a.max() / b.max() - 1) * 100:9.1f}{r.median():13.3f}"
          f"{r[b < med].median():20.3f}{r[b >= med].median():20.3f}")

print("\n== 2) Scheitel-Lag und daraus Wellengeschwindigkeit (Flusskilometer)")
tabs = {(t['up'], t['dn']): t for t in (pair_table(W, u, d) for u, d in PAIRS)}
for (u, d), t in tabs.items():
    if u in KM and d in KM and t["lag_peak_h"]:
        L = (KM[u] - KM[d]) if (u, d) != ("Echtershausen", "Wiersdorf") else (KM[u] - KM[d])
        print(f"{u:>14} -> {d:<18} {L:5.1f} km  Scheitel-Lag {t['lag_peak_h']:5.2f} h  -> {L * 1000 / (t['lag_peak_h'] * 3600):4.2f} m/s"
              f"  | Schwerpunkt-Lag {t['lag_centroid_h']:5.2f} h -> {L * 1000 / (t['lag_centroid_h'] * 3600):4.2f} m/s")

print("\n== 3) Prümzurlay = a*Wiersdorf(t-L1) + b*Sinspelt(t-L2) + c   (Lags 0..12 h, Kleinste Quadrate)")
r = fit_lag(W["Wiersdorf"], W["Prümzurlay"], extra=W["Sinspelt"])
print(f"beste Anpassung: Wiersdorf-Lag {r['lag_h']:.2f} h, Sinspelt-Lag {r['lag2_h']:.2f} h, a={r['a']:.2f}, b={r['b']:.2f}, c={r['c']:.2f}, NSE={r['nse']:.3f}, n={r['n']}")
r1 = fit_lag(W["Wiersdorf"], W["Prümzurlay"])
print(f"nur Wiersdorf: Lag {r1['lag_h']:.2f} h, a={r1['a']:.2f}, c={r1['c']:.2f}, NSE={r1['nse']:.3f}")
print("Empfindlichkeit (Sinspelt-Lag 2 h fest): NSE gegen Wiersdorf-Lag")
for L1 in (2, 3, 4, 5, 6, 7, 8, 10):
    X = pd.concat([W["Wiersdorf"].shift(int(L1 * 4)), W["Sinspelt"].shift(8)], axis=1)
    m = X.notna().all(axis=1)
    A = np.c_[X[m].to_numpy(), np.ones(m.sum())]
    yy = W["Prümzurlay"][m].to_numpy()
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    print(f"   Lag {L1:2d} h: NSE {1 - ((yy - A @ coef) ** 2).sum() / ((yy - yy.mean()) ** 2).sum():.4f}")

print("\n== 4) Stausee Bitburg: Zufluss (Echtershausen) gegen Abgabe (Wiersdorf)")
S = pd.read_csv(EV, parse_dates=["time_utc"], dtype={"number": str})
see = S[(S["number"] == "26280366")].set_index("time_utc")["value"]
dt = 900.0
qi, qo = W["Echtershausen"], W["Wiersdorf"]
print(f"Scheitel Echtershausen {qi.max():.2f} m³/s ({qi.idxmax():%d.%m. %H:%M}), Wiersdorf {qo.max():.2f} m³/s ({qo.idxmax():%d.%m. %H:%M}): "
      f"gedämpft auf {qo.max() / qi.max() * 100:.0f} %")
print(f"Seestand {see.min():.0f} bis {see.max():.0f} cm ({see.idxmin():%d.%m. %H:%M} bis {see.idxmax():%d.%m. %H:%M}), Anstieg {see.max() - see.iloc[0]:.0f} cm")
vi, vo = (qi.sum() * dt) / 1e6, (qo.sum() * dt) / 1e6
print(f"Volumen im Fenster: Echtershausen {vi:.2f} Mio m³, Wiersdorf {vo:.2f} Mio m³")

print("\n== 5) Lag-Fit getrennt für steigenden Ast (bis zum Scheitel des Unterliegers) und fallenden Ast (danach)")
for u, d in PAIRS:
    y, x = W[d], W[u]
    tp = y.idxmax()
    out = []
    for name, sel in (("steigend", y.index <= tp), ("fallend", y.index > tp)):
        extra = W["Sinspelt"] if (d == "Prümzurlay" and u == "Wiersdorf") else None
        r = fit_lag(x, y, sel, extra)
        out.append(f"{name}: L={r['lag_h']:.2f} h" + (f", L2={r['lag2_h']:.2f} h" if extra is not None else "") + f", a={r['a']:.2f}, NSE={r['nse']:.3f}, n={r['n']}")
    print(f"{u:>14} -> {d:<18} " + " | ".join(out))
