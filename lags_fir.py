"""Mittlere Verzögerung je Abflussklasse aus einer nicht-negativen Übertragungsfunktion (FIR), statt Scheitel-Lags je Ereignis.

  python lags_fir.py [--pkg data/lfu_pkg/PKG04_Q.csv.gz] [--boot 200]

Warum: Scheitel- und Anstiegs-Lags je Ereignis streuen stark, weil der Regen die Zwischengebiete zu verschiedenen Zeiten trifft
(Wiersdorf -> Prümzurlay: -4 bis +17 h) und bei Prümzurlay Prüm und Enz zusammenlaufen. Die Übertragungsfunktion sucht dagegen für ein
ganzes Ereignisfenster Gewichte h_k >= 0 mit  y(t) = sum_k h_k * x(t - k*15 min) [+ sum_k g_k * z(t - k*15 min)] + c_Ereignis,
mit Glättung der zweiten Differenzen. Die mittlere Verzögerung ist der Schwerpunkt von h (sum k*h_k / sum h_k), die Verstärkung sum h_k.
Ereignisse werden zu Klassen nach dem Scheitel des Oberliegers zusammengefasst. Unsicherheit der Klassen mit mehreren Ereignissen:
Bootstrap über Ereignisse (Mediane und 10-/90-%-Quantile); für ein Einzelereignis: Spanne über drei Glättungsstärken.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import lags_history as lh

HERE = Path(__file__).parent
STEP_H = 0.25
KMAX = 96          # 24 h
PRE, POST = 36, 12  # Fenster um den Scheitel unten in Stunden


def design(inputs: list[np.ndarray], wins: list[tuple[int, int]], k: int):
    """Spalten: je Eingang k+1 Verzögerungen, dann je Ereignis eine freie Konstante. Zeilen: alle Zeitschritte der Fenster."""
    rows, cols = [], []
    for a, b in wins:
        idx = np.arange(a, b + 1)
        cols.append(np.hstack([np.stack([x[idx - j] for j in range(k + 1)], axis=1) for x in inputs]))
        rows.append(idx)
    return cols, rows


def fit_fir(inputs: list[np.ndarray], y: np.ndarray, wins: list[tuple[int, int]], lam: float, k: int = KMAX) -> dict:
    cols, rows = design(inputs, wins, k)
    nin, ne = len(inputs), len(wins)
    X = np.vstack([np.hstack([c, np.eye(ne)[[e] * len(c)]]) for e, c in enumerate(cols)])
    t = np.concatenate([y[r] for r in rows])
    scale = float(np.std(t)) or 1.0
    # Glättung: zweite Differenz je Eingang, Gewicht lam * scale / Zahl der Gewichte
    pen = []
    for q in range(nin):
        for j in range(1, k):
            r = np.zeros(X.shape[1])
            r[q * (k + 1) + j - 1: q * (k + 1) + j + 2] = [1, -2, 1]
            pen.append(r * lam * scale)
    A = np.vstack([X, np.array(pen)])
    b = np.concatenate([t, np.zeros(len(pen))])
    lo = np.r_[np.zeros(nin * (k + 1)), -np.inf * np.ones(ne)]
    res = lsq_linear(A, b, bounds=(lo, np.inf), method="bvls", max_iter=2000)
    h = res.x[:nin * (k + 1)].reshape(nin, k + 1)
    pred = X @ res.x
    nse = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    gain = h.sum(axis=1)
    delay = (h * np.arange(k + 1)).sum(axis=1) / np.where(gain > 0, gain, np.nan) * STEP_H
    return dict(h=h, gain=gain, delay_h=delay, nse=float(nse), n=len(t))


def windows(Q: pd.DataFrame, dn: str, idxs: list[int]) -> list[tuple[int, int]]:
    return [(i - int(PRE / STEP_H), i + int(POST / STEP_H)) for i in idxs if i - int(PRE / STEP_H) - KMAX >= 0 and i + int(POST / STEP_H) < len(Q)]


def classify(q: np.ndarray, label_extreme: float) -> list[str]:
    """Klassen nach Oberlieger-Scheitel: extrem (>= label_extreme), groß (obere Hälfte der Rest), klein (untere Hälfte)."""
    med = float(np.median(q[q < label_extreme])) if (q < label_extreme).any() else 0.0
    return ["extrem" if v >= label_extreme else "groß" if v >= med else "klein" for v in q]


def run(Q: pd.DataFrame, reaches: list[tuple[str, str, str | None]], boot: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for up, dn, partner in reaches:
        ev = lh.events_of(Q[dn])
        ev = [i for i in ev if i - int(PRE / STEP_H) - KMAX >= 0 and i + int(POST / STEP_H) < len(Q)]
        qu = np.array([Q[up].iloc[i - int(24 / STEP_H):i + int(6 / STEP_H) + 1].max() for i in ev])
        cls = classify(qu, 150.0)
        ins = [Q[up].to_numpy()] + ([Q[partner].to_numpy()] if partner else [])
        y = Q[dn].to_numpy()
        for c in ("klein", "groß", "extrem"):
            sel = [i for i, k in zip(ev, cls) if k == c]
            if not sel:
                continue
            w = windows(Q, dn, sel)
            qmed = float(np.median([qu[ev.index(i)] for i in sel]))
            row = dict(up=up, dn=dn, partner=partner or "", klasse=c, n_ev=len(sel), q_up_median=qmed,
                       q_up_min=float(min(qu[ev.index(i)] for i in sel)), q_up_max=float(max(qu[ev.index(i)] for i in sel)))
            fits = {lam: fit_fir(ins, y, w, lam) for lam in (0.3, 1.0, 3.0)}
            f = fits[1.0]
            row.update(delay=f["delay_h"][0], gain=f["gain"][0], nse=f["nse"],
                       delay_lo=min(v["delay_h"][0] for v in fits.values()), delay_hi=max(v["delay_h"][0] for v in fits.values()))
            if partner:
                row.update(delay2=f["delay_h"][1], gain2=f["gain"][1])
            if len(sel) >= 3 and boot:
                d = []
                for _ in range(boot):
                    pick = rng.integers(0, len(w), len(w))
                    d.append(fit_fir(ins, y, [w[p] for p in pick], 1.0)["delay_h"][0])
                d = np.array(d)
                row.update(boot_p10=float(np.nanpercentile(d, 10)), boot_p90=float(np.nanpercentile(d, 90)))
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=str(HERE / "data" / "lfu_pkg" / "PKG04_Q.csv.gz"))
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "data" / "lags_fir_classes.csv"))
    a = ap.parse_args()
    Q = pd.read_csv(a.pkg, index_col=0, parse_dates=True)
    reaches = [("Prüm 2", "Echtershausen", None), ("Echtershausen", "Wiersdorf", None), ("Wiersdorf", "Prümzurlay", "Sinspelt"),
               ("Wiersdorf", "Prümzurlay", None), ("Giesdorf", "Seffern", None), ("Seffern", "Alsdorf-Oberecken", None)]
    df = run(Q, reaches, a.boot)
    df.to_csv(a.out, index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
