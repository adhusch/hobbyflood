"""Synthetische Kaskade zum Testen der Pipeline (keine echten Prüm-Daten!).

System 1 (freie Strecke): Giesdorf -> Seffern  (Laufzeit, Dämpfung, Zwischenzufluss)
System 2 (Speicher):      Echtershausen -> Stausee -> Wiersdorf (Abgabe nach Betriebsregel)
Im letzten Jahr (Testjahr) werden Starkregen eingebaut, die größer sind als im Training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rain_series(n, rng, storms=()):
    wet = np.zeros(n, bool)
    s = False
    u = rng.random(n)
    for i in range(n):
        s = (u[i] < 0.03) if not s else (u[i] >= 0.12)
        wet[i] = s
    t = np.arange(n)
    season = 1 + 0.35 * np.cos(2 * np.pi * (t / 8766 - 0.05))
    r = np.where(wet, rng.gamma(0.7, 0.6, n) * season, 0.0)
    for start, hours, mmh in storms:
        r[start:start + hours] += mmh * rng.uniform(0.7, 1.3, hours)
    return r


def catchment(rain, area_km2, soil0=60.0):
    n = len(rain)
    S, soil, fast, slow = 100.0, soil0, 0.0, 0.0
    q = np.zeros(n)
    for i in range(n):
        eff = rain[i] * (soil / S) ** 2
        soil = min(S, max(0.0, soil + rain[i] - eff - 0.04))
        fast += 0.6 * eff
        slow += 0.4 * eff
        of, os_ = fast / 10.0, slow / 150.0
        fast -= of
        slow -= os_
        q[i] = (of + os_) * area_km2 / 3.6
    return q


def muskingum(x, K, X=0.2, dt=1.0):
    d = K * (1 - X) + 0.5 * dt
    c0, c1, c2 = (-K * X + 0.5 * dt) / d, (K * X + 0.5 * dt) / d, (K * (1 - X) - 0.5 * dt) / d
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = c0 * x[i] + c1 * x[i - 1] + c2 * y[i - 1]
    return np.maximum(y, 0)


def reservoir(inflow, vmax=2.4e6):
    V, out, lvl = 0.5 * vmax, np.zeros_like(inflow), np.zeros_like(inflow)
    for i, qi in enumerate(inflow):
        f = V / vmax
        if f < 0.4:
            o = min(3.0, V / 3600)
        elif f < 0.8:
            o = max(0.0, 3.0 + (f - 0.4) / 0.4 * (qi - 3.0))
        else:
            o = qi + (V - 0.8 * vmax) / (6 * 3600)
        V = min(vmax, max(0.0, V + (qi - o) * 3600))
        out[i], lvl[i] = o, V / vmax
    return out, lvl


def generate(years=5, seed=1):
    rng = np.random.default_rng(seed)
    n = years * 8760
    idx = pd.date_range("2019-01-01", periods=n, freq="h")
    ty = n - 8760  # Beginn Testjahr
    storms_train = [(int(rng.integers(0, ty - 100)), 8, 3.0) for _ in range(4)]
    storms_test = [(ty + 1500, 14, 5.5), (ty + 4200, 12, 6.5), (ty + 6900, 16, 5.0)]
    out = {}
    # System 1
    r = rain_series(n, rng, storms_train + storms_test)
    rm = np.maximum(r * rng.lognormal(0, 0.3, n), 0)
    qg = catchment(r, 120)
    qm = catchment(rm, 70)
    out["Giesdorf"] = qg * rng.lognormal(0, 0.03, n)
    out["Seffern"] = (muskingum(qg, 6) + muskingum(qm, 2)) * rng.lognormal(0, 0.03, n)
    out["Regen_Seffern"] = rm
    # System 2
    r2 = rain_series(n, rng, storms_train + storms_test)
    rm2 = np.maximum(r2 * rng.lognormal(0, 0.3, n), 0)
    qe = catchment(r2, 150)
    qz = catchment(rm2, 60)
    rel, lvl = reservoir(qe)
    out["Echtershausen"] = qe * rng.lognormal(0, 0.03, n)
    out["Stausee_W"] = lvl + rng.normal(0, 0.002, n)
    out["Wiersdorf"] = (muskingum(rel, 2) + muskingum(qz, 2)) * rng.lognormal(0, 0.03, n)
    out["Regen_Wiersdorf"] = rm2
    return pd.DataFrame(out, index=idx)


if __name__ == "__main__":
    d = generate()
    print(d.describe().T[["mean", "max"]])
    ty = len(d) - 8760
    for c in ("Giesdorf", "Seffern", "Echtershausen", "Wiersdorf"):
        print(f"{c:<14} max Training {d[c].iloc[:ty].max():6.1f}   max Test {d[c].iloc[ty:].max():6.1f}")
