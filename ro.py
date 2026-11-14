"""Runoff-to-runoff-Baselines: Datensatz, Modelle, Metriken.

Aufgabe: Q_unten(t) aus Q_oben(t-1 ... t-L) (+ optional eigene Vergangenheit, Zusatzfeatures).
Setting A: mit eigener Historie (Spalten "<ziel>|lagK"), Setting B: ohne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def make_dataset(df, target, sources, max_lag=24, own_lags=0, extras=(), extra_lags=6):
    cols = {}
    for s in sources:
        for k in range(1, max_lag + 1):
            cols[f"{s}|lag{k}"] = df[s].shift(k)
    for k in range(1, own_lags + 1):
        cols[f"{target}|lag{k}"] = df[target].shift(k)
    for e in extras:
        for k in range(1, extra_lags + 1):
            cols[f"{e}|lag{k}"] = df[e].shift(k)
    X = pd.DataFrame(cols, index=df.index)
    y = df[target]
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


class Model:
    name = "?"

    def __init__(self, target, sources):
        self.target, self.sources = target, sources
        self.own = f"{target}|lag1"

    def base(self, X):
        return X[self.own].to_numpy() if self.own in X else np.zeros(len(X))

    def fit(self, X, y):
        return self


class Persistence(Model):
    name = "Persistenz"

    def predict(self, X):
        return self.base(X)


class ScaledLag(Model):
    """Summe der Oberlieger um k Stunden verschoben, mit einem Skalar (Flächenverhältnis/Zwischenzufluss)."""
    name = "Skaliert+Lag"

    def fit(self, X, y):
        best = None
        yv = y.to_numpy()
        for k in range(1, 1 + max(int(c.split("lag")[1]) for c in X.columns if c.split("|")[0] in self.sources)):
            xs = sum(X[f"{s}|lag{k}"].to_numpy() for s in self.sources)
            c = float(xs @ yv / (xs @ xs))
            sse = float(((yv - c * xs) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, k, c)
        _, self.k, self.c = best
        return self

    def predict(self, X):
        return self.c * sum(X[f"{s}|lag{self.k}"].to_numpy() for s in self.sources)


class NNLSTransfer(Model):
    """Nichtnegative lineare Übertragungsfunktion (Muskingum-artig), ohne Achsenabschnitt."""
    name = "NNLS-Transfer"

    def fit(self, X, y):
        self.cols = list(X.columns)
        self.coef, _ = nnls(X.to_numpy(float), y.to_numpy(float))
        return self

    def predict(self, X):
        return X[self.cols].to_numpy(float) @ self.coef


class RidgeARX(Model):
    name = "Ridge-ARX"

    def fit(self, X, y):
        self.m = Ridge(alpha=1.0).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GBM(Model):
    name = "GBM"

    def fit(self, X, y):
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)
        self.m.fit(X, y.to_numpy() - self.base(X))
        return self

    def predict(self, X):
        return self.m.predict(X) + self.base(X)


class LinearPlusGBM(Model):
    """Linearer Kern (extrapoliert) + GBM auf dem Residuum."""
    name = "NNLS+GBM"

    def fit(self, X, y):
        self.lin = NNLSTransfer(self.target, self.sources).fit(X, y)
        res = y.to_numpy() - self.lin.predict(X)
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)
        self.m.fit(X, res)
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.m.predict(X)


def nse(o, p):
    return float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2))


def peak_stats(y: pd.Series, pred: np.ndarray, q=0.98, dist=72, win=36):
    """Fehler der Scheitelhöhe (relativ) und des Zeitpunkts (Stunden) an den größten beobachteten Spitzen."""
    obs = y.to_numpy()
    pk, _ = find_peaks(obs, height=np.quantile(obs, q), distance=dist)
    rel, dt = [], []
    for i in pk:
        a, b = max(0, i - win), min(len(obs), i + win + 1)
        io, ip = a + int(np.argmax(obs[a:b])), a + int(np.argmax(pred[a:b]))
        rel.append((pred[a:b].max() - obs[a:b].max()) / obs[a:b].max())
        dt.append((ip - io) * (y.index[1] - y.index[0]).total_seconds() / 3600)
    return len(pk), float(np.median(rel)) if rel else np.nan, float(np.median(np.abs(dt))) if dt else np.nan


def evaluate(models, Xtr, ytr, Xte, yte, ref_name):
    rows, preds = [], {}
    for M in models:
        M.fit(Xtr, ytr)
        p = np.maximum(M.predict(Xte), 0.0)
        preds[M.name] = p
    o = yte.to_numpy()
    mse_ref = float(((o - preds[ref_name]) ** 2).mean())
    for name, p in preds.items():
        n, rel, dt = peak_stats(yte, p)
        rows.append(
            dict(model=name, nse=nse(o, p), kge=kge(o, p), rmse=float(np.sqrt(((o - p) ** 2).mean())),
                 skill=float(1 - ((o - p) ** 2).mean() / mse_ref), peaks=n, peak_err=rel, peak_dt_h=dt)
        )
    return pd.DataFrame(rows), preds
