"""Prüfungen für routing.py (python test_routing.py): C-Kern gegen numpy-Fassung, Verzögerung = L / c, Massenerhaltung,
Wiederfinden bekannter Parameter an synthetischen Daten (Zufluss mit abflussabhängiger Laufzeit, Zwischenzufluss als Verstärkung)."""
from __future__ import annotations

import numpy as np

import routing as R


def hydrograph(t, peak, t0, width, base):
    return base + peak * np.exp(-(((t - t0) / width) ** 2))


def test_kernel() -> None:
    t = np.arange(400) * 0.25
    x = np.stack([hydrograph(t, 80, 20, 6, 5), hydrograph(t, 30, 30, 10, 3)])
    lq, lc = np.log([10, 50, 200]), np.log([1.0, 1.5, 2.5])
    a, b = R.cascade(x, 8, 24.3, lq, lc), R._cascade_np(x, 8, 24.3, 0.25, lq, lc)
    assert np.abs(a - b).max() < 1e-9, "C-Kern und numpy-Fassung weichen ab"
    for n in (4, 8, 16):
        y = R.cascade(x[:1], n, 24.3, np.log([10.0]), np.log([1.3]))[0]
        lag = (y.argmax() - x[0].argmax()) * 0.25
        assert abs(lag - 24.3 / (3.6 * 1.3)) < 0.5, (n, lag)
        assert abs(y.sum() / x[0].sum() - 1) < 1e-3, "Volumen nicht erhalten"
    print("Kern: ok")


def test_recover() -> None:
    """Synthetik: bekannter Zusammenhang Q_unten = 2 + 1.2 * Kaskade(Q_oben) mit c = 1.0 (klein) bis 2.0 (groß); Anpassung muss ihn wiederfinden."""
    rng = np.random.default_rng(3)
    T, E = 337, 120
    t = np.arange(T) * 0.25
    X = np.stack([hydrograph(t, rng.uniform(10, 150), rng.uniform(30, 50), rng.uniform(4, 9), rng.uniform(2, 6)) for _ in range(E)])
    nodes = (8.0, 40.0, 150.0)
    true = R.Model("dn", [R.Link("up", 24.0, nodes, 6, np.log([1.0, 1.4, 2.0]), 1.2)], 2.0)
    Y = true.route({"up": X})
    Y = Y + rng.normal(0, 0.3, Y.shape)
    mask = np.ones_like(Y, bool)
    mask[:, :48] = False
    d = R.Data(np.arange(E), {"up": X}, Y, mask, Y.max(axis=1), np.arange(E))

    class Q:      # minimale Hülle für _nodes_for
        def __getitem__(self, k):
            import pandas as pd
            return pd.Series(np.r_[np.linspace(0, 8, 950), np.linspace(8, 40, 45), np.linspace(40, 120, 5)])

    m, cost, dx0 = R.fit("dn", [("up", 24.0)], d, Q(), dx_grid=(12, 8, 4, 2))
    c_fit = np.exp(m.links[0].lc)
    c_true = true.links[0].c(np.array(m.links[0].nodes_q))      # wahre Wellengeschwindigkeit an den Stützstellen der Anpassung
    print("Stützstellen", m.links[0].nodes_q, "c wahr", c_true.round(2), "c angepasst", c_fit.round(2), "a", round(m.links[0].a, 2), "c0", round(m.c0, 2), "n", m.links[0].n)
    seen = np.array(m.links[0].nodes_q) < 0.8 * X.max()        # Stützstellen oberhalb der Daten sind nicht bestimmt
    assert seen.sum() >= 2 and np.all(np.abs(c_fit[seen] / c_true[seen] - 1) < 0.2), "c(Q) nicht wiedergefunden"
    assert abs(m.links[0].a - 1.2) < 0.1


if __name__ == "__main__":
    test_kernel()
    test_recover()
    print("alle Prüfungen bestanden")
