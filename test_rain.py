"""Prüfungen für lateral.py / rain.py (python test_rain.py). NUR synthetische Daten: Gewichte, Lückenbehandlung, Massenerhaltung des Zwischengebiets,
Wiederfinden bekannter Regen-Abfluss-Parameter im gemeinsamen Fit mit dem Routing. Mit echten Regendaten hat das nichts zu tun."""
from __future__ import annotations

import numpy as np
import pandas as pd

import lateral as L
import rain as R
import routing as RT


def storms(n_steps: int, rng, mean_gap_h=90.0) -> np.ndarray:
    p = np.zeros(n_steps)
    t = int(rng.exponential(mean_gap_h) * 4)
    while t < n_steps - 200:
        dur = int(rng.uniform(6, 30) * 4)
        peak = rng.gamma(2.0, 2.5)                                  # mm/h
        shape = np.sin(np.linspace(0.05, np.pi - 0.05, dur)) ** 1.5 * rng.uniform(0.6, 1.4, dur)
        p[t:t + dur] += peak * shape
        t += dur + int(rng.exponential(mean_gap_h) * 4)
    return p


def test_weights_and_gaps() -> None:
    st = {"A": (50.20, 6.40), "B": (50.30, 6.60), "C": (50.10, 6.30)}
    w = R.idw_weights(st, (50.19, 6.41))
    assert abs(sum(w.values()) - 1) < 1e-12 and w["A"] > w["B"] and w["A"] > w["C"]
    idx = pd.date_range("2020-01-01", periods=48, freq="1h")
    s = {k: pd.Series(1.0, index=idx) for k in st}
    s["A"].iloc[10] = np.nan
    s["B"].iloc[10] = np.nan
    s["C"].iloc[10] = np.nan
    s["A"].iloc[20] = np.nan
    a = R.areal_series(s, w)
    assert np.isnan(a.iloc[10]) and abs(a.iloc[20] - 1) < 1e-12 and abs(a.iloc[0] - 1) < 1e-12, "Renormierung der Gewichte"
    print("Gewichte und Lücken: ok")


def test_mass() -> None:
    rng = np.random.default_rng(1)
    p = storms(4000, rng)
    api = R.api_series(p)
    lat = L.Lateral("x", 120.0, 0.35, np.log(30.0), np.log(2.0), np.log(15.0))
    q = L.response(L.Rain(p[None], api[None], 0), lat)[0]
    phi = lat.phi_max * (1 - np.exp(-api / np.exp(lat.log_s)))
    vol_in = (phi * p).sum() * 0.25 * 120.0 * 1e6 / 1000.0          # m³ (mm * km² -> m³: 1 mm auf 1 km² = 1000 m³)
    vol_out = q.sum() * 0.25 * 3600.0
    assert abs(vol_out / vol_in - 1) < 0.03, (vol_out, vol_in)
    print("Massenerhaltung Zwischengebiet: ok (Verhältnis", round(vol_out / vol_in, 4), ")")


def test_recover() -> None:
    """Synthetik: Q_dn = 3 + 1,1 * Kaskade(Q_oben) + Zwischengebiet aus Regen; die Anpassung muss Regen-Abfluss-Anteil und Routing gemeinsam finden."""
    rng = np.random.default_rng(7)
    T = 4 * 24 * 900
    idx = pd.date_range("2000-01-01", periods=T, freq="15min")
    p_up, p_lat = storms(T, rng), storms(T, rng)
    api_up, api_lat = R.api_series(p_up), R.api_series(p_lat)
    up = 1.5 + L.response(L.Rain(p_up[None], api_up[None], 0), L.Lateral("u", 80.0, 0.45, np.log(40.0), np.log(3.0), np.log(20.0)))[0]
    true_lat = L.Lateral("lat", 130.0, 0.35, np.log(30.0), np.log(2.0), np.log(15.0))
    link = RT.Link("up", 20.0, (4.0, 20.0, 60.0), 4, np.log([1.2, 1.5, 1.8]), 1.1)
    truth = RT.Model("dn", [link], 3.0, lat=[true_lat])
    q_lat = L.response(L.Rain(p_lat[None], api_lat[None], 0), true_lat)[0]
    q_dn = truth.route({"up": up[None]})[0] + q_lat
    q_dn = q_dn + rng.normal(0, 0.05, T)
    Q = pd.DataFrame({"up": up, "dn": q_dn}, index=idx)
    rain = {"lat": pd.DataFrame({"p": p_lat, "api": api_lat}, index=idx)}
    d = RT.make_data(Q, None, "dn", ["up"], rain=rain)
    assert d.P is not None and len(d.peaks) >= 40, len(d.peaks)
    tr = np.arange(len(d.peaks)) % 4 != 0
    dtr, dte = d.subset(tr), d.subset(~tr)
    start = [L.Lateral("lat", 130.0)]
    m, cost, dx0 = RT.fit("dn", [("up", 20.0)], dtr, Q, dx_grid=(10, 5), lat=start, max_nfev=60)
    l = m.lat[0]
    print("Anpassung: phi_max %.2f (0,35)  S %.1f (30)  K1 %.1f (2)  K2 %.1f (15)  a %.2f (1,1)  c0 %.2f (3,0)" % (
        l.phi_max, np.exp(l.log_s), np.exp(l.log_k1), np.exp(l.log_k2), m.links[0].a, m.c0))
    sim_rain = m.route(dte.X, dte.P)
    sim_norain = m.route(dte.X, None)
    nse_rain = RT.nse(dte.Y[dte.mask], sim_rain[dte.mask])
    nse_norain = RT.nse(dte.Y[dte.mask], RT.Model(m.dn, m.links, m.c0).route(dte.X)[dte.mask])
    print("Test-Ereignisse %d: NSE mit Regenanteil %.3f, Routing allein (gleiche Parameter) %.3f" % (len(dte.peaks), nse_rain, nse_norain))
    assert nse_rain > 0.97, nse_rain
    assert nse_rain - nse_norain > 0.05
    assert abs(l.phi_max * 1.0 - 0.35) < 0.12 and 0.5 < np.exp(l.log_k1) / 2.0 < 2.0 and 0.5 < np.exp(l.log_k2) / 15.0 < 2.0
    assert abs(m.links[0].a - 1.1) < 0.1
    del sim_norain


def test_terms() -> None:
    """Erweiterungen des Abflussbeiwerts: ohne terms identisch zum Grundmodell, Intensität und Ereignisfeuchte erhöhen den Abfluss bei starkem Regen,
    Parameter (Intensität bzw. Ereignisfeuchte) werden aus synthetischen Ereignissen wiedergefunden."""
    rng = np.random.default_rng(3)
    p = storms(6000, rng)
    api, api_s = R.api_series(p), R.api_series(p, tau_h=R.TAU_API_S_H)
    rn = L.Rain(p[None], api[None], 0, api_s[None])
    base = L.Lateral("x", 120.0, 0.3, np.log(30.0), np.log(2.0), np.log(15.0))
    assert np.allclose(L.response(rn, base), L.response(L.Rain(p[None], api[None], 0), base)), "ohne terms identisch"
    assert base.with_terms(()).theta() == base.theta() and len(base.with_terms(("int", "acc")).theta()) == 7
    ext = L.Lateral.from_dict(base.with_terms(("int",)).to_dict())
    assert ext.terms == ("int",) and len(L.Lateral.from_dict(base.to_dict()).theta()) == 4
    phi0 = L.runoff_coefficient(rn, base)[0]
    for terms in (("int",), ("acc",), ("int", "acc")):
        lt = base.with_terms(terms)
        lt.w_acc, lt.rho, lt.log_pc = 4.0, 0.4, float(np.log(3.0))
        phi = L.runoff_coefficient(rn, lt)[0]
        assert (phi >= phi0 - 1e-12).all() and phi.max() <= 1.0 + 1e-12 and phi.mean() > phi0.mean() + 0.005, terms
    # Wiederfinden: Ereignisse, deren Abfluss nur mit der Erweiterung stimmt; Anpassung mit Linear-Routing (a, c0 bekannt) über least_squares
    from scipy.optimize import least_squares
    for terms, truth in ((("int",), dict(rho=0.5, log_pc=np.log(6.0))), (("acc",), dict(w_acc=3.0))):
        t_lat = L.Lateral("x", 120.0, 0.25, np.log(40.0), np.log(2.0), np.log(15.0), terms, truth.get("rho", 0.2), truth.get("log_pc", np.log(10.0)), truth.get("w_acc", 1.0))
        q = L.response(rn, t_lat)[0] + rng.normal(0, 0.02, 6000)
        start = base.with_terms(terms)
        lo, hi = start.bounds()
        f = lambda th: (L.response(rn, start.with_theta(th))[0] - q)
        res = least_squares(f, np.clip(np.array(start.theta()), np.array(lo) + 1e-6, np.array(hi) - 1e-6), bounds=(lo, hi), x_scale="jac", max_nfev=80)
        got = start.with_theta(res.x)
        print("Erweiterung", terms, "wahr", {k: round(float(v), 2) for k, v in truth.items()}, "angepasst", {k: round(float(getattr(got, k)), 2) for k in truth},
              "phi_max %.2f (0,25)" % got.phi_max)
        for k, v in truth.items():
            assert abs(getattr(got, k) - v) < 0.25 * max(abs(v), 1.0), (terms, k, getattr(got, k), v)
        assert abs(got.phi_max - 0.25) < 0.08
    print("Erweiterungen des Abflussbeiwerts: ok")


def test_wetness() -> None:
    """Erweiterung bf (synthetisch): ohne Feuchtewert Fehler; Faktor B^g / (B^g + b50^g) monoton in B, bei B = b50 genau 1/2, bei großem B gegen 1;
    Wasserbilanz: abflusswirksamer Regen = Faktor * Regen ohne bf."""
    lt0 = L.Lateral("X", 10.0, 0.6, float(np.log(20.0)), float(np.log(2.0)), float(np.log(10.0)))
    lt = lt0.with_terms(("bf",))
    lt.log_b50, lt.log_gb = float(np.log(0.5)), float(np.log(1.5))
    rain = np.zeros((4, 400)); rain[:, 100:120] = 4.0
    api = np.full_like(rain, 30.0)
    try:
        L.runoff_coefficient(L.Rain(rain, api, 50), lt)
        raise AssertionError("ohne bf hätte es einen Fehler geben müssen")
    except ValueError:
        pass
    B = np.array([0.05, 0.5, 2.0, 50.0])
    r = L.Rain(rain, api, 50, None, B)
    f = L.runoff_coefficient(r, lt)[:, 0] / L.runoff_coefficient(L.Rain(rain, api, 50), lt0)[:, 0]
    assert np.all(np.diff(f) > 0) and abs(f[1] - 0.5) < 1e-12 and f[3] > 0.99 and f[0] < 0.05, f
    v = L.response(r, lt).sum(axis=1) / L.response(L.Rain(rain, api, 50), lt0).sum(axis=1)
    assert np.allclose(v, f, rtol=1e-6), (v, f)
    d = lt.to_dict(); assert L.Lateral.from_dict(d) == lt
    la = lt0.with_terms(("bfa",)); la.w_bf = 40.0                       # additiv: x = API + w * B
    ra = L.Rain(rain, api, 50, None, B)
    x = 30.0 + 40.0 * B
    assert np.allclose(L.runoff_coefficient(ra, la)[:, 0], 0.6 * (1 - np.exp(-x / 20.0)))
    assert L.Lateral.from_dict(la.to_dict()) == la
    print("Gebietsfeuchte (bf, bfa): ok")


def test_saturation() -> None:
    """Sättigungsüberschuss sat/satx (synthetisch): SCS-Steigung 0 bis lam * Sv, monoton, gegen 1; rs = 0 gibt das Grundmodell; Wasserbilanz:
    Volumen = Summe (phi_a + (1 - phi_a) * rs * Steigung) * Regen; nasses Gebiet (größeres B) gibt mehr Abfluss; satx rechnet wie sat, ohne
    angepasste Parameter."""
    pe = np.linspace(0, 400, 401)
    g = L.scs_slope(pe, np.array(150.0))
    assert np.all(g[pe <= 30] == 0) and np.all(np.diff(g) >= -1e-12) and g[-1] > 0.75 and g.max() < 1
    lt0 = L.Lateral("X", 25.0, 0.5, float(np.log(300.0)), float(np.log(3.0)), float(np.log(12.0))).with_terms(("bfa",))
    lt0.w_bf = 100.0
    rain = np.zeros((2, 900)); rain[:, 200:296] = 5.0                  # 120 mm in 24 h
    api = np.full_like(rain, 20.0) + np.cumsum(rain, axis=1) * 0.25
    aps = np.zeros_like(rain)
    for t in range(1, rain.shape[1]):
        aps[:, t] = aps[:, t - 1] * np.exp(-0.25 / 24.0) + rain[:, t] * 0.25
    r = L.Rain(rain, api, 100, aps, np.array([0.2, 1.5]))
    ls = lt0.with_terms(("bfa", "sat")); ls.rs, ls.log_sev, ls.g_ev, ls.log_kf = 0.0, float(np.log(150.0)), 0.5, float(np.log(2.0))
    assert np.allclose(L.response(r, ls), L.response(r, lt0))
    ls.rs = 1.0
    phi = L.runoff_coefficient(r, ls)
    vol = L.response(L.Rain(np.pad(rain, ((0, 0), (0, 3000))), np.pad(api, ((0, 0), (0, 3000)), mode="edge"), 100,
                            np.pad(aps, ((0, 0), (0, 3000))), r.bf), ls).sum(axis=1) * 3.6 / 25.0 * 0.25
    assert np.allclose(vol, (phi * rain)[:, 100:].sum(axis=1) * 0.25, rtol=1e-3), (vol, (phi * rain).sum(axis=1) * 0.25)
    assert vol[1] > vol[0] > L.response(r, lt0)[0].sum() * 3.6 / 25.0 * 0.25
    lx = ls.with_terms(("bfa", "satx"))
    assert np.allclose(L.response(r, lx), L.response(r, ls)) and "rs" not in lx.keys() and L.Lateral.from_dict(lx.to_dict()) == lx
    print("Sättigungsüberschuss (sat, satx): ok")


def test_nonlinear() -> None:
    """Nichtlineare Speicher (nlr, synthetisch): Wasserbilanz exakt, a = 0 wie die linearen Speicher (bis auf die Annahme konstanten Zuflusses
    je Schritt), a > 0 spitzerer und früherer Scheitel, kleine Wellen (S/K weit unter oref) fast unverändert; nlrx rechnet wie nlr."""
    p = np.zeros((1, 3000)); p[0, 100:140] = 12.0
    lin = L._two_stores(p, 4.0, 20.0, 0.25)
    nl0 = L._two_stores(p, 4.0, 20.0, 0.25, (0.5, 0.0))
    nl = L._two_stores(p, 4.0, 20.0, 0.25, (0.5, 0.7))
    for o in (lin, nl0, nl):
        assert abs(o.sum() * 0.25 - p.sum() * 0.25) < 1e-6
    assert np.max(np.abs(nl0 - lin)) < 0.05 * lin.max()
    assert nl.max() > 2 * lin.max() and nl.argmax() < lin.argmax()
    small = p * 0.01
    assert abs(L._two_stores(small, 4.0, 20.0, 0.25, (5.0, 0.7)).max() / L._two_stores(small, 4.0, 20.0, 0.25).max() - 1) < 0.02
    lt = L.Lateral("X", 30.0, 0.5, float(np.log(300.0)), float(np.log(3.0)), float(np.log(12.0))).with_terms(("bfa", "nlr"))
    lt.log_oref, lt.a_nl = float(np.log(0.5)), 0.6
    r = L.Rain(np.pad(p, ((0, 0), (0, 0))), np.full_like(p, 50.0), 0, np.zeros_like(p), np.array([1.0]))
    assert np.allclose(L.response(r, lt), L.response(r, lt.with_terms(("bfa", "nlrx")))) and "a_nl" not in lt.with_terms(("bfa", "nlrx")).keys()
    print("Nichtlineare Speicher (nlr, nlrx): ok")


if __name__ == "__main__":
    test_weights_and_gaps()
    test_mass()
    test_recover()
    test_terms()
    test_wetness()
    test_saturation()
    test_nonlinear()
    print("alle Prüfungen bestanden (synthetische Daten)")
