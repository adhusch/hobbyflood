"""Routing-Modell kalibrieren und prüfen: Prüm, Enz, Nims (LfU-Pakete 1989 bis 2022).

  python routing_fit.py [--jobs 2] [--boot 40] [--quick]

Gruppen (Unterlieger <- Oberlieger, Strecke in km nach Fluss-km der Pegel; Enz bis Prümzurlay geschätzt):
  Echtershausen <- Prüm 2 (41,1)        Wiersdorf <- Echtershausen (6,3, Stausee Bitburg dazwischen: Sonderfall, c bis 12 m/s)
  Prümzurlay <- Wiersdorf (24,3) + Sinspelt/Enz (19,9)        Seffern <- Giesdorf (18,5)        Alsdorf-Oberecken <- Seffern (28,8)
Kalibrierung auf allen Ereignissen mit Scheitel des Unterliegers über dem 95-%-Quantil (Fenster 60 h vor bis 24 h nach dem Scheitel, ohne die vier
Hochwasser mit abgeschnittenem Wiersdorf bei Strecken mit Wiersdorf). Prüfung: Leave-one-flood-out über die neun Hochwasser mit Prümzurlay > 400 cm
(Training ohne alle Ereignisse im Abstand von 3,5 Tagen), Vergleich mit festem Lag und mit konstanter Wellengeschwindigkeit; Vorhersage mit
Beobachtung bis zum Ausgabezeitpunkt, danach Oberlieger konstant. Bootstrap über Ereignisse für die Unsicherheit von c(Q).
Ausgaben: data/routing_params.json, routing_skill_floods.csv, routing_forecast_skill.csv, routing_sim_floods.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import floods as F
import pkgdata
import routing as R

HERE = Path(__file__).parent
DT = R.DT_H


@dataclass
class Group:
    key: str
    dn: str
    links: list[tuple[str, float]]
    cmax: float = R.C_MAX
    note: str = ""


GROUPS = [
    Group("EC", "Echtershausen", [("Prüm 2", 41.1)], note="Zwischengebiet liefert den größten Teil (Prüm 2 hat 16 % der Fläche)"),
    Group("WI", "Wiersdorf", [("Echtershausen", 6.3)], cmax=12.0, note="Stausee Bitburg dazwischen (Sonderfall, Abgabe gesteuert)"),
    Group("PZ", "Prümzurlay", [("Wiersdorf", 24.3), ("Sinspelt", 19.9)], note="Enz mündet oberhalb; Strecke Sinspelt bis Prümzurlay geschätzt"),
    Group("SE", "Seffern", [("Giesdorf", 18.45)], note="Nims, mündet unterhalb von Prümzurlay in die Prüm"),
    Group("AL", "Alsdorf-Oberecken", [("Seffern", 28.81)], note="Nims"),
]
G = {g.key: g for g in GROUPS}
PRE_TEST, POST_TEST = 48, 36          # Testfenster um den Scheitel in Prümzurlay (h), davon 12 h Einschwingen
FC_OFFSETS = np.arange(-30, 1)         # Ausgabezeitpunkte relativ zum Scheitel in Prümzurlay (h)
HORIZON_H = 12
_S: dict = {}                          # gemeinsamer Zustand für Prozesse (fork)


def ups(g: Group) -> list[str]:
    return [u for u, _ in g.links]


# ------------------------------------------------------------------ Hilfen
def test_window(Q: pd.DataFrame, ipz: int, names: list[str]) -> dict[str, np.ndarray]:
    a, b = int(PRE_TEST / DT), int(POST_TEST / DT)
    idx = ipz + np.arange(-a, b + 1)
    return {n: Q[n].to_numpy()[idx][None, :] for n in names}


def metrics(obs: np.ndarray, sim: np.ndarray, spin: int) -> dict:
    o, s = obs[spin:], sim[spin:]
    io, is_ = int(o.argmax()), int(s.argmax())
    return dict(nse=R.nse(o, s), rmse=float(np.sqrt(((o - s) ** 2).mean())), peak_obs=float(o.max()), peak_sim=float(s.max()),
                peak_err_pct=float(100 * (s.max() - o.max()) / o.max()), peak_dt_h=float((is_ - io) * DT))


def clipped_flood(W: pd.DataFrame, Q: pd.DataFrame, ipz: int) -> bool:
    a, b = int(PRE_TEST / DT), int(POST_TEST / DT)
    sl = slice(ipz - a, ipz + b + 1)
    return bool(R.clipped_window(W["Wiersdorf"].to_numpy()[sl][None, :], Q["Wiersdorf"].to_numpy()[sl][None, :])[0])


def forecast_any(model, ups_: list[str], Q: pd.DataFrame, t0: np.ndarray, correct_h: float | None, hist_h: float = 48) -> dict:
    """Vorhersage für Modell oder Lag-Modell: Oberlieger bis t0 beobachtet, danach konstant; Fehlerkorrektur am Unterlieger abklingend."""
    a, H = int(hist_h / DT), int(HORIZON_H / DT)
    idx = t0[:, None] + np.arange(-a, H + 1)[None, :]
    X = {}
    for u in ups_:
        v = Q[u].to_numpy()[idx].copy()
        v[:, a + 1:] = v[:, a:a + 1]
        X[u] = v
    y = model.route(X)
    sim = y[:, a + 1:]
    err0 = Q[model.dn].to_numpy()[t0] - y[:, a]
    lead = (np.arange(1, H + 1) * DT)[None, :]
    corr = sim + (err0[:, None] * np.exp(-lead / correct_h) if correct_h else 0.0)
    return dict(sim=sim, corr=np.maximum(corr, 0))


# ------------------------------------------------------------------ Gesamtanpassung
def global_fits(Q, W, log=print) -> dict:
    out = {}
    for g in GROUPS:
        t0 = time.time()
        d = R.make_data(Q, W, g.dn, ups(g))
        full, cf, dx0 = R.fit(g.dn, g.links, d, Q, dx_grid=DX_GRID, cmax=g.cmax)
        tied, ct, dxt = R.fit(g.dn, g.links, d, Q, dx_grid=DX_GRID, tied=True, cmax=g.cmax)
        lag = R.fit_lag_model(g.dn, ups(g), d)
        ins = {k: R.nse(d.Y[d.mask], m.route(d.X)[d.mask]) for k, m in (("full", full), ("tied", tied), ("lag", lag))}
        out[g.key] = dict(data=d, full=full, tied=tied, lag=lag, dx0=dx0, dx0_tied=dxt, cost=dict(full=cf, tied=ct), nse_in=ins)
        log(f"[global] {g.key} {g.dn}: {len(d.peaks)} Ereignisse, dx0 {dx0}, NSE in-sample voll {ins['full']:.3f} fest-c {ins['tied']:.3f} "
            f"Lag {ins['lag']:.3f}, Zielwert {cf:.2f} gegen {ct:.2f} ({time.time() - t0:.0f} s)")
    return out


DX_GRID = (100, 20, 12, 8, 5, 3, 2, 1.2)


# ------------------------------------------------------------------ Leave-one-flood-out
def fold(fi: int) -> dict:
    """Ein Hochwasser auslassen, alle Gruppen neu anpassen (Warmstart aus der Gesamtanpassung), auf dem Hochwasser prüfen und Vorhersagen rechnen."""
    Q, W, GL, floods = _S["Q"], _S["W"], _S["GL"], _S["floods"]
    fl = floods[fi]
    ipz = int(Q.index.get_loc(fl["t_peak"]))
    clipped = clipped_flood(W, Q, ipz)
    models: dict[str, dict] = {}
    rows, sims, fcs = [], {}, {}
    Qp = Q
    for g in GROUPS:
        t_fold = time.time()
        d_all = GL[g.key]["data"]
        keep = np.abs(d_all.peaks - ipz) > int(3.5 * 24 / DT)
        d = d_all.subset(keep)
        full, _, _ = R.fit(g.dn, g.links, d, Q, dx_grid=DX_GRID, start=GL[g.key]["full"], cmax=g.cmax)
        tied, _, _ = R.fit(g.dn, g.links, d, Q, dx_grid=DX_GRID, tied=True, start=GL[g.key]["tied"], cmax=g.cmax)
        lag = R.fit_lag_model(g.dn, ups(g), d)
        models[g.key] = dict(full=full, tied=tied, lag=lag)
        # Wiersdorf-Messwert abgeschnitten: Eingang für Prümzurlay aus Echtershausen über das Modell der Strecke Echtershausen -> Wiersdorf ersetzen
        usable, chain = True, False
        if g.dn == "Wiersdorf" and clipped:
            usable = False
        if g.key == "PZ" and clipped:
            chain = True
            a, b = int(PRE_TEST / DT) + int(60 / DT), int(POST_TEST / DT) + int(36 / DT)
            sl = slice(ipz - a, ipz + b + 1)
            Qp = Q.copy()
            Qp.iloc[sl, Qp.columns.get_loc("Wiersdorf")] = models["WI"]["full"].route({"Echtershausen": Q["Echtershausen"].to_numpy()[sl][None, :]})[0]
        Qg = Qp if g.key == "PZ" else Q
        Xt = test_window(Qg, ipz, ups(g))
        obs = test_window(Qg, ipz, [g.dn])[g.dn][0]
        spin = int(12 / DT)
        for var, m in (("full", full), ("tied", tied), ("lag", lag)):
            sim = m.route(Xt)[0]
            sims[(g.key, var)] = sim
            if usable:
                rows.append(dict(flood=f"{fl['t_peak']:%Y-%m-%d}", group=g.key, dn=g.dn, variant=var, chain=chain, **metrics(obs, sim, spin)))
        if usable:
            t0s = ipz + FC_OFFSETS * int(1 / DT)
            pk = float(Qg[g.dn].to_numpy()[ipz - int(36 / DT): ipz + int(36 / DT) + 1].max())
            H = int(HORIZON_H / DT)
            tidx = t0s[:, None] + np.arange(1, H + 1)[None, :]
            fc = dict(obs=Qg[g.dn].to_numpy()[tidx], pers=np.repeat(Qg[g.dn].to_numpy()[t0s][:, None], H, axis=1), peak=pk, offsets=FC_OFFSETS)
            for var, m in (("full", full), ("tied", tied), ("lag", lag)):
                f = forecast_any(m, ups(g), Qg, t0s, 6.0)
                fc[var] = f["corr"]
                fc[var + "_raw"] = f["sim"]
            fcs[g.key] = fc
    win = dict(obs={g.dn: Q[g.dn].to_numpy()[ipz - int(36 / DT): ipz + int(36 / DT) + 1] for g in GROUPS})
    a0 = int(PRE_TEST / DT) - int(36 / DT)
    series = {}
    for g in GROUPS:
        series[g.key] = {v: sims[(g.key, v)][a0:a0 + int(72 / DT) + 1] for v in ("full", "tied", "lag")}
    return dict(fi=fi, clipped=clipped, rows=rows, series=series, fc=fcs, models={k: {v: (m.to_dict()) for v, m in d.items()} for k, d in models.items()})


# ------------------------------------------------------------------ Bootstrap
def boot(args) -> dict:
    key, seed = args
    GL, Q = _S["GL"], _S["Q"]
    g = G[key]
    d = GL[key]["data"]
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(d.peaks), len(d.peaks))
    ds = R.Data(d.peaks[k], {u: v[k] for u, v in d.X.items()}, d.Y[k], d.mask[k], d.wts[k], d.times[k])
    m, _, _ = R.fit(g.dn, g.links, ds, Q, dx_grid=(GL[key]["dx0"],), start=GL[key]["full"], cmax=g.cmax, max_nfev=25)
    return dict(key=key, lc=[list(map(float, l.lc)) for l in m.links], a=[float(l.a) for l in m.links])


# ------------------------------------------------------------------ Auswertung
def pool_forecast(results: list[dict]) -> pd.DataFrame:
    rows = []
    H = int(HORIZON_H / DT)
    for g in GROUPS:
        acc = {v: [] for v in ("pers", "lag", "tied", "full", "lag_raw", "tied_raw", "full_raw")}
        obs_all, wts = [], []
        for r in results:
            fc = r["fc"].get(g.key)
            if fc is None:
                continue
            for v in acc:
                acc[v].append(fc[v] / fc["peak"])
            obs_all.append(fc["obs"] / fc["peak"])
        if not obs_all:
            continue
        O = np.concatenate(obs_all)
        mse_p = ((np.concatenate(acc["pers"]) - O) ** 2).mean(axis=0)
        for v in acc:
            P = np.concatenate(acc[v])
            mse = ((P - O) ** 2).mean(axis=0)
            for h in range(H):
                rows.append(dict(group=g.key, dn=g.dn, variant=v, lead_h=(h + 1) * DT, nrmse=float(np.sqrt(mse[h])),
                                 skill_vs_persistence=float(1 - mse[h] / mse_p[h]) if mse_p[h] > 0 else np.nan, n=int(len(O))))
    return pd.DataFrame(rows)


def run(a) -> None:
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    floods = F.find_floods(W["Prümzurlay"], 400)
    if a.quick:
        floods = floods[-2:]
    t_all = time.time()
    GL = global_fits(Q, W)
    _S.update(Q=Q, W=W, GL=GL, floods=floods)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.jobs) as pool:
        results = []
        for r in pool.imap_unordered(fold, range(len(floods))):
            results.append(r)
            print(f"[LOFO] Hochwasser {floods[r['fi']]['t_peak']:%d.%m.%Y} fertig ({len(results)}/{len(floods)}, {time.time() - t_all:.0f} s)", flush=True)
        results.sort(key=lambda r: r["fi"])
        boots = pool.map(boot, [(g.key, 1000 + i) for g in GROUPS for i in range(a.boot)]) if a.boot else []
    print(f"[Bootstrap] {len(boots)} Anpassungen ({time.time() - t_all:.0f} s)", flush=True)

    # --- Ausgaben
    skill = pd.DataFrame([row for r in results for row in r["rows"]])
    skill.to_csv(HERE / "data" / "routing_skill_floods.csv", index=False, float_format="%.4g")
    fcs = pool_forecast(results)
    fcs.to_csv(HERE / "data" / "routing_forecast_skill.csv", index=False, float_format="%.4g")

    qgrid = np.exp(np.linspace(np.log(0.5), np.log(500), 60))
    params = {}
    for g in GROUPS:
        gl = GL[g.key]
        links = []
        for k, lk in enumerate(gl["full"].links):
            bl = [np.array(b["lc"][k]) for b in boots if b["key"] == g.key]
            curves = np.array([np.exp(np.interp(np.log(qgrid), lk.lq, v)) for v in bl]) if bl else None
            nodes_ci = {}
            if bl:
                B = np.array(bl)
                T = lk.length_km / (3.6 * np.exp(B))
                nodes_ci = dict(c_lo=np.exp(np.percentile(B, 5, axis=0)).tolist(), c_hi=np.exp(np.percentile(B, 95, axis=0)).tolist(),
                                T_lo=np.percentile(T, 5, axis=0).tolist(), T_hi=np.percentile(T, 95, axis=0).tolist())
            links.append(dict(up=lk.up, length_km=lk.length_km, n=lk.n, nodes_q=list(lk.nodes_q), c=np.exp(lk.lc).tolist(), a=lk.a,
                              c_at_bound=[bool(c >= 0.97 * g.cmax) for c in np.exp(lk.lc)],
                              T_h=(lk.length_km / (3.6 * np.exp(lk.lc))).tolist(),
                              c_tied=float(np.exp(gl["tied"].links[k].lc[0])), a_tied=gl["tied"].links[k].a, n_tied=gl["tied"].links[k].n,
                              curve=dict(q=qgrid.tolist(), c=lk.c(qgrid).tolist(),
                                         lo=np.percentile(curves, 5, axis=0).tolist() if curves is not None else None,
                                         hi=np.percentile(curves, 95, axis=0).tolist() if curves is not None else None),
                              **nodes_ci))
        params[g.key] = dict(dn=g.dn, note=g.note, n_events=int(len(gl["data"].peaks)), c0=gl["full"].c0, dx0=gl["dx0"], links=links,
                             nse_in=gl["nse_in"], cost=gl["cost"], lag=gl["lag"].to_dict(), full=gl["full"].to_dict(), tied=gl["tied"].to_dict())
    (HERE / "data" / "routing_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")

    sims = {}
    for r in results:
        fl = floods[r["fi"]]
        sims[f"{fl['t_peak']:%Y-%m-%d}"] = dict(
            t_peak=f"{fl['t_peak']:%Y-%m-%d %H:%M}", clipped=r["clipped"],
            series={k: {v: [round(float(x), 2) for x in arr] for v, arr in d.items()} for k, d in r["series"].items()},
            fc={k: dict(offsets=FC_OFFSETS.tolist(), **{v: np.round(f[v][::3][:, ::1], 2).tolist() for v in ("full", "lag")}) for k, f in r["fc"].items()},
            models=r["models"])
    (HERE / "data" / "routing_sim_floods.json").write_text(json.dumps(sims, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"fertig in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args())
