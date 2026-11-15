"""Regenschwellen: Wie viel Regen in 1, 3 oder 6 h (gleichmäßig über das Einzugsgebiet, ab jetzt) braucht es beim jetzigen Zustand, bis die Kette
mit dieser Regenvorhersage die Warnschwelle mit 10 % (Vorwarnung) bzw. 50 % (Warnung) Wahrscheinlichkeit in 48 h erreicht?

Gerechnet wird genau wie auf der Live-Seite: Kette mit Regenvorhersage (kette_vorhersage), Fehlertabelle nach Zustand, Abflussstufe und
Starkregenlage, Wahrscheinlichkeit aus uncert.exceed_prob. Die Schwellen erben damit die Schwächen der Kette (Starkregen-Scheitel zu niedrig und
zu spät); die Fehlertabelle gleicht das im Mittel aus. Prüfung an den Ereignissen 2004-2022: python rain_thr.py check"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import rain as RN
import uncert as UN

HERE = Path(__file__).parent
DURS_H = (3, 6, 12, 24)
R_GRID = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 110.0, 150.0)      # mm
LEVELS = (0.1, 0.5)


def frames_of(p: dict, Qn: dict) -> dict:
    out = {k: dict(p=v, api=RN.api_series(v, C.DT, RN.TAU_API_H), api_s=RN.api_series(v, C.DT, RN.TAU_API_S_H)) for k, v in p.items()}
    return C.add_wetness(out, Qn)


def interp_tab(T) -> np.ndarray:
    T = np.asarray(T)
    L = np.arange(1, C.H + 1)
    return np.stack([np.interp(L, UN.LEADS, T[:, k]) for k in range(T.shape[1])], axis=1)


def chain_weights(k: str, info: dict) -> tuple[np.ndarray, list[str]]:
    ks = [x for x in C.chain(k) if x in C.RAINK]
    a = np.array([info[x]["area_km2"] for x in ks])
    return a / a.sum(), ks


def first_cross(P: np.ndarray, lev: float) -> float | None:
    """Kleinste Regenmenge, bei der die Wahrscheinlichkeit lev erreicht (linear zwischen den Stützstellen); None, wenn nie."""
    R = np.r_[0.0, R_GRID]
    for j in range(1, len(R)):
        if P[j] >= lev:
            if P[j] <= P[j - 1]:
                return float(R[j])
            return float(R[j - 1] + (lev - P[j - 1]) / (P[j] - P[j - 1]) * (R[j] - R[j - 1])) if P[j - 1] < lev else float(R[j - 1])
    return None


def thresholds(models: dict, target: str, Qn: dict, p_none: dict, i0: int, info: dict, tabs: list, st: int, mq: float, d: float, qv: float,
               durs=DURS_H) -> dict:
    """p_none: Gebiet -> Regen mm/h auf dem Raster, nach i0 null. tabs: Fehlertabellen des Pegels je Klasse (uncert_tables, Variante vorhersage).
    Rückgabe je Dauer: p (Wahrscheinlichkeit je Stützstelle, erste = ohne weiteren Regen), r10, r50 in mm (None = auch mit 150 mm nicht)."""
    A = C.A
    w, ks = chain_weights(target, info)
    wet = int(Qn[C.DN[target]][i0] >= UN.WET_EDGE * mq)
    out = {}

    def prob(p: dict) -> float:
        s = C.forecast(models, target, Qn, frames_of(p, Qn), np.array([i0]), "kette_vorhersage")[target][0]
        lev = int(np.digitize(max(s[A], np.max(s[A + 1:])) / mq, UN.LEVEL_EDGES)) * 2 + wet
        heavy = bool(UN.heavy_flags(sum(wi * p[x] for wi, x in zip(w, ks)), np.array([i0]))[1][0])
        tab = interp_tab(tabs[st * UN.NSUB + lev + UN.NB * heavy])
        return float(UN.exceed_prob(np.maximum(s[A + 1:], 0)[None, :], tab[None], qv, d)[0][-1])

    p0 = prob(p_none)
    for D in durs:
        P = [p0]
        for R in R_GRID:
            p = {k: v.copy() for k, v in p_none.items()}
            for k in ks:
                p[k][i0 + 1:i0 + 1 + int(D / C.DT)] = R / D
            P.append(prob(p))
            if P[-1] >= 0.9:                                   # darüber ändert sich nichts mehr
                P += [P[-1]] * (len(R_GRID) + 1 - len(P))
                break
        P = np.array(P)
        out[D] = dict(p=[round(float(x), 3) for x in P], r10=first_cross(P, LEVELS[0]), r50=first_cross(P, LEVELS[1]))
    return out


def max_sums(r: np.ndarray, i0: int, durs=DURS_H) -> dict:
    """Größte Regensumme (mm) je Dauer in den 48 h nach i0; r in mm/h auf dem 15-min-Raster."""
    x = np.nan_to_num(r[i0 + 1:i0 + 1 + C.H])
    cs = np.r_[0, np.cumsum(x)] * C.DT
    return {D: float((cs[int(D / C.DT):] - cs[:-int(D / C.DT)]).max()) for D in durs}


# ------------------------------------------------------------------ Prüfung an den Ereignissen 2004-2022 (kreuzvalidiert)
HIST = int(40 * 24 / C.DT)        # Vorlauf wie im Live-Betrieb (live.HISTORY_D)


def local_state(t: int, Qg: dict, Pg: dict, info: dict, month: np.ndarray) -> tuple:
    """Fenster wie im Live-Betrieb: Abfluss und Regen bis t, danach leer bzw. null; setzt die Zustände von chain_fc und uncert auf dieses Fenster."""
    a, b = t - HIST, t + C.H + 1
    i0 = HIST
    Qn = {dn: q[a:b].copy() for dn, q in Qg.items()}
    for q in Qn.values():
        q[i0 + 1:] = np.nan
    p_none = {k: np.nan_to_num(v[a:b]).copy() for k, v in Pg.items()}
    for v in p_none.values():
        v[i0 + 1:] = 0.0
    C._S.update(pm=C.storm_index({k: dict(p=p_none[k]) for k in C.RAINK}, info), month=month[a:b])
    UN._S.update(Qn=Qn, rain={k: sum(w * p_none[x] for w, x in zip(*chain_weights(k, info))) for k in C.ORDER})
    return Qn, p_none, i0


def check_job(b: int) -> list[dict]:
    S = C._S
    Q, W, frames, info = S["Q"], S["W"], S["frames"], S["info"]
    Qg = {c: Q[c].to_numpy() for c in Q.columns}
    Pg = {k: frames[k]["p"].to_numpy() for k in C.RAINK}
    month = Q.index.month.to_numpy()
    doc = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    U = json.loads((HERE / "data" / "uncert_tables.json").read_text(encoding="utf-8"))
    models = C.from_json(doc["sets"][str(b)])
    ev = pd.read_csv(HERE / "data" / "uncert_warn_events.csv")
    ev = ev[(ev.pruefung == "cv_2004_2022") & (ev.variante == "kette_regen")]
    key = {v: k for k, v in C.DN.items()}
    import routing as R
    floods = {k: C.events_for(k, Q, frames, C.BLOCKS[b], need_rain=True, W=W) for k in C.ORDER}
    stormy = {k: np.array([bool(x) for x in C.is_storm(k, floods[k])]) if len(floods[k]) else np.zeros(0, bool) for k in C.ORDER}
    rain_ev = {}
    for k in C.ORDER:
        if k in C.RAINK:
            an = C.rain_events(k, Q, frames[k], R.find_events(Q, C.DN[k], pre_h=60, post_h=24))
            y = Q.index[an].year
            rain_ev[k] = an[(y >= C.BLOCKS[b][0]) & (y <= C.BLOCKS[b][1])]
    rows = []
    for k in C.ORDER:
        qv, mq, d = U["warn"][k]["q"], U["mq"][k], U["offset"][k]
        tabs = U["vorhersage"][str(b)][k]["q"]
        w, ks = chain_weights(k, info)
        actual = sum(wi * np.nan_to_num(Pg[x]) for wi, x in zip(w, ks))
        issues = []
        for r in ev[ev.pegel == C.DN[k]].itertuples():
            tb = pd.Timestamp(r.beginn)
            if C.BLOCKS[b][0] <= tb.year <= C.BLOCKS[b][1]:
                issues += [("hq10", Q.index.get_loc(tb) - h * 4, h) for h in (24, 12, 6)]
        fl, stf = floods[k], stormy[k]
        issues += [("starkregen-hw", int(p) - 96, 24) for p in fl[stf]] + [("winter-hw", int(p) - 96, 24) for p in fl[~stf][::3]]
        issues += [("regen ohne hw", int(a) - 96, 24) for a in rain_ev.get(k, [])]
        for kind, t, h in issues:
            if t - HIST < 0 or not np.isfinite(Qg[C.DN[k]][t - 96:t + 1]).all():
                continue
            pm_g, mo_g = C._S["pm"], C._S["month"]
            try:
                Qn, p_none, i0 = local_state(t, Qg, Pg, info, month)
                if not all(np.isfinite(Qn[C.DN[x]][i0 - C.A:i0 + 1]).all() for x in C.chain(k)):
                    continue
                feats = UN.features_at(k, np.array([i0]))
                st = int(UN.state_class(feats, k)[0])
                th = thresholds(models, k, Qn, p_none, i0, info, tabs, st, mq, d, qv)
            finally:
                C._S.update(pm=pm_g, month=mo_g)
            ms = max_sums(actual, t)
            qmax = float(np.nanmax(Qg[C.DN[k]][t + 1:t + 1 + C.H]))
            rows.append(dict(block=b, pegel=k, art=kind, zeit=str(Q.index[t]), vor_h=h, q0_mq=float(Qg[C.DN[k]][t] / mq), qmax_rel=qmax / qv, p0=th[DURS_H[0]]["p"][0],
                             **{f"r10_{D}h": th[D]["r10"] for D in DURS_H}, **{f"r50_{D}h": th[D]["r50"] for D in DURS_H},
                             **{f"regen_{D}h": ms[D] for D in DURS_H}))
    print(f"Block {b}: {len(rows)} Ausgaben", flush=True)
    return rows


def check(jobs: int) -> None:
    import multiprocessing as mp
    C.setup_state()
    with mp.get_context("fork").Pool(jobs) as pool:
        rows = [r for part in pool.map(check_job, range(len(C.BLOCKS))) for r in part]
    pd.DataFrame(rows).to_csv(HERE / "data" / "rain_thr_check.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("--jobs", type=int, default=2)
    check(ap.parse_args().jobs)
