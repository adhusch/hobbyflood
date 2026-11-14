"""Starkregen-Prüfung der Kette mit perfektem Regen (Abflussmodell allein, ohne Fehler der Regenvorhersage).

  python extreme_fit.py base [--blocks 0,1,2,3,4]
        gespeicherte Modellsätze (data/chain_models.json, je Block ohne den Block angepasst) auswerten
  python extreme_fit.py try NAME --terms bfa,acc [--wexp 0.5] [--blocks 3,4] [--jobs 2]
        Modellsätze je Block ohne den Block neu anpassen (wie chain_fc.fit_set, mit anderen Erweiterungen des Abflussbeiwerts oder Gewichten)
        und auswerten
  python extreme_fit.py regional NAME [--wexp 0.5] [--blocks 3,4] [--jobs 2]
        Grundmodelle wie gespeichert; dazu in allen sieben Regengebieten der schnelle Abflussweg (lateral.py, "fast") mit drei gemeinsamen
        Parametern (rho, Pc, Kf), angepasst an allen Hochwasser- und Regenfenstern aller Gebiete ohne den Block. Je Pegel gibt es nur ein bis zwei
        Extremhochwasser, an den Oberläufen nur 2021; gemeinsam angepasst lernt etwa Prüm 2 2021 aus Echtershausen 2018 und Seffern 2016.
  python extreme_fit.py compare [NAME ...]
        Ergebnisse aus data/extreme/ nebeneinander

Auswertung je Block, Modelle ohne den Block, mit dem gemessenen Regen auch nach der Ausgabezeit (perfekte Regenvorhersage, Variante kette_ideal):
  hq10    jedes Hochwasser über HQ10 2004-2022 (data/uncert_warn_events.csv), Ausgabe 24, 12 und 6 h vor dem Überschreiten: Scheitel und
          Direktabfluss-Volumen der Vorhersage (48 h) gegen die Messung, Zeitfehler des Scheitels
  floods  alle Hochwasser des Blocks (wie chain_fc.run): NSE bei 6, 12, 24 h Vorlauf und Scheitelverhältnis bei Ausgabe 6, 12, 24 h vor dem Scheitel,
          mit perfektem Regen (kette_ideal) und mit Regen nur bis zur Ausgabezeit (kette_regen)
  dry     Regenlagen ohne Hochwasser (chain_fc.rain_events): Höchstwert der Vorhersage, ausgegeben 24 h vor dem gemessenen Höchstwert, gegen diesen;
          zeigt zu viel Abfluss nach Trockenphasen (Anlass der Gebietsfeuchte im Abflussbeiwert, November 2026)
Die HQ10-Hochwasser liegen alle in den Blöcken 2016-18 und 2019-22; die Auswahl der Modellform sieht sie also (über die Blöcke) mit, das ist bei
drei Lagen nicht zu vermeiden und steht so in der README.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import routing as R

HERE = Path(__file__).parent
OUT = HERE / "data" / "extreme"
BEFORE_H = (24, 12, 6)
LEADS_H = (6, 12, 24)


def _hq10_events() -> list[tuple[str, pd.Timestamp]]:
    ev = pd.read_csv(HERE / "data" / "uncert_warn_events.csv")
    ev = ev[(ev.pruefung == "cv_2004_2022") & (ev.variante == "kette_regen")]
    key = {v: k for k, v in C.DN.items()}
    return [(key[r.pegel], pd.Timestamp(r.beginn)) for r in ev.itertuples()]


def eval_block(models: dict, b: int) -> dict:
    S = C._S
    Q, W, frames = S["Q"], S["W"], S["frames"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    y0, y1 = C.BLOCKS[b]
    out = dict(block=b, hq10=[], floods={}, dry=[])
    # HQ10-Hochwasser
    for k, tb in _hq10_events():
        if not (y0 <= tb.year <= y1):
            continue
        q = Qn[C.DN[k]]
        s = Q.index.get_loc(tb)
        t0 = np.array([s - h * 4 for h in BEFORE_H])
        fc = C.forecast(models, k, Qn, frames, t0, "kette_ideal")[k][:, C.A + 1:]
        for j, h in enumerate(BEFORE_H):
            ob = q[t0[j] + 1:t0[j] + 1 + C.H]
            base = q[t0[j]]
            vo, vf = np.nansum(np.maximum(ob - base, 0)), np.sum(np.maximum(fc[j] - base, 0))
            out["hq10"].append(dict(key=k, beginn=str(tb), vor_h=h, q_obs=float(np.nanmax(ob)), q_fc=float(fc[j].max()),
                                    peak=float(fc[j].max() / np.nanmax(ob)), vol=float(vf / vo) if vo > 0 else None,
                                    dt_h=float((fc[j].argmax() - np.nanargmax(ob)) * C.DT)))
    # alle Hochwasser des Blocks
    for k in C.ORDER:
        ev = C.events_for(k, Q, frames, (y0, y1), need_rain=True, W=W)
        if not len(ev):
            continue
        r = C.evaluate(models, k, ev, ("kette_regen", "kette_ideal"), Qn, frames)
        rec = {}
        for v in ("kette_regen", "kette_ideal", "pers"):
            a = r["acc"][v]
            li = [int(h / C.DT) - 1 for h in LEADS_H]
            pk = r["peaks"][v]                                      # [Ereignisse, Ausgaben vor dem Scheitel, (Verhältnis, Zeit)]
            before = C.OFF[C.OFF <= 0]
            bi = [int(np.where(before == -int(h / C.DT))[0][0]) for h in BEFORE_H]
            rec[v] = dict(sse=[float(a["sse"][i]) for i in li], so=[float(a["so"][i]) for i in li], so2=[float(a["so2"][i]) for i in li], n=int(a["n"]),
                          peak=[[float(x) for x in pk[:, j, 0]] for j in bi])
        out["floods"][k] = rec
    # Regenlagen ohne Hochwasser
    for k in C.ORDER:
        if k not in C.RAINK:
            continue
        flood = R.find_events(Q, C.DN[k], pre_h=60, post_h=24)
        an = C.rain_events(k, Q, frames[k], flood)
        yrs = Q.index[an].year
        an = an[(yrs >= y0) & (yrs <= y1)]
        q = Qn[C.DN[k]]
        for a in an:
            t0 = np.array([a - 96])
            ob = q[a - 96 + 1:a - 96 + 1 + C.H]
            if not np.isfinite(ob).all():
                continue
            fc = C.forecast(models, k, Qn, frames, t0, "kette_ideal")[k][0, C.A + 1:]
            out["dry"].append(dict(key=k, t=str(Q.index[a]), q_obs=float(ob.max()), q_fc=float(fc.max()), bf=float(frames[k]["bf"].to_numpy()[a - 96 - C.A])))
    return out


def _configure(terms: tuple, wexp: float) -> None:
    C.LAT_TERMS = tuple(terms)
    C.WTS_EXP = float(wexp)
    for k, ms in C._S["start"].items():
        if "rain_full" in ms:
            ms["rain_full"].lat = [lt.with_terms(tuple(terms)) for lt in ms["rain_full"].lat]


def job(spec: tuple) -> dict:
    name, terms, wexp, b = spec
    t = time.time()
    if name == "base":
        sets = json.loads(MODELS.read_text(encoding="utf-8"))["sets"]
        models = C.from_json(sets[str(b)])
    else:
        _configure(terms, wexp)
        models = C.fit_set(b)
    res = eval_block(models, b)
    res["models"] = C.to_json(models)
    print(f"{name} Block {C.BLOCKS[b][0]}-{C.BLOCKS[b][1]}: {time.time() - t:.0f} s", flush=True)
    return res


MODES = {   # gemeinsame Parameter: Name -> (Start, untere, obere Grenze)
    "fast": dict(rho=(0.4, 0.0, 1.0), log_pc=(np.log(15.0), np.log(2.0), np.log(60.0)), log_kf=(np.log(1.0), np.log(0.25), np.log(12.0))),
    "sat": dict(rs=(0.5, 0.0, 1.0), log_sev=(np.log(150.0), np.log(10.0), np.log(3000.0)), g_ev=(1.0, 0.0, 10.0),
                log_kf=(np.log(2.0), np.log(0.25), np.log(12.0))),
}
MODES["satlam"] = {**MODES["sat"], "lam": (0.2, 0.02, 0.9)}
MODES["satfast"] = {**MODES["sat"], **{k: v for k, v in MODES["fast"].items() if k != "log_kf"}}


def with_quick(models: dict, th, mode: str, only_storm: bool = False) -> dict:
    """Kopie der Modelle mit dem schnellen Weg (mode: fast, sat, satfast; gemeinsame Parameter th) in jedem Zwischengebiet."""
    names = list(MODES[mode])
    terms = {"fast": ("fast",), "sat": ("sat",), "satlam": ("sat",), "satfast": ("fast", "sat")}[mode]
    out = {}
    for k, ms in models.items():
        out[k] = {}
        for v, m in ms.items():
            m2 = R.Model.from_dict(m.to_dict())
            m2.beta = np.array(m.beta, float)
            if m2.lat and (not only_storm or v.endswith("_storm")):
                lats = []
                for lt in m2.lat:
                    lt2 = lt.with_terms(tuple(t for t in lt.terms if t not in ("int", "fast", "sat", "satx")) + terms)
                    for a, x in zip(names, th):
                        setattr(lt2, a, float(x))
                    lats.append(lt2)
                m2.lat = lats
            out[k][v] = m2
    return out


def _quick_parts(m: R.Model, d: R.Data) -> list[tuple]:
    """Je Zwischengebiet: (Regen, (1 - phi_a) * Regen, api_s, Gebietsfeuchte, Fläche, lead)."""
    import lateral as Lt
    out = []
    for lt in m.lat:
        r = d.P[lt.name]
        phi_a = Lt._slow_coefficient(r, lt)
        out.append((r.rain, (1.0 - phi_a) * r.rain, r.api_s, np.maximum(np.asarray(r.bf, float), 0.0)[:, None], lt.area_km2, r.lead))
    return out


def _quick_q(parts: list[tuple], th, mode: str) -> np.ndarray:
    import lateral as Lt
    P = dict(zip(MODES[mode], th))
    kf = np.exp(P["log_kf"])
    tot = 0.0
    for rain, u, aps, bf, area, lead in parts:
        c = 0.0
        if "rho" in P:
            p2 = rain * rain
            c = c + P["rho"] * p2 / (p2 + np.exp(2 * P["log_pc"]))
        if "rs" in P:
            c = c + P["rs"] * Lt.scs_slope(aps, np.exp(P["log_sev"]) * np.exp(-P["g_ev"] * bf), P.get("lam", 0.2))
        tot = tot + area / 3.6 * Lt._two_stores(u * c, kf, kf, Lt.DT_H)[:, lead:]
    return tot


def regional_job(spec: tuple) -> dict:
    """Gemeinsame Parameter des schnellen Abflussweges für einen Block anpassen (Grundmodelle des Blocks fest) und auswerten."""
    from scipy.optimize import least_squares
    name, wexp, b, only_storm, mode = spec
    t = time.time()
    C.WTS_EXP = float(wexp)
    S = C._S
    Q, W, frames = S["Q"], S["W"], S["frames"]
    sets = json.loads(MODELS.read_text(encoding="utf-8"))["sets"]
    base = C.from_json(sets[str(b)] if b is not None else sets["alle"])
    stack = []
    for k in C.RAINK:
        d = C.data_for(k, Q, W, frames)
        if b is not None:
            yrs = pd.DatetimeIndex(d.times).year.to_numpy()
            test = (yrs >= C.BLOCKS[b][0]) & (yrs <= C.BLOCKS[b][1])
            near = (np.abs(d.peaks[:, None] - d.peaks[test][None, :]) <= int(3.5 * 24 / C.DT)).any(axis=1)
            d = d.subset(~test & ~near)
        mt = C.rain_type(k)
        if only_storm:
            d = d.subset(C.is_storm(k, d.peaks))
            mt = mt + "_storm"
        m = base[k][mt]
        stack.append((d, m.route(d.X, d.P), _quick_parts(m, d)))
    spec_ = MODES[mode]
    lo = np.array([v[1] for v in spec_.values()], float)
    hi = np.array([v[2] for v in spec_.values()], float)
    st0 = np.array([v[0] for v in spec_.values()], float)

    def resid(th):
        return np.concatenate([((d.Y - yb - _quick_q(fp, th, mode)) / d.wts[:, None])[d.mask] for d, yb, fp in stack])
    best = None
    rng = np.random.default_rng(1)
    for j in range(4):
        th0 = st0 if j == 0 else lo + (hi - lo) * rng.uniform(0.15, 0.85, len(lo))
        r = least_squares(resid, np.clip(th0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=80)
        if best is None or r.cost < best.cost:
            best = r
    th = best.x
    cost0 = 0.5 * float(np.sum(np.concatenate([((d.Y - yb) / d.wts[:, None])[d.mask] for d, yb, fp in stack]) ** 2))
    models = with_quick(base, th, mode, only_storm)
    for k in C.RAINK:                                       # Fehlerkorrektur neu (wie fit_set)
        d = C.data_for(k, Q, W, frames)
        if b is not None:
            yrs = pd.DatetimeIndex(d.times).year.to_numpy()
            test = (yrs >= C.BLOCKS[b][0]) & (yrs <= C.BLOCKS[b][1])
            near = (np.abs(d.peaks[:, None] - d.peaks[test][None, :]) <= int(3.5 * 24 / C.DT)).any(axis=1)
            d = d.subset(~test & ~near)
        mt = C.rain_type(k)
        st = C.is_storm(k, d.peaks)
        models[k][mt].beta = C.error_beta(models[k][mt], d)
        for reg, sel in (("winter", ~st), ("storm", st)):
            v = f"{mt}_{reg}"
            models[k][v].beta = C.error_beta(models[k][v], d.subset(sel)) if sel.sum() >= 4 else models[k][mt].beta
    res = eval_block(models, b) if b is not None else dict(block=None)
    res["models"] = C.to_json(models)
    res["shared"] = dict(mode=mode, **{n: float(x) for n, x in zip(spec_, th)}, cost=float(best.cost), cost_ohne=cost0)
    shown = ", ".join(f"{n} {np.exp(x) if n.startswith('log_') else x:.3g}" for n, x in zip(spec_, th))
    print(f"{name} Block {b}: {shown}; Kosten {best.cost:.1f} (ohne {cost0:.1f}), {time.time() - t:.0f} s", flush=True)
    return res


def run_regional(name: str, wexp: float, blocks: list[int], jobs: int, only_storm: bool = False, mode: str = "fast") -> None:
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    C.setup_state()
    specs = [(name, wexp, b, only_storm, mode) for b in blocks]
    with mp.get_context("fork").Pool(min(jobs, len(specs))) as pool:
        res = sorted(pool.map(regional_job, specs), key=lambda r: r["block"])
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, terms=["bfa", f"{mode} (gemeinsam)"], wexp=wexp, blocks=res), ensure_ascii=False),
                                      encoding="utf-8")
    print(summary(name, blocks))


SAT_KEYS = ("rs", "log_sev", "g_ev", "log_kf")


def refit_job(spec: tuple) -> dict:
    """Grundmodelle des Blocks neu anpassen mit dem Sättigungsüberschuss als festem Baustein (Parameter aus einer gemeinsamen Anpassung)."""
    import rain_fit as RF2
    name, src, b = spec
    t = time.time()
    sh = next(r for r in json.loads((OUT / f"{src}.json").read_text(encoding="utf-8"))["blocks"] if r["block"] == b)["shared"]
    vals = {k: sh[k] for k in SAT_KEYS}
    C.LAT_TERMS = ("bfa", "satx")
    orig = RF2.lat_start

    def lat_start(info, key, kind=0):
        lt = orig(info, key, kind)
        for a, x in vals.items():
            setattr(lt, a, float(x))
        return lt
    RF2.lat_start = lat_start
    for k, ms in C._S["start"].items():
        if "rain_full" in ms:
            lats = []
            for lt in ms["rain_full"].lat:
                lt2 = lt.with_terms(("bfa", "satx"))
                for a, x in vals.items():
                    setattr(lt2, a, float(x))
                lats.append(lt2)
            ms["rain_full"].lat = lats
    models = C.fit_set(b)
    res = eval_block(models, b)
    res["models"] = C.to_json(models)
    res["shared"] = dict(sh)
    print(f"{name} Block {b}: {time.time() - t:.0f} s", flush=True)
    return res


def run_refit(name: str, src: str, blocks: list[int], jobs: int) -> None:
    import multiprocessing as mp
    C.setup_state()
    with mp.get_context("fork").Pool(min(jobs, len(blocks))) as pool:
        res = sorted(pool.map(refit_job, [(name, src, b) for b in blocks]), key=lambda r: r["block"])
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, terms=["bfa", f"satx aus {src}"], wexp=0.0, blocks=res), ensure_ascii=False),
                                      encoding="utf-8")
    print(summary(name, blocks))


import os
MODELS = Path(os.environ.get("PRUEM_MODELS", str(HERE / "data" / "chain_models.json")))    # Grundmodelle (z. B. data/stand_station/chain_models.json)

FULL_SPEC = {   # Sättigungsüberschuss + nichtlineare Speicher, gemeinsam: Name -> (Start, untere, obere Grenze)
    "rs": (0.97, 0.0, 1.0), "log_sev": (float(np.log(170.0)), float(np.log(10.0)), float(np.log(3000.0))), "g_ev": (0.2, 0.0, 10.0),
    "log_kf": (float(np.log(2.5)), float(np.log(0.25)), float(np.log(12.0))), "log_oref": (float(np.log(0.5)), float(np.log(0.02)), float(np.log(50.0))),
    "a_nl": (0.5, 0.0, 2.0)}


def _lat_with(lt, th: dict, terms=("satx", "nlrx")):
    lt2 = lt.with_terms(tuple(t for t in lt.terms if t not in ("int", "fast", "sat", "satx", "nlr", "nlrx")) + tuple(terms))
    for a, x in th.items():
        setattr(lt2, a, float(x))
    return lt2


def full_job(spec: tuple) -> dict:
    """Sättigungsüberschuss und nichtlineare Speicher gemeinsam für alle Gebiete (6 Parameter), Grundmodelle des Blocks fest; ganze
    Zwischengebietsantwort je Auswertung neu gerechnet."""
    from scipy.optimize import least_squares
    import lateral as Lt
    name, b, keys, nstart = spec
    t = time.time()
    S = C._S
    Q, W, frames = S["Q"], S["W"], S["frames"]
    sets = json.loads(MODELS.read_text(encoding="utf-8"))["sets"]
    base = C.from_json(sets[str(b)] if b is not None else sets["alle"])
    stack = []
    for k in C.RAINK:
        d = C._training(k, b)
        m = base[k][C.rain_type(k)]
        rest = m.route(d.X, d.P) - sum(Lt.response(d.P[lt.name], lt) for lt in m.lat)
        stack.append((d, rest, m.lat))
    names = list(keys)
    lo = np.array([FULL_SPEC[n][1] for n in names]); hi = np.array([FULL_SPEC[n][2] for n in names])
    fixed = {n: v[0] for n, v in FULL_SPEC.items() if n not in names}
    terms = ("satx",) + (("nlrx",) if ("a_nl" in names or fixed.get("a_nl", 0) > 0) else ())

    def resid(th):
        P = {**fixed, **dict(zip(names, th))}
        out = []
        for d, rest, lats in stack:
            y = rest + sum(Lt.response(d.P[lt.name], _lat_with(lt, P, terms)) for lt in lats)
            out.append(((d.Y - y) / d.wts[:, None])[d.mask])
        return np.concatenate(out)
    best, rng = None, np.random.default_rng(2)
    for j in range(nstart):
        th0 = np.array([FULL_SPEC[n][0] for n in names]) if j == 0 else lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
        r = least_squares(resid, np.clip(th0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=60)
        if best is None or r.cost < best.cost:
            best = r
        print(f"  {name} Block {b} Start {j}: Kosten {r.cost:.1f}, {time.time() - t:.0f} s", flush=True)
    P = {**fixed, **dict(zip(names, best.x))}
    models = {}
    for k, ms in base.items():
        models[k] = {}
        for v, m in ms.items():
            m2 = R.Model.from_dict(m.to_dict()); m2.beta = np.array(m.beta, float)
            if m2.lat:
                m2.lat = [_lat_with(lt, P, terms) for lt in m2.lat]
            models[k][v] = m2
    for k in C.RAINK:
        d = C._training(k, b)
        mt = C.rain_type(k)
        st = C.is_storm(k, d.peaks)
        models[k][mt].beta = C.error_beta(models[k][mt], d)
        for reg, sel in (("winter", ~st), ("storm", st)):
            models[k][f"{mt}_{reg}"].beta = C.error_beta(models[k][f"{mt}_{reg}"], d.subset(sel)) if sel.sum() >= 4 else models[k][mt].beta
    res = eval_block(models, b) if b is not None else dict(block=None)
    res["models"] = C.to_json(models)
    res["shared"] = dict(**{n: float(x) for n, x in P.items()}, cost=float(best.cost))
    shown = ", ".join(f"{n} {np.exp(x) if n.startswith('log_') else x:.3g}" for n, x in P.items())
    print(f"{name} Block {b}: {shown}; Kosten {best.cost:.1f}, {time.time() - t:.0f} s", flush=True)
    return res


def run_full(name: str, keys: list[str], blocks: list[int], jobs: int, nstart: int) -> None:
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    C.setup_state()
    with mp.get_context("fork").Pool(min(jobs, len(blocks))) as pool:
        res = sorted(pool.map(full_job, [(name, b, tuple(keys), nstart) for b in blocks]), key=lambda r: r["block"])
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, terms=["bfa", "satx+nlrx gemeinsam: " + ",".join(keys)], wexp=0.0, blocks=res),
                                                  ensure_ascii=False), encoding="utf-8")
    print(summary(name, blocks))


def run(name: str, terms: tuple, wexp: float, blocks: list[int], jobs: int) -> None:
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    C.setup_state()
    specs = [(name, terms, wexp, b) for b in blocks]
    with mp.get_context("fork").Pool(min(jobs, len(specs))) as pool:
        res = sorted(pool.map(job, specs), key=lambda r: r["block"])
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, terms=list(terms), wexp=wexp, blocks=res), ensure_ascii=False), encoding="utf-8")
    print(summary(name, blocks))


def summary(name: str, blocks: list[int] | None = None) -> str:
    d = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    bl = [r for r in d["blocks"] if blocks is None or r["block"] in blocks]
    lines = [f"== {name}: Erweiterungen {d['terms']}, Gewichtsexponent {d['wexp']}, Blöcke {[r['block'] for r in bl]}"]
    h = pd.DataFrame([x for r in bl for x in r["hq10"]])
    if len(h):
        g = h.groupby("vor_h").agg(scheitel_med=("peak", "median"), scheitel_min=("peak", "min"), scheitel_max=("peak", "max"), volumen_med=("vol", "median"),
                                   zeit_med_h=("dt_h", "median"), n=("peak", "size")).sort_index(ascending=False)
        lines.append("HQ10-Hochwasser, perfekter Regen (Vorhersage/Messung):\n" + g.round(2).to_string())
    rows = []
    for k in C.ORDER:
        recs = [r["floods"][k] for r in bl if k in r["floods"]]
        if not recs:
            continue
        for v in ("kette_ideal", "kette_regen"):
            sse = np.sum([x[v]["sse"] for x in recs], axis=0)
            so = np.sum([x[v]["so"] for x in recs], axis=0)
            so2 = np.sum([x[v]["so2"] for x in recs], axis=0)
            n = sum(x[v]["n"] for x in recs)
            nse = 1 - sse / (so2 - so ** 2 / n)
            pk = [np.concatenate([x[v]["peak"][j] for x in recs]) for j in range(len(BEFORE_H))]
            rows.append(dict(pegel=k, variante=v, **{f"nse_{L}h": round(float(z), 3) for L, z in zip(LEADS_H, nse)},
                             **{f"scheitel_{h}h": round(float(np.median(p)), 2) for h, p in zip(BEFORE_H, pk)}, hw=len(pk[0])))
    if rows:
        lines.append("Alle Hochwasser (NSE je Vorlauf; Scheitelverhältnis, Median, Ausgabe x h vor dem Scheitel):\n" + pd.DataFrame(rows).to_string(index=False))
    dr = pd.DataFrame([x for r in bl for x in r["dry"]])
    if len(dr):
        dr["ratio"] = (dr.q_fc + 0.05) / (dr.q_obs + 0.05)
        g = dr.groupby("key").agg(median=("ratio", "median"), ueber_1_5=("ratio", lambda x: float(np.mean(x > 1.5))), ueber_2=("ratio", lambda x: float(np.mean(x > 2))),
                                  n=("ratio", "size"))
        lines.append("Regenlagen ohne Hochwasser (Höchstwert Vorhersage/Messung, 24 h vorher ausgegeben):\n" + g.round(2).to_string())
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "try", "regional", "refit", "full", "compare"])
    ap.add_argument("--keys", default="rs,log_sev,g_ev,log_kf,log_oref,a_nl")
    ap.add_argument("--nstart", type=int, default=2)
    ap.add_argument("--src", default="sat0")
    ap.add_argument("--only-storm", action="store_true")
    ap.add_argument("--mode", default="sat", choices=["fast", "sat", "satlam", "satfast"])
    ap.add_argument("--tau-s", type=float, default=None, help="Abklingzeit des Ereignisregens api_s in h (Standard rain.TAU_API_S_H)")
    ap.add_argument("name", nargs="*")
    ap.add_argument("--terms", default="bfa")
    ap.add_argument("--wexp", type=float, default=0.0)
    ap.add_argument("--blocks", default="3,4")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    bl = [int(x) for x in a.blocks.split(",")]
    if a.tau_s:
        import rain as RN
        RN.TAU_API_S_H = float(a.tau_s)
    if a.cmd == "base":
        run("base", tuple(C.LAT_TERMS), 0.0, bl, a.jobs)
    elif a.cmd == "try":
        run(a.name[0], tuple(t for t in a.terms.split(",") if t), a.wexp, bl, a.jobs)
    elif a.cmd == "full":
        run_full(a.name[0], a.keys.split(","), bl, a.jobs, a.nstart)
    elif a.cmd == "refit":
        run_refit(a.name[0], a.src, bl, a.jobs)
    elif a.cmd == "regional":
        run_regional(a.name[0], a.wexp, bl, a.jobs, a.only_storm, a.mode)
    else:
        for n in (a.name or sorted(p.stem for p in OUT.glob("*.json"))):
            print(summary(n))
