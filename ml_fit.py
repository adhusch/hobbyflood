"""Lernender Abflussbeiwert ("ml" in lateral.py): ein kleines Netz für alle sieben Regengebiete, angepasst an allen Hochwasser- und Regenfenstern.

  python ml_fit.py run NAME [--blocks 0,1,2,3,4] [--seeds 3] [--steps 600] [--wd 1e-4] [--wexp 0] [--hidden 16]
        je Block: Grundmodelle des Blocks (chain_models.json, ohne den Block angepasst) fest; das Netz verschiebt den langsamen und den schnellen
        Anteil des Regens (lateral.ml_fractions). Angepasst mit denselben Fenstern und demselben Fehlermaß wie chain_fc.fit_regional_sat (Fehler je
        Fenster durch den Scheitel geteilt), dazu die drei gemeinsamen Speichergrößen (Kf, oref, a). Auswertung wie extreme_fit.py (eval_block).
  python ml_fit.py final NAME    Netz mit allen Jahren (Satz "alle") anpassen und alle Sätze in data/chain_models.json eintragen

Schutz gegen Überanpassung: ein Zeitblock der Trainingsjahre dient nur zum Abbruch (bester Stand auf diesem Block), Gewichte werden gedämpft
(weight decay), mehrere Startwerte werden gemittelt, und das Netz beginnt beim bisherigen Modell (letzte Schicht null).
Die Rechnung läuft mit JAX (automatische Ableitung durch die Speicherkaskade); lateral.py rechnet dasselbe Netz in numpy (Prüfung in test_ml.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import optax

import chain_fc as C
import extreme_fit as E
import lateral as L
import routing as R

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
OUT = HERE / "data" / "extreme"
DT = C.DT


# ------------------------------------------------------------------ Modell in JAX
def nl_stores(peff, k1, k2, oref, a):
    """Wie lateral._nl_stores, peff [T, E]."""
    s1, s2 = k1 * peff[0], k2 * peff[0]

    def step(c, p):
        s1, s2 = c
        kt1 = k1 / jnp.power(1.0 + s1 / (k1 * oref), a)
        e1 = jnp.exp(-DT / kt1)
        s1n = s1 * e1 + p * kt1 * (1.0 - e1)
        i2 = p - (s1n - s1) / DT
        kt2 = k2 / jnp.power(1.0 + s2 / (k2 * oref), a)
        e2 = jnp.exp(-DT / kt2)
        s2n = s2 * e2 + i2 * kt2 * (1.0 - e2)
        return (s1n, s2n), i2 - (s2n - s2) / DT
    _, out = jax.lax.scan(step, (s1, s2), peff[1:])
    return jnp.concatenate([peff[:1], out], axis=0)


def mlp(net, x):
    h = jnp.tanh(x @ net["W1"] + net["b1"])
    h = jnp.tanh(h @ net["W2"] + net["b2"])
    return h @ net["W3"] + net["b3"]


def logit(x):
    x = jnp.clip(x, 1e-4, 1.0 - 1e-4)
    return jnp.log(x / (1.0 - x))


def lat_flow(th, a):
    """Zwischengebietsabfluss [E, T] in m³/s für ein Gebiet a (dict mit festen Größen)."""
    d = mlp(th["net"], a["X"])
    s = jax.nn.sigmoid(logit(a["phi"]) + d[..., 0])
    q = (1.0 - s) * jax.nn.sigmoid(logit(a["q0"]) + d[..., 1])
    kf, oref, anl = jnp.exp(th["log_kf"]), jnp.exp(th["log_oref"]), th["a_nl"]
    rain = a["rain"]
    o = nl_stores((s * rain).T, a["k1"], a["k2"], oref, anl) + nl_stores((q * rain).T, kf, kf, oref, anl)
    return a["area"] / 3.6 * o.T[:, a["lead"]:]


def sse(th, areas):
    num, den = 0.0, 0.0
    for a in areas:
        y = a["rest"] + sum(lat_flow(th, p) for p in a["lats"])
        r = (a["Y"] - y) / a["wts"][:, None] * a["w"][:, None]
        num = num + jnp.sum(jnp.where(a["mask"], r * r, 0.0))
        den = den + jnp.sum(a["mask"] * a["w"][:, None] ** 2)
    return num / den


def loss(th, areas, wd):
    n = th["net"]
    return sse(th, areas) + wd * (jnp.sum(n["W1"] ** 2) + jnp.sum(n["W2"] ** 2) + jnp.sum(n["W3"] ** 2))


def clip_globals(th):
    return dict(th, log_kf=jnp.clip(th["log_kf"], np.log(0.25), np.log(12.0)), log_oref=jnp.clip(th["log_oref"], np.log(0.02), np.log(50.0)),
                a_nl=jnp.clip(th["a_nl"], 0.0, 2.0))


# ------------------------------------------------------------------ Daten
NORAD = False     # Gegenprobe: Radargrößen durch die Stationswerte ersetzen (wie im Betrieb ohne Radar)


def no_radar(r: L.Rain) -> L.Rain:
    import rain_fit as RF2
    ex = np.stack([r.rain] + [np.maximum(r.rain - f, 0.0) for f in RF2.EXCESS_F], axis=-1).astype("float32")
    return L.Rain(r.rain, r.api, r.lead, r.api_s, r.bf, ex)


def pack(d: R.Data, m: R.Model, wexp: float) -> dict:
    lats = []
    for lt in m.lat:
        r = no_radar(d.P[lt.name]) if NORAD else d.P[lt.name]
        phi = L._slow_coefficient(r, lt)
        quick = L._quick_part(r, lt, phi)
        X = L.ml_features(r, phi, quick)
        lats.append(dict(X=jnp.asarray(X), phi=jnp.asarray(phi), q0=jnp.asarray(quick / np.maximum(1.0 - phi, 1e-6)), rain=jnp.asarray(r.rain),
                         k1=float(np.exp(lt.log_k1)), k2=float(np.exp(lt.log_k2)), area=float(lt.area_km2), lead=int(r.lead)))
    rest = m.route(d.X, d.P) - sum(L.response(d.P[lt.name], lt) for lt in m.lat)
    w = (d.wts / np.median(d.wts)) ** wexp if wexp else np.ones(len(d.wts))
    return dict(Y=jnp.asarray(d.Y), rest=jnp.asarray(rest), mask=jnp.asarray(d.mask), wts=jnp.asarray(d.wts), w=jnp.asarray(w), lats=lats)


def val_block(b: int | None) -> int:
    """Block nur zum Abbruch: einer ohne HQ10-Hochwasser (die liegen in 2016-18 und 2019-22 und sollen ins Training)."""
    return 2 if b == 1 else 1


def datasets(b: int | None, base: dict, wexp: float) -> tuple[list, list]:
    vb = val_block(b)
    tr, va = [], []
    for k in C.RAINK:
        d = C._training(k, b)
        m = base[k][C.rain_type(k)]
        yrs = pd.DatetimeIndex(d.times).year.to_numpy()
        isv = (yrs >= C.BLOCKS[vb][0]) & (yrs <= C.BLOCKS[vb][1])
        tr.append(pack(d.subset(~isv), m, wexp))
        va.append(pack(d.subset(isv), m, 0.0))
    return tr, va


def init(seed: int, hidden: int, P: dict) -> dict:
    rng = np.random.default_rng(seed)
    nf = L.ML_NF
    net = dict(W1=rng.normal(0, 1 / np.sqrt(nf), (nf, hidden)), b1=np.zeros(hidden), W2=rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)),
               b2=np.zeros(hidden), W3=np.zeros((hidden, 2)), b3=np.zeros(2))
    return dict(net={k: jnp.asarray(v) for k, v in net.items()}, log_kf=jnp.asarray(P["log_kf"]), log_oref=jnp.asarray(P["log_oref"]),
                a_nl=jnp.asarray(P["a_nl"]))


def train(tr: list, va: list, P: dict, seeds: int, steps: int, wd: float, hidden: int, lr: float, tag: str) -> tuple[list, dict, list]:
    """Rückgabe: Netze (je Startwert der beste Stand auf dem Abbruch-Block), gemittelte Speichergrößen, Protokoll."""
    grad = jax.jit(jax.value_and_grad(lambda th: loss(th, tr, wd)))       # Daten als Konstanten (lead muss fest sein)
    val_tr = jax.jit(lambda th: sse(th, tr))
    val = jax.jit(lambda th: sse(th, va))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.1))
    nets, glob, log = [], [], []
    for sd in range(seeds):
        th = init(sd, hidden, P)
        st = opt.init(th)
        v0 = float(val(th)); t0 = float(val_tr(th))
        best = (v0, th, 0)
        t = time.time()
        for i in range(1, steps + 1):
            l, g = grad(th)
            up, st = opt.update(g, st)
            th = clip_globals(optax.apply_updates(th, up))
            if i % 10 == 0:
                v = float(val(th))
                if v < best[0]:
                    best = (v, th, i)
        tb = float(val_tr(best[1]))
        log.append(dict(seed=sd, train0=t0, val0=v0, train=tb, val=best[0], step=best[2], sec=round(time.time() - t)))
        print(f"  {tag} Start {sd}: Training {t0:.4f} -> {tb:.4f}, Abbruch-Block {v0:.4f} -> {best[0]:.4f} (Schritt {best[2]}), {time.time() - t:.0f} s", flush=True)
        nets.append({k: np.asarray(v).round(5).tolist() for k, v in best[1]["net"].items()})
        glob.append([float(best[1]["log_kf"]), float(best[1]["log_oref"]), float(best[1]["a_nl"])])
    g = np.mean(glob, axis=0)
    return nets, dict(log_kf=float(g[0]), log_oref=float(g[1]), a_nl=float(g[2])), log


def apply(base: dict, nets: list, G: dict, b: int | None) -> dict:
    """Modelle mit Netz in jedem Zwischengebiet, Fehlerkorrektur neu (wie extreme_fit.full_job)."""
    models = {}
    for k, ms in base.items():
        models[k] = {}
        for v, m in ms.items():
            m2 = R.Model.from_dict(m.to_dict()); m2.beta = np.array(m.beta, float)
            if m2.lat:
                out = []
                for lt in m2.lat:
                    lt2 = lt.with_terms(tuple(t for t in lt.terms if t != "ml") + ("ml",))
                    lt2.net = nets
                    lt2.log_kf, lt2.log_oref, lt2.a_nl = G["log_kf"], G["log_oref"], G["a_nl"]
                    out.append(lt2)
                m2.lat = out
            models[k][v] = m2
    for k in C.RAINK:
        d = C._training(k, b)
        mt = C.rain_type(k)
        st = C.is_storm(k, d.peaks)
        models[k][mt].beta = C.error_beta(models[k][mt], d)
        for reg, sel in (("winter", ~st), ("storm", st)):
            models[k][f"{mt}_{reg}"].beta = C.error_beta(models[k][f"{mt}_{reg}"], d.subset(sel)) if sel.sum() >= 4 else models[k][mt].beta
    return models


def fit_block(b: int | None, a) -> tuple[dict, dict]:
    doc = json.loads(E.MODELS.read_text(encoding="utf-8"))
    key = str(b) if b is not None else "alle"
    base = C.from_json(doc["sets"][key])
    P = doc["sat"][key]
    tr, va = datasets(b, base, a.wexp)
    nets, G, log = train(tr, va, P, a.seeds, a.steps, a.wd, a.hidden, a.lr, f"{a.name[0]} Block {key}")
    models = apply(base, nets, G, b)
    return models, dict(nets=nets, shared=G, log=log)


def run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C.setup_state()
    res = []
    for b in [int(x) for x in a.blocks.split(",")]:
        t = time.time()
        models, info = fit_block(b, a)
        r = E.eval_block(models, b)
        r["models"] = C.to_json(models)
        r["ml"] = dict(shared=info["shared"], log=info["log"])
        res.append(r)
        print(f"{a.name[0]} Block {b}: Kf {np.exp(info['shared']['log_kf']):.2f} h, oref {np.exp(info['shared']['log_oref']):.2f}, a {info['shared']['a_nl']:.2f}; {time.time() - t:.0f} s",
              flush=True)
    name = a.name[0]
    (OUT / f"{name}.json").write_text(json.dumps(dict(name=name, terms=["bfa", "satx", "nlrx", f"ml: {a.seeds} Netze, {a.hidden} Knoten, wd {a.wd}, wexp {a.wexp}"],
                                                      wexp=a.wexp, blocks=res), ensure_ascii=False), encoding="utf-8")
    print(E.summary(name, [r["block"] for r in res]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("name", nargs=1)
    ap.add_argument("--blocks", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--wexp", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--norad", action="store_true", help="Gegenprobe ohne Radar: Netz nur mit Stationsregen")
    a = ap.parse_args()
    if a.norad:
        NORAD = True
        import rain_fit as RF2
        import os
        RF2.radar_hourly = lambda: None            # Fenster ohne Radarspalten ...
        _rw = C.rain_window
        C.rain_window = lambda frame, t0, ideal: no_radar(_rw(frame, t0, ideal))   # ... und Vorhersagefenster mit Stationsgrößen
        _tr = C._training

        def _training_norad(key, block):
            d = _tr(key, block)
            d.P = {k: no_radar(v) for k, v in d.P.items()}
            return d
        C._training = _training_norad
    run(a)
