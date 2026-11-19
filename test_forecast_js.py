"""Prüft die Portierung der Kettenvorhersage nach JavaScript (forecast_core.js, läuft in der Karte) gegen chain_fc.forecast (python test_forecast_js.py).
Beide rechnen auf denselben Eingaben (Hochwasserfenster der Karte aus data/chain_floods.json, Modelle aus data/chain_models.json); verglichen werden
alle Pegel der Kette für mehrere Hochwasser, Ausgabezeiten und Varianten. Braucht node."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import chain_fc as C
import rain as RN

HERE = Path(__file__).parent


def py_forecast(fl: dict, models: dict, target: str, i: int, variant: str) -> dict:
    """chain_fc.forecast auf den Fenstern eines Hochwassers: Abflussreihen vorn um LEAD aufgefüllt, damit Abfluss- und Regenindex übereinstimmen."""
    pad = np.full(C.LEAD, np.nan)
    Qn = {C.DN[k]: np.concatenate([pad, np.array([np.nan if v is None else v for v in fl["q"][C.DN[k]]], float)]) for k in C.ORDER}
    frames = {}
    if fl["rain"]:
        for k, p in fl["p"].items():
            p = np.array(p, float)
            frames[k] = dict(p=p, api=RN.api_series(p, C.DT, RN.TAU_API_H) + fl["api0"][k][0] * np.exp(-np.arange(1, len(p) + 1) * C.DT / RN.TAU_API_H),
                             api_s=RN.api_series(p, C.DT, RN.TAU_API_S_H) + fl["api0"][k][1] * np.exp(-np.arange(1, len(p) + 1) * C.DT / RN.TAU_API_S_H))
    if not C_CONSTS:
        C_CONSTS.update(json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["consts"])
    if fl["rain"] and "WET" in C_CONSTS:
        for k, (ref, mq) in C_CONSTS["WET"].items():                  # Gebietsfeuchte wie in der Karte: Abfluss des Bezugspegels im Fenster / MQ
            if k in frames:
                frames[k]["bf"] = Qn[ref] / mq
    if fl["rain"] and fl.get("sm"):                                    # Bodenfeuchte wie die Abflussreihen vorn um LEAD aufgefüllt
        for k, v in fl["sm"].items():
            if k in frames:
                frames[k]["sm"] = np.concatenate([pad, np.array(v, float)])
                if "BFMIN" in C_CONSTS and k in C_CONSTS["BFMIN"] and "bf" in frames[k]:     # Untergrenze der Gebietsfeuchte wie chain_fc.add_wetness
                    frames[k]["bf"] = np.maximum(frames[k]["bf"], np.interp(np.where(np.isfinite(frames[k]["sm"]), frames[k]["sm"], C.SM_FALLBACK), *C_CONSTS["BFMIN"][k]))
    storm = np.array([fl.get("storm", {}).get(target, "0" * (i + 1))[i] == "1"])
    S = C.forecast(models, target, Qn, frames, np.array([i + C.LEAD]), variant, storm=storm)
    return {k: v[0] for k, v in S.items()}


C_CONSTS: dict = {}


def check_lateral_terms() -> None:
    """Zwischengebiet allein, alle Erweiterungen des Abflussbeiwerts (auch fast, int, acc), zufälliger Regen mit Starkregenspitzen."""
    import lateral as L
    rng = np.random.default_rng(3)
    T, lead = 700, 288
    p = np.where(rng.random(T) < 0.15, rng.gamma(0.6, 6.0, T), 0.0)
    p[400:412] += 35.0                                                  # Gewitter: 35 mm/h über 3 h
    api = np.cumsum(p) * 0.05 + 20.0
    aps = np.convolve(p, np.exp(-np.arange(96) / 96.0), mode="full")[:T] * 0.25
    cases = []
    for terms in [(), ("bfa",), ("int",), ("acc",), ("bfa", "acc"), ("bfa", "fast"), ("bfa", "acc", "fast"), ("fast",), ("bfa", "sat"),
                  ("bfa", "fast", "sat"), ("bfa", "nlr"), ("bfa", "satx", "nlrx"), ("bfa", "sma"), ("bfa", "sma", "satx", "nlrx"), ("bfa", "smf"), ("bfa", "smf", "satx", "nlrx"), ("bf", "smf", "int")]:
        lt = L.Lateral("X", 87.0, 0.6, float(np.log(150.0)), float(np.log(2.5)), float(np.log(14.0)), tuple(terms), 0.45, float(np.log(12.0)), 2.0,
                       w_bf=120.0, log_kf=float(np.log(0.8)), rs=0.7, log_sev=float(np.log(60.0)), g_ev=1.3, lam=0.35,
                       log_oref=float(np.log(0.4)), a_nl=0.8, w_sm=85.0, log_s50=float(np.log(0.93)), log_gs=float(np.log(14.0)))
        py = L.response(L.Rain(p[None, :], api[None, :], lead, aps[None, :], np.array([0.7]), None, np.array([118.0])), lt)[0]
        cases.append(dict(lt=lt.to_dict(), py=py.tolist()))
    nets = [dict(W1=rng.normal(0, 0.4, (L.ML_NF, 16)).tolist(), b1=rng.normal(0, 0.2, 16).tolist(), W2=rng.normal(0, 0.3, (16, 16)).tolist(),
                 b2=rng.normal(0, 0.2, 16).tolist(), W3=rng.normal(0, 0.6, (16, 2)).tolist(), b3=[0.4, -0.3]) for _ in range(2)]
    for net in (nets[0], nets):                                         # lernender Abflussbeiwert: ein Netz und Mittel aus zwei Netzen
        lt = L.Lateral("X", 87.0, 0.6, float(np.log(150.0)), float(np.log(2.5)), float(np.log(14.0)), ("bfa", "satx", "nlrx", "ml"), 0.45, float(np.log(12.0)),
                       2.0, w_bf=120.0, log_kf=float(np.log(0.8)), rs=0.7, log_sev=float(np.log(60.0)), g_ev=1.3, lam=0.35,
                       log_oref=float(np.log(0.4)), a_nl=0.8, net=net)
        py = L.response(L.Rain(p[None, :], api[None, :], lead, aps[None, :], np.array([0.7])), lt)[0]
        cases.append(dict(lt=lt.to_dict(), py=py.tolist()))
    js = r"""
const M = require(process.argv[1]), cases = JSON.parse(require("fs").readFileSync(0)), p = JSON.parse(process.argv[2]);
const r = { p: Float64Array.from(p.p), api: Float64Array.from(p.api), aps: Float64Array.from(p.aps), lead: p.lead, bf: 0.7, sm: 118.0 };
process.stdout.write(JSON.stringify(cases.map(c => Array.from(M.lateral(r, c.lt)))));
"""
    arg = json.dumps(dict(p=p.tolist(), api=api.tolist(), aps=aps.tolist(), lead=lead))
    res = json.loads(subprocess.run(["node", "-e", js, str(HERE / "forecast_core.js"), arg], input=json.dumps(cases), capture_output=True, text=True,
                                    check=True).stdout)
    worst = max(float(np.max(np.abs(np.array(j) - np.array(c["py"])) / np.maximum(1.0, np.abs(c["py"])))) for c, j in zip(cases, res))
    print(f"Zwischengebiet, {len(cases)} Kombinationen der Erweiterungen: größte rel. Abweichung JS-Python {worst:.1e}")
    assert worst < 1e-9


def main() -> None:
    check_lateral_terms()
    mj = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    C_CONSTS.update(mj["consts"])
    floods = json.loads((HERE / "data" / "chain_floods.json").read_text(encoding="utf-8"))
    cases = []
    for fl in floods:
        for target in ("PZ", "AL", "WI", "P2"):
            st = fl.get("storm", {}).get(target, "")
            extra = [st.find("1"), st.find("1") + 3] if "1" in st else []           # Ausgabezeiten mit Starkregen-Einteilung, falls vorhanden
            for i in [fl["i_peak"] - 48 * 4, fl["i_peak"] - 13 * 4 - 1, fl["i_peak"], fl["i_peak"] + 12 * 4] + [x for x in extra if fl["i_peak"] - 48 * 4 <= x <= fl["i_peak"] + 12 * 4]:
                for v in C.VARIANTS + ("kette_vorhersage",):
                    if v not in C.NORAIN and not fl["rain"]:
                        continue
                    cases.append(dict(f=fl["id"], target=target, i=i, v=v))
                    if fl is floods[-1] and target in ("PZ", "AL"):                 # dasselbe mit Niveaufaktor (niveau.py)
                        cases.append(dict(f=fl["id"], target=target, i=i, v=v, nv={"PZ": 1.16, "AL": 0.76}))
                        cases.append(dict(f=fl["id"], target=target, i=i, v=v, cf={"PZ": {"WI": 0.7, "SI": 1.3}, "AL": {"SE": 1.6}}))
    js = r"""
const fs = require("fs"), FC = require(process.argv[1]);
const M = JSON.parse(fs.readFileSync(process.argv[2])), FL = JSON.parse(fs.readFileSync(process.argv[3])), cases = JSON.parse(fs.readFileSync(0));
FC.setConsts(M.consts);
const out = cases.map(c => { const fl = FL.find(x => x.id === c.f); fl.niveau = c.nv || null; fl.celer = c.cf || null; const S = FC.forecast(M.sets[fl.models], fl, c.target, c.i, c.v);
  return S ? Object.fromEntries(Object.entries(S).map(([k, a]) => [k, Array.from(a)])) : null; });
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(["node", "-e", js, str(HERE / "forecast_core.js"), str(HERE / "data" / "chain_models.json"), str(HERE / "data" / "chain_floods.json")],
                       input=json.dumps(cases), capture_output=True, text=True, check=True)
    res = json.loads(r.stdout)
    sets = {k: C.from_json(v) for k, v in mj["sets"].items()}
    worst, n = 0.0, 0
    for c, j in zip(cases, res):
        fl = next(x for x in floods if x["id"] == c["f"])
        C.LEVEL.clear()
        C.LEVEL.update(c.get("nv") or {})
        C.CELER.clear()
        C.CELER.update(c.get("cf") or {})
        py = py_forecast(fl, sets[fl["models"]], c["target"], c["i"], c["v"])
        C.LEVEL.clear()
        C.CELER.clear()
        ok_py = all(np.isfinite(v[:C.A + 1]).all() for v in py.values())
        if j is None:
            assert not ok_py, ("JS ohne Ergebnis, Python mit", c)
            continue
        for k, a in j.items():
            d = np.max(np.abs(np.array(a) - py[k]) / np.maximum(1.0, np.abs(py[k])))
            worst = max(worst, float(d))
            n += 1
    # Faktor 1 muss exakt dasselbe ergeben wie kein Faktor
    fl = floods[-1]
    for v in ("kette", "kette_regen"):
        if v not in C.NORAIN and not fl["rain"]:
            continue
        a = py_forecast(fl, sets[fl["models"]], "PZ", fl["i_peak"], v)
        C.CELER.update({"PZ": {"WI": 1.0, "SI": 1.0}, "AL": {"SE": 1.0}})
        b = py_forecast(fl, sets[fl["models"]], "PZ", fl["i_peak"], v)
        C.CELER.clear()
        assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a), "Faktor 1 ändert die Vorhersage"
    print(f"{len(cases)} Fälle, {n} Pegelreihen verglichen, größte relative Abweichung {worst:.2e}")
    assert worst < 1e-6, worst
    print("JavaScript-Vorhersage stimmt mit Python überein")


if __name__ == "__main__":
    main()
