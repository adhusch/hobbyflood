"""Prüft die JavaScript-Seite der Unsicherheit (UQ in forecast_core.js) gegen Python (python test_uncert_js.py, braucht node):
Band (Quantile je 15-min-Vorlauf, linear zwischen den gespeicherten Vorläufen), Überschreitungswahrscheinlichkeit (laufendes Maximum),
Scheitel mit Spanne und die W-Q-Umrechnung (rating.w_of_q / q_of_w). Fälle: Hochwasserfenster der Karte, mehrere Pegel und Ausgabezeiten."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import chain_fc as C
import rating as RT
import uncert as UN
from test_forecast_js import py_forecast

HERE = Path(__file__).parent


def py_analyse(U: dict, fl: dict, key: str, i: int, s: np.ndarray, variant: str) -> dict:
    A, H = C.A, C.H
    if variant == "kette_vorhersage":
        tab = U["vorhersage"].get(fl["models"], U["vorhersage"]["alle"])[key]
    else:
        tab = (U["regen"][fl["models"]] if fl["rain"] else U["kette"]["alle"])[key]
    code = int(U["floods"][fl["id"]][key][i], 16)
    st = code & 3
    hv = 0 if (not fl["rain"] or variant == "kette") else ((code >> 3) & 1 if variant == "kette_vorhersage" else (code >> 2) & 1)
    mx = max(s[A:A + H + 1])
    lv = mx / U["mq"][key]
    cls = st * 6 + (2 if lv >= 8 else 1 if lv >= 2 else 0) * 2 + int(s[A] >= U["wet_edge"] * U["mq"][key]) + 24 * hv
    T = np.array(tab["q"][cls])                                  # [Vorläufe, Stufen]
    leads = np.round(np.array(U["leads_h"]) * 4).astype(int)
    L = np.arange(1, H + 1)
    Ti = np.stack([np.interp(L, leads, T[:, k]) for k in range(T.shape[1])], axis=1)
    d = U["offset"][key]
    f = np.maximum(s[A + 1:A + H + 1], 0)
    q = np.maximum((f[:, None] + d) * np.exp(Ti) - d, 0)

    def exceed(thr: float) -> np.ndarray:
        x = np.log((thr + d) / (f + d))
        return np.maximum.accumulate(1 - UN.cdf_fast(x[None, :], Ti[None, :, :])[0])
    peak = None
    ja = int(np.argmax(s[A + 1:A + H + 1])) + A + 1
    fcmax, fcarg = s[ja], ja - A
    if st == 2 and tab["peak"] and fcmax > U["peak_rise"] * s[A] and fcmax >= U["peak_min_mq"] * U["mq"][key] and fcarg < H - 4:
        e = np.array(U["peak_buckets_h"])
        b = int(np.clip(np.searchsorted(e, fcarg * 0.25, side="left") - 1, 0, len(e) - 2))
        peak = dict(q=[max((fcmax + d) * np.exp(r) - d, 0) for r in tab["peak"]["lnr"][b]], t_h=[fcarg * 0.25 + x for x in tab["peak"]["dt"][b]])
    return dict(cls=cls, q=q, exceed=exceed, peak=peak)


def main() -> None:
    U = json.loads((HERE / "data" / "uncert_tables.json").read_text(encoding="utf-8"))
    mj = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    floods = json.loads((HERE / "data" / "chain_floods.json").read_text(encoding="utf-8"))
    rc = json.loads((HERE / "data" / "rating_curves.json").read_text(encoding="utf-8"))["pegel"]
    K = UN.kennwerte()
    sets = {k: C.from_json(v) for k, v in mj["sets"].items()}
    cases = []
    for fl in floods:
        for key in ("PZ", "AL", "WI", "P2", "EC"):
            for i in range(fl["i_peak"] - 40 * 4, fl["i_peak"] + 8 * 4, 13):
                v = ("kette_regen", "kette_vorhersage")[(i // 13) % 2] if fl["rain"] else "kette"
                cases.append(dict(f=fl["id"], key=key, i=i, v=v,
                                  thr=[K[C.DN[key]]["Q"]["HQ2"] * 0.5, K[C.DN[key]]["Q"]["HQ2"], K[C.DN[key]]["Q"]["HQ10"]]))
    js = r"""
const fs = require("fs"), M = require(process.argv[1]), FC = M, UQ = M.UQ;
const MJ = JSON.parse(fs.readFileSync(process.argv[2])), FL = JSON.parse(fs.readFileSync(process.argv[3])), U = JSON.parse(fs.readFileSync(process.argv[4]));
const RC = JSON.parse(fs.readFileSync(process.argv[5])).pegel, cases = JSON.parse(fs.readFileSync(0));
FC.setConsts(MJ.consts); UQ.setTables(U);
const out = cases.map(c => { const fl = FL.find(x => x.id === c.f), S = FC.forecast(MJ.sets[fl.models], fl, c.key, c.i, c.v);
  if (!S) return null;
  const s = S[c.key], an = UQ.analyse(fl, c.key, c.i, s, FC.A, FC.H, c.v);
  return { s: Array.from(s), cls: an.cls, q: an.q, ex: c.thr.map(t => Array.from(an.exceed(t))), peak: an.peak }; });
const w = Object.fromEntries(Object.entries(RC).map(([g, c]) => [g, [0.05, 0.5, 3, 20, 80, 150, 300, 600].map(q => UQ.wOfQ(c, q))]));
const qw = Object.fromEntries(Object.entries(RC).map(([g, c]) => [g, [30, 100, 200, 400, 700].map(x => UQ.qOfW(c, x))]));
process.stdout.write(JSON.stringify({ out, w, qw }));
"""
    r = subprocess.run(["node", "-e", js, str(HERE / "forecast_core.js"), str(HERE / "data" / "chain_models.json"), str(HERE / "data" / "chain_floods.json"),
                        str(HERE / "data" / "uncert_tables.json"), str(HERE / "data" / "rating_curves.json")],
                       input=json.dumps(cases), capture_output=True, text=True, check=True)
    res = json.loads(r.stdout)
    worst_q = worst_p = worst_pk = 0.0
    n = npk = nheavy = nvor = 0
    for c, j in zip(cases, res["out"]):
        if j is None:
            continue
        fl = next(x for x in floods if x["id"] == c["f"])
        s = py_forecast(fl, sets[fl["models"]], c["key"], c["i"], c["v"])[c["key"]]
        assert np.max(np.abs(np.array(j["s"]) - s) / np.maximum(1, np.abs(s))) < 1e-6
        py = py_analyse(U, fl, c["key"], c["i"], s, c["v"])
        assert py["cls"] == j["cls"], (c, py["cls"], j["cls"])
        nheavy += int(j["cls"] >= 24)
        nvor += int(c["v"] == "kette_vorhersage")
        worst_q = max(worst_q, float(np.max(np.abs(np.array(j["q"]) - py["q"]) / np.maximum(1, py["q"]))))
        for t, e in zip(c["thr"], j["ex"]):
            worst_p = max(worst_p, float(np.max(np.abs(np.array(e) - py["exceed"](t)))))
        assert (py["peak"] is None) == (j["peak"] is None), (c, py["peak"], j["peak"])
        if py["peak"]:
            npk += 1
            worst_pk = max(worst_pk, max(abs(a - b) / max(1, abs(b)) for a, b in zip(j["peak"]["q"] + j["peak"]["t_h"], py["peak"]["q"] + py["peak"]["t_h"])))
        n += 1
    wq = max(abs(a - b) for g, c in rc.items() for a, b in zip(res["w"][g], RT.w_of_q(c, np.array([0.05, 0.5, 3, 20, 80, 150, 300, 600]))))
    qq = max(abs(a - b) / b for g, c in rc.items() for a, b in zip(res["qw"][g], [RT.q_of_w(c, x) for x in (30, 100, 200, 400, 700)]))
    print(f"{n} Fälle ({npk} mit Scheitel, {nheavy} in einer Starkregenlage, {nvor} mit perfektem Regen): Band rel. {worst_q:.1e}, Wahrscheinlichkeit {worst_p:.1e}, Scheitel {worst_pk:.1e}; W(Q) {wq:.1e} cm, Q(W) rel. {qq:.1e}")
    assert worst_q < 1e-6 and worst_p < 1e-6 and worst_pk < 1e-6 and wq < 1e-6 and qq < 1e-6
    assert npk > 0, "kein Fall mit Scheitelvorhersage geprüft"
    assert nheavy > 0 and nvor > 0, "keine Starkregenlage oder kein Fall mit perfektem Regen geprüft"
    print("JavaScript-Unsicherheit stimmt mit Python überein")


if __name__ == "__main__":
    main()
