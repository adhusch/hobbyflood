"""Scheitel-Regel prüfen: Kettenvorhersage aller Hochwasser 2004-2022 (Modelle je Block ohne den Block) für verschiedene Abklingzeiten
HOLD_TAU_H (0 = Regel aus, inf = strenge Regel). Ausgabe data/hold_test.csv: NSE je Vorlauf und Scheitelverhältnis je Pegel und Variante.

  python hold_test.py 0 6 12 24 inf
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C

HERE = Path(__file__).parent
LEADS = (3, 6, 12, 24, 48)
VARS = ("kette_regen", "kette_vorhersage")


def job(tau: float) -> list:
    C.HOLD_TAU_H = tau
    S = C._S
    Q, W, frames = S["Q"], S["W"], S["frames"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    sets = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]
    rows = []
    for k in C.UPS:
        acc = {v: dict(sse=0, so=0, so2=0, n=0) for v in VARS}
        pk = {v: [] for v in VARS}
        for b in range(len(C.BLOCKS)):
            ev = C.events_for(k, Q, frames, C.BLOCKS[b], need_rain=True, W=W)
            if not len(ev):
                continue
            r = C.evaluate(C.from_json(sets[str(b)]), k, ev, VARS, Qn, frames)
            for v in VARS:
                for x in ("sse", "so", "so2", "n"):
                    acc[v][x] = acc[v][x] + r["acc"][v][x]
                pk[v].append(r["peaks"][v])
        before = C.OFF[C.OFF <= 0]
        for v in VARS:
            a = acc[v]
            nse = 1 - a["sse"] / (a["so2"] - a["so"] ** 2 / a["n"])
            p = np.concatenate(pk[v])                                  # [Ereignisse, Ausgaben vor dem Scheitel, (Verhältnis, Zeit)]
            rec = dict(tau=tau, pegel=k, variante=v, hw=len(p), **{f"nse_{L}h": round(float(nse[int(L / C.DT) - 1]), 3) for L in LEADS})
            for h in (24, 12, 6, 0):
                j = int(np.where(before == -int(h / C.DT))[0][0])
                rec[f"scheitel_med_{h}h"] = round(float(np.median(p[:, j, 0])), 2)
                rec[f"scheitel_p90_{h}h"] = round(float(np.quantile(p[:, j, 0], 0.9)), 2)
                rec[f"zeit_med_{h}h"] = round(float(np.median(p[:, j, 1])), 2)
            rows.append(rec)
    print("tau", tau, "fertig", flush=True)
    return rows


if __name__ == "__main__":
    taus = [float(x) for x in sys.argv[1:]] or [0.0, 12.0, 24.0, float("inf")]
    C.setup_state()
    with mp.get_context("fork").Pool(2) as pool:
        res = pool.map(job, taus)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(HERE / "data" / "hold_test.csv", index=False)
    print(df.to_string(index=False))
