"""Prüfung des Niveaufaktors: Median Vorhersage/Messung je Vorlauf im heutigen Zustand mit und ohne Faktor, dazu die Bezugsjahre 2012-2018 ohne
Faktor (kreuzvalidierte Modellsätze). Ausgabe: data/niveau_check.csv"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import niveau as N
import uncert as UN

HERE = Path(__file__).parent


def bias(models, k, a, b, variant):
    S = C._S
    Q, Qn = S["Q"], S["Qn"]
    q = Qn[C.DN[k]]
    mq = UN.kennwerte()[C.DN[k]]["Q"]["MQ"]
    i0, i1 = Q.index.get_loc(pd.Timestamp(a)), min(Q.index.get_loc(pd.Timestamp(b)), len(q) - C.H - 2)
    t0 = np.arange(i0, i1, 12)
    idx = t0[:, None] + np.arange(-C.A, C.H + 1)[None, :]
    ok = np.ones(len(t0), bool)
    for x in C.chain(k):
        ok &= np.isfinite(Qn[C.DN[x]][idx]).all(axis=1)
    t0 = t0[ok]
    L = np.array([6, 12, 24, 48]) * 4
    num = {h: [] for h in L}
    for j in range(0, len(t0), 300):
        tt = t0[j:j + 300]
        fc = C.forecast(models, k, Qn, S["frames"], tt, variant)[k][:, C.A + 1:]
        for h in L:
            o = q[tt + h]
            m = o >= N.MIN_MQ * mq
            num[h] += list(fc[m, h - 1] / o[m])
    return {f"{h // 4}h": round(float(np.median(v)), 2) if len(v) > 50 else None for h, v in num.items()}, len(num[L[2]])


def main():
    C.setup_state()
    C._S["Qn"] = {c: C._S["Q"][c].to_numpy() for c in C._S["Q"].columns}
    sets = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]
    fac = json.loads((HERE / "data" / "niveau.json").read_text(encoding="utf-8"))["pegel"]
    rows = []
    for k in ("PZ", "AL"):
        for variant in ("kette", "kette_vorhersage"):
            for name, blk, a, b, lev in (("Bezug 2012-2015", "2", "2012-01-01", "2015-12-31", None), ("Bezug 2016-2018", "3", "2016-01-01", "2018-12-31", None),
                                         ("heute, ohne Faktor", "4", fac[k]["seit"], "2022-12-31", None), ("heute, mit Faktor", "4", fac[k]["seit"], "2022-12-31", fac[k]["faktor"])):
                C.LEVEL.clear()
                if lev:
                    C.LEVEL[k] = lev
                r, n = bias(C.from_json(sets[blk]), k, a, b, variant)
                rows.append(dict(pegel=C.DN[k], variante=variant, zeitraum=name, **r, werte=n))
                print(rows[-1], flush=True)
    C.LEVEL.clear()
    pd.DataFrame(rows).to_csv(HERE / "data" / "niveau_check.csv", index=False)


if __name__ == "__main__":
    main()
