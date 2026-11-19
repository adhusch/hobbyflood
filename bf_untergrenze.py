"""Untergrenze der Gebietsfeuchte aus der Bodenfeuchte: typischer Grundabfluss (Q / MQ am Bezugspegel) bei gegebener Bodenfeuchte im Winter.

    python bf_untergrenze.py [Quantil]      -> data/bodenfeuchte/bf_untergrenze.json (Standard: Quantil 0,25)

Anlass: Sturm 16.-18.11.2026. Das Modell schätzt die Gebietsfeuchte aus dem Abfluss vor dem Ereignis (B = Q / MQ). Nach dem trockenen Sommer
war das Grundwasser leer (Sinspelt B = 0,2), der Oberboden nach dem nassen Oktober aber gesättigt (Vortag 128-130 % nFK); bei dieser
Bodenfeuchte lag der Grundabfluss im Winter 2004-2022 im unteren Viertel bei 0,8 MQ, im Median bei 1,4-1,8 MQ. Das Modell hielt das Gebiet deshalb für trocken und setzte 3-4-mal zu wenig
Regen in Abfluss um.

Tabelle je Regengebiet: Bodenfeuchte des Vortags (DWD AMBAV 0-60 cm, % nFK) in Klassen zu 5 %, darin das Quantil von B (kleinstes Tagesmittel
der letzten 3 Tage, damit Hochwassertage die Tabelle nicht nach oben ziehen: bei 130-135 % sonst 2,4 statt 0,85 an Sinspelt; Oktober bis
März 2004-2022), nur Klassen mit mindestens MIN_N Tagen, nach oben nie fallend (laufendes Maximum). Im Modell (chain_fc.add_wetness) gilt
B_eff = max(B, Untergrenze(Bodenfeuchte)), zwischen den Stützstellen linear, außerhalb konstant. Ein Quantil statt des Medians, damit die
Untergrenze nur greift, wenn B deutlich unter dem Üblichen liegt (normale Jahre bleiben unverändert).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import chain_fc as C

OUT = C.HERE / "data" / "bodenfeuchte" / "bf_untergrenze.json"
EDGES = np.arange(60.0, 160.1, 5.0)
MIN_N = 40


def main(q: float = 0.25) -> None:
    C.setup_state()
    Q = C._S["Q"]
    mq = C.wet_mq()
    sm_tab = pd.read_csv(C.SM_FILE, index_col=0, parse_dates=True)
    out = dict(quantil=q, monate="10-3", jahre="2004-2022", tabelle={})
    for k in C.RAINK:
        b = (Q[C.WET_REF[k]] / mq[k]).resample("1D").mean().rolling(3).min()   # Grundabfluss: kleinstes Tagesmittel der letzten 3 Tage
        s = sm_tab[k].shift(1).reindex(b.index)                  # Bodenfeuchte des Vortags wie im Modell
        df = pd.DataFrame(dict(b=b, s=s)).dropna()
        df = df[(df.index.year >= 2004) & (df.index.year <= 2022) & df.index.month.isin([10, 11, 12, 1, 2, 3])]
        pts = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            x = df.b[(df.s >= lo) & (df.s < hi)]
            if len(x) >= MIN_N:
                pts.append((float((lo + hi) / 2), float(x.quantile(q)), int(len(x))))
        if not pts:
            continue
        sm_pts = [p[0] for p in pts]
        b_pts = list(np.maximum.accumulate([p[1] for p in pts]))
        out["tabelle"][k] = dict(sm=sm_pts, b=[round(v, 4) for v in b_pts], n=[p[2] for p in pts])
        print(k, " ".join(f"{s:.0f}%:{v:.2f}" for s, v in zip(sm_pts, b_pts)))
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.25)
