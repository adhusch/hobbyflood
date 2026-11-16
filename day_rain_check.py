"""Prüfung des Stundenregens aus Tagessummen + ERA5 (day_rain.py) an 2004-2022: dieselbe Kette (kreuzvalidierte Modelle, nicht neu angepasst),
einmal mit dem gemessenen Stundenregen, einmal mit dem Ersatzregen als bekannte Vorhersage. Ausgabe: data/tag/check_events.csv"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import rain_fit as RF

HERE = Path(__file__).parent


def main():
    C.setup_state()
    S = C._S
    Q, W, frames, info = S["Q"], S["W"], S["frames"], S["info"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    F2 = C.add_wetness(RF.rain_frames(Q.index, 1.0, extra=C.HW, source="tag")[0], Q)
    pm1, pm2 = S["pm"], C.storm_index(F2, info)
    sets = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]
    rows = []
    for b in range(len(C.BLOCKS)):
        models = C.from_json(sets[str(b)])
        for k in C.ORDER:
            ev = C.events_for(k, Q, frames, C.BLOCKS[b], need_rain=True, W=W)
            if not len(ev):
                continue
            st = C.is_storm(k, ev)
            q = Qn[C.DN[k]]
            for h in (24, 12, 6):
                t0 = ev - h * 4
                ok = np.array([np.isfinite(np.asarray(F2[x]["p"])[t - C.A - C.LEAD:t + C.H + 1]).all() and np.isfinite(np.asarray(F2[x]["api"])[t - C.A - C.LEAD - 1]) for t in t0 for x in [k if k in C.RAINK else C.chain(k)[0]]])
                S["pm"] = pm1
                f1 = C.forecast(models, k, Qn, frames, t0, "kette_ideal")[k][:, C.A + 1:]
                S["pm"] = pm2
                try:
                    f2 = C.forecast(models, k, Qn, F2, t0, "kette_ideal")[k][:, C.A + 1:]
                finally:
                    S["pm"] = pm1
                for i, p in enumerate(ev):
                    if not (np.isfinite(f2[i]).all() and np.isfinite(f1[i]).all()):
                        continue
                    rows.append(dict(block=b, pegel=k, scheitel=str(Q.index[p]), storm=bool(st[i]), vor_h=h, q=float(q[p]),
                                     r_mess=float(max(f1[i].max(), q[p - h * 4]) / q[p]), r_ersatz=float(max(f2[i].max(), q[p - h * 4]) / q[p]),
                                     dt_mess=float((f1[i].argmax() + 1) / 4 - h), dt_ersatz=float((f2[i].argmax() + 1) / 4 - h)))
        print("Block", b, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "data" / "tag" / "check_events.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
