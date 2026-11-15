"""Fehlerzerlegung der Starkregen-Hochwasser (kreuzvalidiert, gemessener Regen als Vorhersage, Ausgabe 24 h vor dem Scheitel):
Scheitel, Volumen, Zeit, Stationsregen gegen Radar (RADKLIM), Abflussbeiwert. Ausgabe: data/storm_diag.csv"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import rain_fit as RF

HERE = Path(__file__).parent


def main() -> None:
    C.setup_state()
    S = C._S
    Q, W, frames, info = S["Q"], S["W"], S["frames"], S["info"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    sets = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]
    rad = RF.radar_hourly().reindex(pd.date_range(Q.index[0], Q.index[-1], freq="1h"))
    rows = []
    for b in range(len(C.BLOCKS)):
        models = C.from_json(sets[str(b)])
        for k in C.ORDER:
            ks = [x for x in C.chain(k) if x in C.RAINK]
            ar = np.array([info[x]["area_km2"] for x in ks]); A = ar.sum(); w = ar / A
            ps = sum(wi * np.nan_to_num(frames[x]["p"].to_numpy()) for wi, x in zip(w, ks))       # mm/h, 15 min
            pr = sum(wi * rad[x] for wi, x in zip(w, ks))                                            # mm je Stunde
            prmax = rad[[x + "_max" for x in ks]].max(axis=1)
            q = Qn[C.DN[k]]
            for p in C.events_for(k, Q, frames, C.BLOCKS[b], need_rain=True, W=W):
                if not C.is_storm(k, [p])[0]:
                    continue
                t0 = p - 96
                fc = C.forecast(models, k, Qn, frames, np.array([t0]), "kette_ideal")[k][0, C.A + 1:]
                ob = q[t0 + 1:t0 + 1 + C.H]
                base = q[t0]
                vo, vf = np.nansum(np.maximum(ob - base, 0)) * 900, np.sum(np.maximum(fc - base, 0)) * 900
                a, e = t0 - 48, p                                                                   # Regenfenster: 36 h vor dem Scheitel
                sl = slice(Q.index[a] + pd.Timedelta(minutes=15), Q.index[e])
                Ps = ps[a + 1:e + 1].sum() * C.DT
                h1 = pd.Series(ps[a + 1:e + 1]).rolling(4).mean().max()
                h3 = pd.Series(ps[a + 1:e + 1]).rolling(12).mean().max() * 3
                r = pr[sl]
                rows.append(dict(block=b, pegel=k, scheitel=str(Q.index[p]), q_obs=float(q[p]), hq10=float(q[p] / C._S["info"].get(k, {}).get("hq10", np.nan)) if False else None,
                                 peak=float(fc.max() / np.nanmax(ob)), vol=float(vf / vo) if vo > 0 else np.nan, dt_h=float((fc.argmax() - np.nanargmax(ob)) * C.DT),
                                 P_station=float(Ps), P_radar=float(r.sum()) if r.notna().all() else np.nan, i1_station=float(h1), i3_station=float(h3),
                                 i1_radar=float(r.max()), i1_pixel=float(prmax[sl].max()),
                                 rc_obs=float(vo / (A * 1e3) / Ps) if Ps > 0 else np.nan, rc_fc=float(vf / (A * 1e3) / Ps) if Ps > 0 else np.nan,
                                 mq=float(base / np.nanmedian(q)), area=float(A)))
    pd.DataFrame(rows).drop(columns="hq10").to_csv(HERE / "data" / "storm_diag.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
