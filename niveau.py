"""Niveaufaktor je Pegel: Verhältnis Messung / reine Simulation (gemessene Oberlieger und gemessener Regen, ohne Fehlerkorrektur) je Jahr.

Hintergrund: Die Abflussreihen beruhen auf der jeweils gültigen Wasserstand-Abfluss-Beziehung. In Prümzurlay hat das Hochwasser vom Juli 2021 das
Gerinne freigespült; seit Januar 2022 gilt eine neue Beziehung, und der Pegel zeigt bei gleichem Zufluss aus Wiersdorf und Sinspelt 20 bis 30 %
mehr Abfluss als in den Jahren davor. An der Nims (Seffern, Alsdorf-Oberecken) gab es mehrere solche Wechsel. Die Kette ist fast nur an den Jahren
bis 2021 angepasst und liegt deshalb im heutigen Maßstab zu niedrig bzw. zu hoch.

  python niveau.py          -> data/niveau_jahre.csv (Verhältnis je Pegel und Jahr) und data/niveau.json (Faktor für den Betrieb)

Der Faktor ist das Verhältnis im jüngsten Zeitraum (SEIT je Pegel bis Ende der Daten) geteilt durch das Verhältnis in den Bezugsjahren. Er
multipliziert die Simulation des Pegels vor der Fehlerkorrektur (chain_fc.LEVEL, forecast_core.js). Für Hochwasser vor dem Wechsel gilt 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import uncert as UN

HERE = Path(__file__).parent
SEIT = {"PZ": "2022-01-01", "AL": "2022-04-01", "SE": "2022-04-01"}       # Beginn des heutigen Zustands. Alsdorf-Oberecken hängt an Seffern, und
                                                                           # Seffern lag von Juli 2021 bis März 2022 vorübergehend höher: daher ab April 2022
OHNE = ()
# Fest gesetzt statt aus der Simulation (die LfU-Reihen reichen nur bis 2022): Seffern seit April 2022 bei kleinem Abfluss 0,99, ab 4 MQ 1,31
# (51 Werte aus einem Ereignis); die Wasserportal-Tagesmittel 2023-2026 zeigen Seffern gegenüber Giesdorf (unverändert) um 19 % (0,5-2 MQ) bzw.
# 37 % (über 2 MQ) höher als 2008-2018. Für den Hochwasserbereich 1,3 (18.11.2026, Sturm: Seffern lag 24 h vorher deutlich zu tief).
FEST = {"SE": 1.3}
BEZUG = (2008, 2018)                                                       # Bezugsjahre (vor den Wechseln ab 2019 an der Nims)
MIN_MQ = 1.5                                                               # nur Zeiten mit Abfluss ab 1,5 MQ (dort zählt die Vorhersage)
STEP = 12                                                                  # Ausgabe alle 3 h


def pure_sim(models: dict, k: str, t0: np.ndarray) -> np.ndarray:
    """Simulation des Pegels k [Fälle, H] ab t0 mit gemessenen Oberliegern und gemessenem Regen, ohne Fehlerkorrektur."""
    S = C._S
    Qn, frames = S["Qn"], S["frames"]
    idx = t0[:, None] + np.arange(-C.A, C.H + 1)[None, :]
    m = models[k][C.rain_type(k) + "_winter"]
    X = {C.DN[u]: Qn[C.DN[u]][idx] for u in C.UPS[k]}
    P = {k: C.rain_window(frames[k], t0, True)} if m.lat else None
    return m.route(X, P)[:, C.A + 1:]


def ratios(k: str, models: dict) -> pd.DataFrame:
    S = C._S
    Q, Qn = S["Q"], S["Qn"]
    q = Qn[C.DN[k]]
    mq = UN.kennwerte()[C.DN[k]]["Q"]["MQ"]
    n = len(q)
    t0 = np.arange(C.A + C.LEAD + 1, n - C.H - 1, STEP)
    idx = t0[:, None] + np.arange(-C.A, C.H + 1)[None, :]
    ok = np.isfinite(q[idx]).all(axis=1)
    for u in C.UPS[k]:
        ok &= np.isfinite(Qn[C.DN[u]][idx]).all(axis=1)
    fr = S["frames"][k]
    ridx = t0[:, None] + np.arange(-C.A - C.LEAD, C.H + 1)[None, :]
    ok &= np.isfinite(np.asarray(fr["p"])[ridx]).all(axis=1) & np.isfinite(np.asarray(fr["api"])[ridx]).all(axis=1)
    t0 = t0[ok]
    rows = []
    for j in range(0, len(t0), 400):
        tt = t0[j:j + 400]
        sim = pure_sim(models, k, tt)
        L = np.array([12, 24, 48]) * 4 - 1                                  # Vorlauf 12, 24, 48 h (Einschwingen der Speicher vorbei)
        obs = q[tt[:, None] + L[None, :] + 1]
        for a in range(len(tt)):
            for b in range(len(L)):
                if obs[a, b] >= MIN_MQ * mq and sim[a, L[b]] > 0:
                    rows.append((Q.index[tt[a]], obs[a, b] / sim[a, L[b]], obs[a, b] / mq))
    return pd.DataFrame(rows, columns=["t", "r", "q_mq"])


def main() -> None:
    C.setup_state()
    C._S["Qn"] = {c: C._S["Q"][c].to_numpy() for c in C._S["Q"].columns}
    doc = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    models = C.from_json(doc["sets"]["alle"])
    out, fac = [], {}
    for k in SEIT:
        d = ratios(k, models)
        y = d.groupby(d.t.dt.year).r.agg(["median", "size"])
        for yr, r in y.iterrows():
            out.append(dict(pegel=C.DN[k], jahr=int(yr), verhaeltnis=round(float(r["median"]), 3), werte=int(r["size"])))
        ref = d[(d.t.dt.year >= BEZUG[0]) & (d.t.dt.year <= BEZUG[1])].r.median()
        new = d[d.t >= pd.Timestamp(SEIT[k])]
        hi = new[new.q_mq >= 4]
        alle = float(new.r.median() / ref)
        hoch = None if len(hi) < 20 else float(hi.r.median() / d[(d.t.dt.year >= BEZUG[0]) & (d.t.dt.year <= BEZUG[1]) & (d.q_mq >= 4)].r.median())
        # Faktor aus dem Hochwasserbereich (ab 4 MQ), wenn dort mindestens 100 Werte liegen, sonst aus allen Werten ab 1,5 MQ
        f = hoch if hoch is not None and len(hi) >= 100 else alle
        fac[k] = dict(faktor=1.0 if k in OHNE else FEST.get(k, round(f, 2)), seit=SEIT[k], werte=int(len(new)), werte_ab_4_mq=int(len(hi)),
                      ab_1_5_mq=round(alle, 3), ab_4_mq=None if hoch is None else round(hoch, 3), angewendet=k not in OHNE)
    pd.DataFrame(out).to_csv(HERE / "data" / "niveau_jahre.csv", index=False)
    (HERE / "data" / "niveau.json").write_text(json.dumps(dict(
        hinweis="Faktor auf die Simulation des Pegels vor der Fehlerkorrektur; Messung/Simulation seit dem Wechsel der Wasserstand-Abfluss-Beziehung, "
                f"bezogen auf {BEZUG[0]}-{BEZUG[1]} (niveau.py)", pegel=fac), ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(out).pivot(index="jahr", columns="pegel", values="verhaeltnis").loc[2005:].to_string())
    print(json.dumps(fac, ensure_ascii=False))


if __name__ == "__main__":
    main()
