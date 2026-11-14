"""Wasserstand-Abfluss-Beziehung (W-Q) je Pegel für die Anzeige in cm und die Meldestufen (python rating.py).

Die Vorhersage rechnet in Abfluss. Für Wasserstände in cm braucht es die heute gültige W-Q-Beziehung. Sie ändert sich mit dem Flussbett:
in Prümzurlay liegt der Wasserstand bei gleichem Abfluss seit Januar 2022 (nach dem Hochwasser 2021) rund 25 cm tiefer als vorher (bei 20 m³/s
76 statt 101 cm), in Seffern war er von Juli 2021 bis März 2022 rund 13 cm höher, in Alsdorf-Oberecken seit November 2021 rund 6 cm tiefer.
Deshalb nur Wertepaare aus der jüngsten Zeit ohne Sprung (Beginn je Pegel in START), dazu die Welle vom Oktober 2026 (Wasserportal-Abfluss und
Wasserstand der HVZ-Schnittstelle, soweit vorhanden) und für den oberen Bereich die Kennwerte der HVZ: HWx wird dort aus HQx mit der aktuell gültigen
W-Q-Beziehung berechnet, die Paare (HQx, HWx) sind also Punkte der heutigen Kurve.

Kurve: Mediane des Wasserstands in 30 logarithmischen Abflussklassen (mindestens 10 Werte) bis 80 % von HQ2, darüber die HVZ-Punkte HQ2 bis HQ100;
monoton gemacht (Pool-Adjacent-Violators), dazwischen linear in log Q. Über HQ100 mit der Steigung zwischen HQ50 und HQ100 verlängert (nur grob).
Ausgabe data/rating_curves.json (Stützstellen je Pegel, Meldehöhe als Abfluss) für die Karte.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
START = {"Prüm 2": "2016-01-01", "Echtershausen": "2016-01-01", "Wiersdorf": "2016-01-01", "Prümzurlay": "2022-01-01",
         "Sinspelt": "2016-01-01", "Giesdorf": "2016-01-01", "Seffern": "2022-04-01", "Alsdorf-Oberecken": "2021-11-01"}
ANCHORS = ("2", "10", "20", "50", "100")


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Monotone (nicht fallende) Anpassung nach Pool-Adjacent-Violators mit Gewichten."""
    blocks = []                                   # [Mittel, Gewicht, Anzahl]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            a, b = blocks.pop(), blocks.pop()
            ws = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / ws, ws, a[2] + b[2]])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


def build(name: str, w: pd.Series, q: pd.Series, k: dict, extra: pd.DataFrame | None = None) -> dict:
    s = (w.index >= START[name]) & w.notna() & q.notna() & (q > 0)
    qq, ww = q[s].to_numpy(), w[s].to_numpy()
    if extra is not None and len(extra):
        qq, ww = np.concatenate([qq, extra.q.to_numpy()]), np.concatenate([ww, extra.w.to_numpy()])
    hq2 = k["Q"]["HQ2"]
    lo, hi = np.quantile(qq, 0.005), min(qq.max(), 0.8 * hq2)
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), 31))
    nodes = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (qq >= a) & (qq < b)
        if m.sum() >= 10:
            nodes.append((float(np.median(qq[m])), float(np.median(ww[m])), float(m.sum())))
    data_n = len(nodes)
    for x in ANCHORS:
        nodes.append((float(k["Q"][f"HQ{x}"]), float(k["W"][f"HW{x}"]), 1e4))     # HVZ-Punkte stark gewichtet
    nodes.sort()
    qn = np.array([n[0] for n in nodes])
    wn = pav(np.array([n[1] for n in nodes]), np.array([n[2] for n in nodes]))
    keep = np.r_[True, np.diff(wn) > 0.05]                                      # gleiche Stufen zusammenfassen (umkehrbar)
    qn, wn = qn[keep], wn[keep]
    return dict(q=[round(float(x), 4) for x in qn], w=[round(float(x), 1) for x in wn], data_nodes=data_n, start=START[name],
                q_max_data=round(float(qq.max()), 1))


def w_of_q(c: dict, q: np.ndarray) -> np.ndarray:
    """Wasserstand (cm) aus Abfluss; linear in log Q, an den Enden mit der Steigung des ersten bzw. letzten Abschnitts."""
    lq, w = np.log(np.asarray(c["q"])), np.asarray(c["w"])
    x = np.log(np.maximum(np.asarray(q, float), 1e-4))
    y = np.interp(x, lq, w)
    s_hi = (w[-1] - w[-2]) / (lq[-1] - lq[-2])
    s_lo = (w[1] - w[0]) / (lq[1] - lq[0])
    y = np.where(x > lq[-1], w[-1] + s_hi * (x - lq[-1]), y)
    y = np.where(x < lq[0], w[0] + s_lo * (x - lq[0]), y)
    return np.maximum(y, 0.0)


def q_of_w(c: dict, w: float) -> float:
    lq, wn = np.log(np.asarray(c["q"])), np.asarray(c["w"])
    if w > wn[-1]:
        s = (wn[-1] - wn[-2]) / (lq[-1] - lq[-2])
        return float(np.exp(lq[-1] + (w - wn[-1]) / s))
    return float(np.exp(np.interp(w, wn, lq)))


def main() -> None:
    import pkgdata
    W, Q = pkgdata.load_all("W"), pkgdata.load_all("Q")
    K = json.loads((HERE / "data" / "pegelkennwerte.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(HERE / "data" / "wasserportal_event_2026-10.csv")
    jw = pd.read_csv(HERE / "data" / "json_w_event_2026-10.csv")
    out = {}
    print(f"{'Pegel':18s} {'ab':10s} {'Stütz.':>6s} {'Qmax':>6s}   W(MQ)  W(HQ2/2)  W(HQ2)=HW2  Meldehöhe → Q")
    for name in W.columns:
        k = K["pegel"][name]
        num = int(k["nummer"])
        q2 = ev[(ev.number == num) & (ev["var"] == "Q")].set_index("time_utc").value
        w2 = jw[(jw.number == num) & (jw["var"] == "W")].set_index("time_utc").value
        extra = pd.concat([q2.rename("q"), w2.rename("w")], axis=1).dropna() if len(w2) else None
        c = build(name, W[name], Q[name], k, extra)
        c["meldehoehe"] = k["meldehoehe"]
        c["q_meldehoehe"] = round(q_of_w(c, k["meldehoehe"]), 2) if k["meldehoehe"] else None
        c["oct2026"] = 0 if extra is None else int(len(extra))
        out[name] = c
        mq, hq2 = k["Q"]["MQ"], k["Q"]["HQ2"]
        wv = w_of_q(c, np.array([mq, hq2 / 2, hq2]))
        print(f"{name:18s} {c['start']:10s} {len(c['q']):6d} {c['q_max_data']:6.1f}   {wv[0]:5.0f}  {wv[1]:8.0f}  {wv[2]:5.0f}={k['W']['HW2']:<4d}"
              f"  {'' if not k['meldehoehe'] else f'{k['meldehoehe']} cm → {c['q_meldehoehe']} m³/s'}")
    meta = dict(quelle="LfU-Pegeldaten (Wertepaare W/Q ab START je Pegel), Welle Oktober 2026 (Wasserportal/HVZ), Kennwerte HQx/HWx der HVZ "
                       "(data/pegelkennwerte.json)", methode=__doc__.strip().split("\n\n")[1].replace("\n", " "))
    (HERE / "data" / "rating_curves.json").write_text(json.dumps(dict(meta=meta, pegel=out), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
