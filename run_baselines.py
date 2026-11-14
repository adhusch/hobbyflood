"""Baselines auf synthetischen Daten (Standard) oder auf gesammelten Snapshots.

  python run_baselines.py                 synthetische Kaskade, Tabelle ausgeben
  python run_baselines.py --snapshots data/snapshots.csv   echte Daten (Graph aus pegel_netz.csv)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import graph
import ro
import synth

MODELS = [ro.Persistence, ro.ScaledLag, ro.NNLSTransfer, ro.RidgeARX, ro.GBM, ro.LinearPlusGBM]

# (Bezeichnung, Ziel, Quellen, Zusatzfeatures)
SYNTH_CASES = [
    ("Seffern <- Giesdorf (freie Strecke)", "Seffern", ["Giesdorf"], []),
    ("Seffern <- Giesdorf + Regen Zwischengebiet", "Seffern", ["Giesdorf"], ["Regen_Seffern"]),
    ("Wiersdorf <- Echtershausen (Speicher dazwischen)", "Wiersdorf", ["Echtershausen"], []),
    ("Wiersdorf <- Echtershausen + Seestand", "Wiersdorf", ["Echtershausen"], ["Stausee_W"]),
]


def run_case(df, label, target, sources, extras, test_from, max_lag=24, own=6):
    out = []
    for setting, own in (("A mit eigener Historie", own), ("B ohne eigene Historie", 0)):
        X, y = ro.make_dataset(df, target, sources, max_lag=max_lag, own_lags=own, extras=extras)
        tr, te = X.index < test_from, X.index >= test_from
        models = [M(target, sources) for M in MODELS if not (own == 0 and M is ro.Persistence)]
        ref = "Persistenz" if own else "Skaliert+Lag"
        table, preds = ro.evaluate(models, X[tr], y[tr], X[te], y[te], ref)
        out.append(dict(case=label, target=target, sources=sources, extras=extras, setting=setting,
                        ref=ref, table=table, obs=y[te], preds=preds))
    return out


def run_all(df, cases, test_from, **kw):
    res = []
    for label, target, sources, extras in cases:
        res += run_case(df, label, target, sources, extras, test_from, **kw)
    return res


def show(res):
    pd.options.display.float_format = "{:.3f}".format
    for r in res:
        print(f"\n== {r['case']} | Setting {r['setting']} | Skill relativ zu {r['ref']}")
        print(r["table"].to_string(index=False))


def load_snapshots(path, freq="15min"):
    """Snapshots (time_utc,number,var,value) -> breite Tabelle; Q-Spalten heißen <nummer>, W-Spalten <nummer>_W."""
    s = pd.read_csv(path, parse_dates=["time_utc"], dtype={"number": str})
    s["col"] = np.where(s["var"] == "Q", s["number"], s["number"] + "_" + s["var"])
    wide = s.pivot_table(index="time_utc", columns="col", values="value", aggfunc="last")
    return wide.resample(freq).mean().interpolate(limit=2)


def real_cases(d, wide):
    cases = []
    for t, v in graph.q_reaches(d).items():
        tn, sn = graph.number(t), [graph.number(s) for s in v["sources"]]
        if v["sources"] and tn in wide and all(s in wide for s in sn):
            if v["reservoir"]:
                continue
            cases.append((f"{d.at[t, 'name']} <- {', '.join(d.at[s, 'name'] for s in v['sources'])}", tn, sn, []))
    # Sonderfall Stausee: Wiersdorf aus Echtershausen, optional mit Seestand
    if {"26280402", "26280300", "26280366_W"} <= set(wide.columns):
        cases.append(("Wiersdorf <- Echtershausen (Stausee dazwischen)", "26280402", ["26280300"], []))
        cases.append(("Wiersdorf <- Echtershausen + Seestand", "26280402", ["26280300"], ["26280366_W"]))
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots")
    ap.add_argument("--freq", default="15min")
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--own-lags", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.35)
    a = ap.parse_args()
    if a.snapshots:
        d = graph.load()
        wide = load_snapshots(a.snapshots, a.freq)
        cut = wide.index[int(len(wide) * (1 - a.test_frac))]
        print(f"{len(wide)} Zeitschritte {wide.index[0]} bis {wide.index[-1]}, Testbeginn {cut}")
        show(run_all(wide, real_cases(d, wide), cut, max_lag=a.max_lag, own=a.own_lags))
    else:
        df = synth.generate()
        show(run_all(df, SYNTH_CASES, df.index[-8760]))
