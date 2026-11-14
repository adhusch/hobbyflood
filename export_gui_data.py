"""Erzeugt gui/data.js (und optional eine einzelne HTML-Datei) aus Graph + Demo-Ergebnissen.

  python export_gui_data.py              schreibt gui/data.js
  python export_gui_data.py --standalone schreibt zusätzlich gui/pruem_gui_standalone.html
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import graph
import run_baselines as rb
import synth

HERE = Path(__file__).parent
GUI = HERE / "gui"

COL = {"Mehlenbach": 0, "Alfbach": 0, "Nims": 0, "Prüm": 1, "Enz": 2}
SKIP = {"LfU-Sauer", "Luxemburg"}
# Demo: synthetische Reihe -> Prüm-Knoten (nur zur Anzeige, keine echten Daten!)
DEMO_MAP = {"Giesdorf": "Giesdorf", "Seffern": "Seffern", "Echtershausen": "Echtershausen",
            "Wiersdorf": "Wiersdorf", "Stausee Bitburg": "Stausee_W"}
DEMO_RAIN = {"Seffern": "Regen_Seffern", "Wiersdorf": "Regen_Wiersdorf"}


def pack(index, values, nd=3):
    return {"t0": int(index[0].timestamp() * 1000), "dt": int((index[1] - index[0]).total_seconds() * 1000),
            "v": [round(float(x), nd) for x in values]}


def base():
    d = graph.load()
    d = d[~d["art"].isin(SKIP)]
    by_name = {r["name"]: i for i, r in d.iterrows()}
    nodes = []
    for i, r in d.iterrows():
        pos = int(r["pos"])
        row = pos if r["gewaesser"] == "Prüm" else (7 + pos if r["strang"] in ("Enz", "Nims") else {"Mehlenbach": 4, "Alfbach": 5}[r["strang"]])
        nodes.append(dict(id=i, number=graph.number(i), name=r["name"], river=r["gewaesser"], strang=r["strang"],
                          art=r["art"], vars=r["messgroessen"], certain=r["reihenfolge_gesichert"],
                          note=r["hinweis"], col=COL[r["strang"]], row=row,
                          area=float(r["catchment_km2"]) if r["catchment_km2"] else None))
    nodes.append(dict(id="MUENDUNG", number="", name="Mündung in die Sauer", river="Sauer", strang="Prüm",
                      art="Mündung", vars="", certain="ja", note="", col=1, row=15, area=None))
    ids = {n["id"] for n in nodes}
    edges = [[p, i] for i, ps in graph.parents(d).items() for p in ps if p in ids]
    edges += [[by_name["Prümzurlay"], "MUENDUNG"], [by_name["Alsdorf-Oberecken"], "MUENDUNG"]]
    return d, by_name, nodes, edges, dict(graph.q_reaches(d))


def pack_results(res, by_name, test_from, name_of=lambda x: x):
    out = []
    for r in res:
        out.append(dict(
            case=r["case"], target=name_of(r["target"]), sources=[name_of(s) for s in r["sources"]],
            extras=r["extras"], setting=r["setting"], ref=r["ref"],
            table=json.loads(r["table"].round(4).replace({np.nan: None}).to_json(orient="records")),
            preds={m: pack(r["obs"].index, p, 2) for m, p in r["preds"].items()},
            testFrom=int(test_from.timestamp() * 1000)))
    return out


def build_synth():
    d, by_name, nodes, edges, reaches = base()
    df = synth.generate()
    test_from = df.index[-8760]
    series = {}
    for name, col in DEMO_MAP.items():
        key = "W" if col == "Stausee_W" else "Q"
        series.setdefault(by_name[name], {})[key] = pack(df.index[-17520:], df[col].iloc[-17520:], 4 if key == "W" else 3)
    for name, col in DEMO_RAIN.items():
        series[by_name[name]]["N"] = pack(df.index[-17520:], df[col].iloc[-17520:], 2)
    res = rb.run_all(df, rb.SYNTH_CASES, test_from)
    results = pack_results(res, by_name, test_from, name_of=lambda n: by_name[n])
    return dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"), source="synthetisch",
                nodes=nodes, edges=edges, reaches=reaches, series=series, results=results)


def build_real(path):
    d, by_name, nodes, edges, reaches = base()
    wide = rb.load_snapshots(path, "15min")
    num2id = {graph.number(i): i for i in d.index}
    series = {}
    for col in wide.columns:
        num, _, var = col.partition("_")
        if num in num2id:
            series.setdefault(num2id[num], {})[var or "Q"] = pack(wide.index, wide[col].to_numpy(), 2)
    cut = wide.index[int(len(wide) * 0.65)]
    res = rb.run_all(wide, rb.real_cases(d, wide), cut, max_lag=12, own=4)
    results = pack_results(res, by_name, cut, name_of=lambda n: num2id[n])
    return dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source=f"Echtdaten hochwasser.rlp.de, {wide.index[0]:%d.%m.} bis {wide.index[-1]:%d.%m.%Y} (15 min)",
                real=True, nodes=nodes, edges=edges, reaches=reaches, series=series, results=results)


def build_daily(path):
    """3 Jahre Tagesmittel aus dem Wasserportal (data/wasserportal/daily_q.csv) mit den Baselines aus run_daily.py."""
    import pandas as pd

    import run_daily

    d, by_name, nodes, edges, reaches = base()
    num2id = {graph.number(i): i for i in d.index}
    df = pd.read_csv(path, index_col=0, parse_dates=True).asfreq("D")
    df.columns = [str(c) for c in df.columns]
    series = {num2id[c]: {"Q": pack(df.index, df[c].to_numpy(), 3)} for c in df.columns if c in num2id}
    cut = df.index[-365]
    res = rb.run_all(df, run_daily.CASES, cut, max_lag=3, own=3)
    results = pack_results(res, by_name, cut, name_of=lambda n: num2id[n])
    return dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source=f"Tagesmittel Wasserportal, {df.index[0]:%d.%m.%y}–{df.index[-1]:%d.%m.%y}, Rohdaten",
                real=True, daily=True, nodes=nodes, edges=edges, reaches=reaches, series=series, results=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="Demo mit synthetischen Daten statt Echtdaten")
    ap.add_argument("--real", default=str(HERE / "data" / "real_5d.csv"))
    ap.add_argument("--daily", default=str(HERE / "data" / "wasserportal" / "daily_q.csv"),
                    help="Tagesmittel-CSV (aus import_wasserportal.py --daily); wird als zweiter Datensatz angeboten, falls vorhanden")
    a = ap.parse_args()
    data = build_synth() if a.synthetic or not Path(a.real).exists() else build_real(a.real)
    GUI.mkdir(exist_ok=True)
    js = "window.PRUEM_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    if not a.synthetic and Path(a.daily).exists():
        js += "window.PRUEM_DATA_DAILY = " + json.dumps(build_daily(a.daily), ensure_ascii=False, separators=(",", ":")) + ";\n"
    (GUI / "data.js").write_text(js, encoding="utf-8")
    print("gui/data.js:", round(len(js) / 1e6, 2), "MB")
    if a.standalone:
        html = (GUI / "index.html").read_text(encoding="utf-8")
        marker = '<script src="data.js"></script>'
        assert marker in html
        (GUI / "pruem_gui_standalone.html").write_text(html.replace(marker, "<script>\n" + js + "</script>"), encoding="utf-8")
        print("gui/pruem_gui_standalone.html geschrieben")
