"""Auswertung des Live-Betriebs: stellt alle archivierten Ausgaben (gui/archiv/live_*.html) gegen die inzwischen gemessenen Wasserstände.

  python live_eval.py [--archiv gui/archiv] [--out gui/auswertung.html] [--tage 14]

Je Pegel: Fehler nach Vorlauf für die Kette (Median), den ICON-D2-Hauptlauf, ICON-RUC, die Vorhersage der HVZ und "bleibt wie jetzt"; Anteil der
Messungen im 80-%-Band; je Ausgabe der vorhergesagte Höchststand gegen den gemessenen und der Warnstatus. Gerechnet in cm (heutige Beziehung).
Die Messwerte stammen aus den Ausgaben selbst (jede enthält die letzten 72 h); ausgewertet wird nur, wofür schon eine Messung vorliegt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LO, MED, HI = 1, 5, 9
LEADS_H = (3, 6, 12, 24, 36, 48)
ORDER = ["P2", "EC", "WI", "PZ", "SI", "GI", "SE", "AL"]


def read_page(p: Path) -> dict | None:
    try:
        h = p.read_text(encoding="utf-8")
        i = h.index("const D = ") + 10
        return json.JSONDecoder().raw_decode(h[i:])[0]
    except (ValueError, OSError):
        return None


def collect(archiv: Path, tage: int) -> dict:
    files = sorted(archiv.glob("live_*.html"))
    pages = []
    for f in files:
        D = read_page(f)
        if D and "res" in D and "meta" in D:
            pages.append(D)
    if not pages:
        return {}
    pages.sort(key=lambda D: D["meta"]["t0"])
    last = pd.Timestamp(pages[-1]["meta"]["t0"])
    pages = [D for D in pages if pd.Timestamp(D["meta"]["t0"]) >= last - pd.Timedelta(days=tage)]
    seen, uniq = set(), []
    for D in pages:                                            # je Ausgabezeit nur die erste Seite (stündliche Läufe können dieselbe Zeit treffen)
        if D["meta"]["t0"] not in seen:
            seen.add(D["meta"]["t0"])
            uniq.append(D)
    out = {}
    for k in ORDER:
        obs = {}
        issues = []
        for D in uniq:
            r = D["res"].get(k)
            if not r:
                continue
            t0, g0 = pd.Timestamp(D["meta"]["t0"]), pd.Timestamp(D["meta"]["grid0"])
            for i, w in enumerate(r["obs_w"]):
                if w is not None:
                    obs[g0 + pd.Timedelta(minutes=15 * i)] = float(w)
            hv = r.get("hvz") or {}
            issues.append(dict(t=t0, w0=r["w0"], med=r["band_w"][MED], lo=r["band_w"][LO], hi=r["band_w"][HI], d2=r["runs_w"].get("ICON-D2"),
                               env=[max(v[l] for v in r["runs_w"].values()) for l in range(len(r["band_w"][MED]))],
                               ruc=(r.get("ruc") or {}).get("w"), hvz=hv.get("p50") or [], hvz_t=hv.get("time"),
                               p48=r["warn"].get("p48"), status=r["warn"].get("status"), max48=r["max48"]["w"],
                               nb=[v["w_max"] for v in (r.get("nachbar") or {}).get("laeufe", [])], rs=r.get("regenschwellen")))
        if not issues:
            continue
        last_r = uniq[-1]["res"][k]
        out[k] = dict(dn=last_r["dn"], obs=pd.Series(obs).sort_index(), issues=issues, thresholds=last_r["thresholds"], warn=last_r["warn"])
    return out


def stats(g: dict) -> dict:
    obs = g["obs"]
    rows = {s: {h: [] for h in LEADS_H} for s in ("kette", "d2", "ruc", "hvz", "pers")}
    cov = {h: [0, 0] for h in LEADS_H}
    for it in g["issues"]:
        for h in LEADS_H:
            t = it["t"] + pd.Timedelta(hours=h)
            if t not in obs.index:
                continue
            o, l = obs[t], h * 4 - 1
            rows["kette"][h].append(it["med"][l] - o)
            rows["pers"][h].append(it["w0"] - o)
            if it["d2"]:
                rows["d2"][h].append(it["d2"][l] - o)
            if it["ruc"] and l < len(it["ruc"]):
                rows["ruc"][h].append(it["ruc"][l] - o)
            hv = {pd.Timestamp(a): b for a, b in it["hvz"]}
            if t in hv:
                rows["hvz"][h].append(hv[t] - o)
            cov[h][0] += 1
            cov[h][1] += int(it["lo"][l] <= o <= it["hi"][l])
    tab = {s: {h: (dict(n=len(v), bias=round(float(np.median(v)), 1), mae=round(float(np.mean(np.abs(v))), 1)) if v else None) for h, v in d.items()} for s, d in rows.items()}
    return dict(fehler=tab, band={h: dict(n=c[0], drin=c[1]) for h, c in cov.items()})


def build(archiv: Path, out: Path, tage: int) -> None:
    G = collect(archiv, tage)
    if not G:
        print("Auswertung: keine archivierten Ausgaben")
        return
    ms = lambda t: int(pd.Timestamp(t).value // 10 ** 6)       # noqa: E731
    data = dict(stand=str(max(g["issues"][-1]["t"] for g in G.values())), pegel={})
    for k, g in G.items():
        obs = g["obs"]
        full = obs.reindex(pd.date_range(obs.index[0], obs.index[-1], freq="15min"))
        iss = []
        for it in g["issues"]:
            end = it["t"] + pd.Timedelta(hours=48)
            seg = obs[(obs.index > it["t"]) & (obs.index <= end)]
            iss.append(dict(t=ms(it["t"]), w0=it["w0"], med=it["med"][3::4], lo=it["lo"][3::4], hi=it["hi"][3::4], env=[round(x, 1) for x in it["env"][3::4]],
                            d2=it["d2"][3::4] if it["d2"] else None, ruc=it["ruc"][3::4] if it["ruc"] else None,
                            hvz=[[ms(a), b] for a, b in it["hvz"]], p48=it["p48"], status=it["status"], max48=it["max48"],
                            nb=[min(it["nb"]), max(it["nb"])] if it["nb"] else None,
                            rs=None if not it["rs"] else dict(r10=it["rs"]["r10"], r50=it["rs"]["r50"], d2=it["rs"]["d2"], eps=[e[2] for e in it["rs"]["eps"]], dauer=it["rs"]["dauer_h"]),
                            obs_max=None if not len(seg) else float(seg.max()), obs_h=round(float((seg.index[-1] - it["t"]) / pd.Timedelta(hours=1)), 1) if len(seg) else 0))
        data["pegel"][k] = dict(dn=g["dn"], t0=ms(full.index[0]), obs=[None if not np.isfinite(v) else v for v in full.to_numpy()], issues=iss,
                                thresholds=g["thresholds"], warn=g["warn"], **stats(g))
    tpl = (HERE / "auswertung_template.html").read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    n = len(next(iter(data["pegel"].values()))["issues"])
    print(f"Auswertung: {n} Ausgaben, {out} ({out.stat().st_size // 1000} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archiv", default=str(HERE / "gui" / "archiv"))
    ap.add_argument("--out", default=str(HERE / "gui" / "auswertung.html"))
    ap.add_argument("--tage", type=int, default=14)
    a = ap.parse_args()
    build(Path(a.archiv), Path(a.out), a.tage)
