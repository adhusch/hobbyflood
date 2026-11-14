"""Baut gui/lag_map.html: Karte der Pegel mit den Lags der Welle vom 24.–26.10.2026 und der Tagesmittel-Gegenprobe.

  python build_lag_map.py [--event data/wasserportal_event_2026-10.csv]

Quelle der Lags: die vollständige Welle (lags_event.py: Scheitel, Schwerpunkt, Fit, Korrelation, Anteile des Anstiegs/Rückgangs).
Zum Vergleich werden die vorläufigen Verfahren A, B, C aus lags.py (5-Tage-Fenster mit noch steigendem Ast, data/real_5d.csv) und
die Spitzentag-Gegenprobe aus data/wasserportal/daily_q.csv mitgeführt. Rechnet UTM32 nach WGS84 um und schreibt eine einzelne
HTML-Datei (Vorlage lag_map_template.html).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import export_gui_data as eg
import graph
import confluence as cf
import lags_event as le

HERE = Path(__file__).parent


def utm32_to_wgs84(E: float, N: float) -> tuple[float, float]:
    """ETRS89/UTM32 (EPSG:25832) -> Breite, Länge in Grad (Krüger-Reihe, GRS80)."""
    a, f, k0, E0, lam0 = 6378137.0, 1 / 298.257222101, 0.9996, 500000.0, math.radians(9.0)
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    beta = [n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96 - n ** 4 / 360,
            n ** 2 / 48 + n ** 3 / 15 - 437 * n ** 4 / 1440,
            17 * n ** 3 / 480 - 37 * n ** 4 / 840,
            4397 * n ** 4 / 161280]
    delta = [2 * n - 2 * n ** 2 / 3 - 2 * n ** 3 + 116 * n ** 4 / 45,
             7 * n ** 2 / 3 - 8 * n ** 3 / 5 - 227 * n ** 4 / 45,
             56 * n ** 3 / 15 - 136 * n ** 4 / 35,
             4279 * n ** 4 / 630]
    xi, eta = N / (k0 * A), (E - E0) / (k0 * A)
    xi_p = xi - sum(b * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, b in enumerate(beta, 1))
    eta_p = eta - sum(b * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, b in enumerate(beta, 1))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    phi = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(delta, 1))
    lam = lam0 + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(phi), math.degrees(lam)


# ---------- Verfahren aus lags.py (gleiche Rechnung, Ergebnisse als Daten) ----------
def xcorr(a, b, maxlag):
    out = []
    for L in range(maxlag + 1):
        x = a[:len(a) - L] if L else a
        out.append(np.corrcoef(x, b[L:])[0, 1])
    return np.array(out)


def lag_methods(w: pd.DataFrame, u: str, d: str) -> dict:
    ra = xcorr(w[u].to_numpy(), w[d].to_numpy(), 96)
    ba = int(np.nanargmax(ra))
    pl = np.where(ra >= ra[ba] - 0.005)[0]
    rb = xcorr(w[u].diff(4).to_numpy()[4:], w[d].diff(4).to_numpy()[4:], 48)
    bb = int(np.nanargmax(rb))
    t0 = pd.Timestamp("2026-10-24 21:00", tz="UTC")

    def cross(series, frac):
        seg = series[series.index >= t0]
        base, top = seg.iloc[:4].mean(), seg.max()
        lvl = base + frac * (top - base)
        idx = np.where(seg.to_numpy() >= lvl)[0]
        if len(idx) == 0 or idx[0] == 0:
            return None
        i = idx[0]
        v = seg.to_numpy()
        f = (lvl - v[i - 1]) / (v[i] - v[i - 1]) if v[i] != v[i - 1] else 0
        return seg.index[i - 1] + (seg.index[i] - seg.index[i - 1]) * f

    c = []
    for fr in (0.1, 0.25, 0.5, 0.75):
        cu, cd = cross(w[u], fr), cross(w[d], fr)
        c.append(round((cd - cu).total_seconds() / 3600, 2) if cu is not None and cd is not None else None)
    return dict(A=dict(best=ba * 0.25, lo=int(pl.min()) * 0.25, hi=int(pl.max()) * 0.25, r=round(float(ra[ba]), 3)),
                B=dict(best=bb * 0.25, r=round(float(rb[bb]), 3), r0=round(float(rb[0]), 3)),
                C=dict(f=[0.1, 0.25, 0.5, 0.75], lag=c))


def peak_day_check(daily: pd.DataFrame, up: str, down: str, n=30, gap=5) -> dict | None:
    if up not in daily or down not in daily:
        return None
    ev = []
    for t, _ in daily[down].sort_values(ascending=False).items():
        if all(abs((t - e).days) > gap for e in ev):
            ev.append(t)
        if len(ev) == n:
            break
    offs = []
    for t in ev:
        win = daily[up].loc[t - pd.Timedelta(days=2): t + pd.Timedelta(days=2)]
        offs.append((win.idxmax() - t).days)
    o = np.array(offs)
    cnt = lambda a: dict(n=int(len(a)), same=int((a == 0).sum()), early=int((a == -1).sum()),
                         late=int((a == 1).sum()), other=int((abs(a) >= 2).sum()))
    return dict(all=cnt(o), big=cnt(o[:10]), small=cnt(o[10:]))


def mez(ts: pd.Timestamp) -> str:
    return (ts + pd.Timedelta(hours=1)).strftime("%d.%m. %H:%M")


def r2(x):
    return None if x is None else round(float(x), 2)


def event_block(W: pd.DataFrame, u: str, d: str, joint: dict | None) -> dict:
    """Ergebnisse von lags_event.py für eine Strecke, als JSON-taugliche Daten."""
    t = le.pair_table(W, u, d)
    fz = lambda f: None if not f else dict(lag=r2(f["lag_h"]), a=r2(f["a"]), c=r2(f["c"]), nse=round(f["nse"], 3), n=f["n"])
    pu, pdn = t["peak_up"], t["peak_dn"]
    return dict(
        peak=dict(lag=r2(t["lag_peak_h"]), lagMid=r2(t["lag_peak_mid_h"]), platUp=r2(pu["plateau_h"]), platDn=r2(pdn["plateau_h"]), tUp=mez(pu["t"]), tDn=mez(pdn["t"]), qUp=r2(pu["q"]), qDn=r2(pdn["q"])),
        centroid=r2(t["lag_centroid_h"]),
        corr=dict(level=dict(lag=r2(t["corr_level_h"]), r=round(t["corr_level_r"], 3)),
                  diff=dict(lag=r2(t["corr_diff_h"]), r=round(t["corr_diff_r"], 3))),
        fit=dict(all=fz(t["fit"]), rise=fz(t["fit_rise"]), fall=fz(t["fit_fall"])),
        joint=joint,
        levels=[dict(f=l["f"], qUp=r2(l["q_up"]), qDn=r2(l["q_dn"]), rise=r2(l["rise_h"]), fall=r2(l["fall_h"])) for l in t["levels"]],
    )


def event_block_w(W: pd.DataFrame, u: str, d: str, primary: dict | None = None, alt: dict | None = None) -> dict:
    """Wie event_block, aber für Wasserstände (cm) einer Strecke zwischen kommunalen Pegeln. primary/alt ersetzen den Fit über das Fenster."""
    e = event_block(W, u, d, None)
    for k in ("qUp", "qDn"):
        e["peak"][k] = None
    if primary:
        e["fit"]["all"] = dict(lag=r2(primary["lag"]), a=r2(primary["a"]), c=None, nse=round(primary["nse"], 3), n=None,
                               lo=r2(primary["lo"]), hi=r2(primary["hi"]))
    if alt:
        e["fit"]["alt"] = dict(lag=r2(alt["lag"]), nse=round(alt["nse"], 3), lo=r2(alt["lo"]), hi=r2(alt["hi"]), label=alt["label"])
    for l in e["levels"]:
        l["qUp"] = None
    return e


def build(event_path: str) -> dict:
    d, by_name, nodes, edges, _ = eg.base()
    id_by_num = {graph.number(i): i for i in d.index}
    nm = {"Prüm 2": "26280050", "Echtershausen": "26280300", "Wiersdorf": "26280402", "Prümzurlay": "26280504",
          "Sinspelt": "26280650", "Giesdorf": "26280708", "Seffern": "26280800", "Alsdorf-Oberecken": "26280854"}
    s = pd.read_csv(HERE / "data" / "real_5d.csv", parse_dates=["time_utc"], dtype={"number": str})
    w = s[s["var"] == "Q"].pivot(index="time_utc", columns="number", values="value")
    WE = le.load_q(event_path)  # Welle vom 24.-26.10., Spalten = Pegelnamen
    ev_all = pd.read_csv(event_path, parse_dates=["time_utc"], dtype={"number": str})
    lake = ev_all[ev_all["number"] == "26280366"].set_index("time_utc")["value"]
    jpf = le.fit_lag(WE["Wiersdorf"], WE["Prümzurlay"], extra=WE["Sinspelt"])
    C = cf.analyse(w_path=str(HERE / "data" / "json_w_event_2026-10.csv"), q_path=event_path, net_path=str(HERE / "pegel_netz.csv"))
    WW = C["W"]
    daily = pd.read_csv(HERE / "data" / "wasserportal" / "daily_q.csv", index_col=0, parse_dates=True)

    out_nodes = []
    for n in nodes:
        if n["id"] == "MUENDUNG":
            continue
        row = d.loc[n["id"]]
        lat, lon = utm32_to_wgs84(float(row["easting"]), float(row["northing"]))
        km = row["km_ob_muendung"]
        out_nodes.append(dict(id=n["id"], name=n["name"], river=n["river"], strang=n["strang"], art=n["art"], vars=n["vars"],
                              lat=round(lat, 5), lon=round(lon, 5), km=float(km) if km else None, area=n["area"], number=n["number"]))
    ids = {n["id"] for n in out_nodes}
    edges = [e for e in edges if e[0] in ids and e[1] in ids]
    par: dict[str, list[str]] = {}
    for a, b in edges:
        par.setdefault(b, []).append(a)

    def path(up: str, down: str) -> list[str]:
        def dfs(x, trail):
            if x == up:
                return trail + [x]
            for p in par.get(x, []):
                r = dfs(p, trail + [x])
                if r:
                    return r
            return None
        r = dfs(down, [])
        return list(reversed(r)) if r else [up, down]

    E = {r["id"]: (float(r["easting"]), float(r["northing"])) for _, r in d.reset_index().rename(columns={"lhp_id": "id"}).iterrows() if r["easting"]}
    straight = lambda a, b: math.hypot(E[a][0] - E[b][0], E[a][1] - E[b][1]) / 1000

    pairs = [("Prüm 2", "Echtershausen", None), ("Echtershausen", "Wiersdorf", "Stausee Bitburg"), ("Wiersdorf", "Prümzurlay", None),
             ("Sinspelt", "Holsthum Schloss", None), ("Holsthum Schloss", "Prümzurlay", None),
             ("Giesdorf", "Seffern", None), ("Seffern", "Alsdorf-Oberecken", None)]
    W_BASED = {("Sinspelt", "Holsthum Schloss"), ("Holsthum Schloss", "Prümzurlay")}
    kmv = {r["name"]: float(r["km_ob_muendung"]) for _, r in d.iterrows() if r["km_ob_muendung"]}
    notes = {
        ("Prüm 2", "Echtershausen"): "Prüm 2 deckt nur 16 % des Einzugsgebiets von Echtershausen ab (53 von 326 km²). Der Scheitel kommt nach 7,6 h an (6,9 h, wenn man die Mitten der flachen Scheitel nimmt), das sind 1,5 bis 1,7 m/s über 41 km und passt zu einer Welle, die die ganze Strecke läuft. Der Anstieg in Echtershausen beginnt aber viel früher (10 % des Anstiegs nach 1,4 h), weil das Zwischengebiet mit 84 % der Fläche früher reagiert. Auf dem fallenden Ast ist die Verzögerung länger (Fit 4,5 h steigend, 9 h fallend).",
        ("Echtershausen", "Wiersdorf"): "Dazwischen liegt der Stausee Bitburg (Speicher, Abgabe gesteuert). Der Scheitel wird von 18,3 auf 12,4 m³/s gedämpft (68 %) und kommt 4,5 h später, der Seestand stieg um 51 cm. Bis zum Seemaximum blieben nach der Bilanz etwa 0,32 Mio. m³ im See. Der Lag beschreibt den Speicher und seine Abgabe, keine Wellenlaufzeit (Fit: steigend 0,25 h, fallend 5,75 h). Bei sehr großen Hochwassern dämpft der Stausee laut Portal kaum (Juli 2021: Echtershausen 445, Wiersdorf 436 m³/s). Die Dämpfung hängt also stark vom Abfluss ab.",
        ("Wiersdorf", "Prümzurlay"): "Prümzurlay bekommt zusätzlich die Enz und das Zwischengebiet (576 − 340 − 102 = 134 km²). Die sauberste Messung dieser Welle ist eine Front: Am 25.10. erhöhte der Stausee die Abgabe, in Wiersdorf sprang der Stand um 21 cm in 30 min (08:15 MEZ). Die Front kam in Brecht um 09:45, in Bettingen um 11:00, in Holsthum Prüm um 12:30 und in Prümzurlay um 13:30 an: 5,25 h über 24,3 km, das sind 1,3 m/s (Teilstrecken 0,8 bis 1,6 m/s, Kilometer der Zwischenpegel geschätzt). Der Scheitel kommt nach 4,1 h an (5,1 h mit den Mitten der flachen Scheitel). Prümzurlay als Summe geschrieben (unten, Zerlegung) mit den festen Lags 5,25 h für die Prüm und 2,75 h für die Enz erreicht NSE 0,994, die freie Anpassung 0,995 (5,5 h und 2,0 h). Referenz Juni 2018 (Screenshot des Portals, abgelesen mit screenshot_lag.py): Scheitel-Lag 7,1 h bei 218 m³/s in Wiersdorf und 209 m³/s in Prümzurlay, also etwa 1,0 m/s und länger als bei der kleinen Welle jetzt. Auf dem Anstieg wächst der Versatz mit dem Niveau (0,9 h bei 10 %, 5,0 h bei 50 %, 5,6 h bei 90 % des eigenen Anstiegs). Der Scheitel nahm 2018 ab (218 → 209), 2026 und 2021 nahm er um knapp 40 % zu (12,4 → 17,4 und 436 → 596 m³/s).",
        ("Sinspelt", "Holsthum Schloss"): "Enz, nur Wasserstände: Abfluss gibt es nur in Sinspelt, die kommunalen Pegel liefern W. Holsthum Schloss liegt an der Mündung und steht ab 25.10. 12:30 MEZ im Rückstau der Prüm (Sprung von 54 auf 69 cm, gleichzeitig mit Holsthum Prüm 120 m weiter), deshalb ist nur der Anstieg bis 12:15 MEZ ausgewertet. Sinspelt → Enzen (8,9 km geschätzt): 1,25 bis 2,25 h, also 1,1 bis 2,0 m/s. Enzen → Schloss (6,3 km): 0 bis 0,5 h, zu schnell für eine Welle, die untere Enz reagiert selbst auf den Regen. Sinspelt → Schloss: 1,75 h bei NSE 0,979 (Plateau 1,0 bis 2,25 h); nimmt man auch den frühen Anstieg vom 24.10. mit, sind es 0 bis 0,75 h bei NSE 0,93. Der Enz-Scheitel in Sinspelt (3,5 m³/s) lag am 25.10. gegen 11:30 MEZ und erreichte die Mündung damit etwa um 13:15 MEZ. Die 15 h der ersten Auswertung waren dagegen der Prüm-Scheitel in Prümzurlay am 26.10. nachts, in dem der Enz-Scheitel aufgeht.",
        ("Holsthum Schloss", "Prümzurlay"): "Ab der Mündung läuft das Enz-Wasser in der Prüm. Der Stand am Schloss wird dort von der Prüm bestimmt, deshalb ist 1,0 h (NSE 0,995, steigend wie fallend) der Lag der Prüm von Holsthum bis Prümzurlay und passt zur Front (1,0 h über 4,8 km, 1,3 m/s). Die Strecke vom Schloss bis Prümzurlay ist geschätzt: Holsthum Prüm liegt etwa 100 m oberhalb der Enz-Mündung. Die Enz erreicht Prümzurlay damit etwa 2,75 h nach Sinspelt (1,0 bis 3,25 h). In Prümzurlay ist die Enz zur Zeit des Scheitels (26.10. 02:15 MEZ, 17,4 m³/s) mit etwa 2,7 m³/s beteiligt (16 %). Ihr eigener Scheitel liegt deutlich früher: 4,6 m³/s am 25.10. gegen 14 Uhr, 12 h vor dem Scheitel der Prüm. Bevor die Front der Stausee-Abgabe um 13:30 ankam, machte die Enz in Prümzurlay 45 bis 57 % des Abflusses aus. Flächencheck der Zerlegung: 1,34 × 101,5 km² (Enz an der Mündung) plus 1,17 × 340 km² (Prüm) ergibt 534 von 576 km².",
        ("Giesdorf", "Seffern"): "Der sauberste Fall: freie Strecke ohne Speicher. Scheitel nach 3,75 h (5,0 h mit den Mitten der flachen Scheitel, Seffern hat ein Plateau von über 3 h), also 1,0 bis 1,4 m/s. Fit 3,0 h steigend und 2,5 h fallend bei NSE über 0,97. Aber Giesdorf hat nur 13 % des Einzugsgebiets von Seffern (18 von 137 km²) und scheitelte mit 1,2 gegen 5,0 m³/s. Seffern wird überwiegend aus dem Zwischengebiet gespeist.",
        ("Seffern", "Alsdorf-Oberecken"): "Scheitel nach 3,25 h, mit den Mitten der flachen Scheitel nur 1,9 h. Das ergibt 2,5 bis 4,3 m/s und ist zu schnell für eine Welle aus Seffern: Alsdorf hat mit 265 km² fast doppelt so viel Einzugsgebiet wie Seffern (137 km²). Der Fit zeigt einen großen Unterschied zwischen steigendem Ast (1,5 h) und fallendem (5,0 h): Der Rückgang an Alsdorf ist langsamer als der Anstieg. Die CSV-Werte von Seffern und Alsdorf liegen im Überlappungszeitraum 4 bis 5 % unter der JSON-Schnittstelle.",
    }
    reaches = []
    for u, dn, via in pairs:
        uid, did = by_name[u], by_name[dn]
        wb = (u, dn) in W_BASED
        early = None if wb else lag_methods(w.rename(columns={v: k for k, v in nm.items()}), u, dn)
        joint = None
        if dn == "Prümzurlay" and u == "Wiersdorf":
            mine, other = ("Wiersdorf", "Sinspelt") if u == "Wiersdorf" else ("Sinspelt", "Wiersdorf")
            joint = dict(lag=r2(jpf["lag_h"] if u == "Wiersdorf" else jpf["lag2_h"]),
                         partner=other, partnerLag=r2(jpf["lag2_h"] if u == "Wiersdorf" else jpf["lag_h"]),
                         a=r2(jpf["a"]), b=r2(jpf["b"]), c=r2(jpf["c"]), nse=round(jpf["nse"], 3), first=(u == "Wiersdorf"))
        extra = {}
        if wb:
            if u == "Sinspelt":
                Wwin = WW[WW.index <= cf.ENZ_END]
                en = C["enz"][("Sinspelt", "Holsthum Schloss")]
                alt = dict(en["window_from_24"], label="Fenster ab 24.10. 12:00 MEZ")
                ev = event_block_w(Wwin, u, dn, en["window_from_25"], alt)
                km, est = 15.1, True
                i1 = int(np.argmax(WW.index > pd.Timestamp("2026-10-25 17:15", tz="UTC") - pd.Timedelta(hours=1)))
                extra = dict(xr=[0, i1], shade=int(np.argmax(WW.index > cf.ENZ_END)), shadeLabel="Rückstau der Prüm am Schloss")
                norm = {k: [float(WW[c].iloc[:8].median()), float(Wwin[c].max())] for k, c in (("up", u), ("dn", dn))}
            else:
                ev = event_block_w(WW, u, dn)
                km, est = round(C["prum_km"][3], 1), True
                extra = dict(xr=[0, len(WW) - 1])
                norm = {k: [float(WW[c].iloc[:8].median()), float(WW[c].max())] for k, c in (("up", u), ("dn", dn))}
            extra.update(unit="W", norm=norm)
            m = dict(ev=ev, early=None, **extra)
        else:
            m = dict(ev=event_block(WE, u, dn, joint), early=early)
            km, est = round(kmv[u] - kmv[dn], 2), False
        if dn == "Prümzurlay":
            m["decomp"] = True
        if (u, dn) == ("Wiersdorf", "Prümzurlay"):
            f = C["front"]
            rows = [dict(name=f[0]["name"], t=mez(f[0]["t"]), lag=None, km=None, speed=None)]
            for i in range(1, len(f)):
                dt = (f[i]["t"] - f[i - 1]["t"]).total_seconds() / 3600
                rows.append(dict(name=f[i]["name"], t=mez(f[i]["t"]), lag=r2(dt), km=r2(C["prum_km"][i - 1]),
                                 speed=r2(C["prum_km"][i - 1] * 1000 / (dt * 3600))))
            tot = (f[-1]["t"] - f[0]["t"]).total_seconds() / 3600
            m["front"] = dict(rows=rows, total=r2(tot), km=r2(sum(C["prum_km"])), speed=r2(sum(C["prum_km"]) * 1000 / (tot * 3600)),
                              rise=r2(f[0]["rise_cm"]))
        dc = None if wb else peak_day_check(daily, nm[u], nm[dn])
        label = f"{u} → {dn}"
        if dc is None and u == "Echtershausen":
            dc, label = peak_day_check(daily, "26280300", "26280504"), "Echtershausen → Prümzurlay (Wiersdorf hat keine Tagesmittel)"
        elif dc is None and u == "Wiersdorf" and not wb:
            dc, label = peak_day_check(daily, "26280300", "26280504"), "Echtershausen → Prümzurlay (Wiersdorf hat keine Tagesmittel)"
        if dc:
            dc["label"] = label
        reaches.append(dict(id=f"{uid}>{did}", up=uid, down=did, upName=u, downName=dn, path=path(uid, did), km=km, kmEst=est,
                            via=via, note=notes[(u, dn)], daily=dc, **m))
    stubs = [dict(from_=by_name["Prümzurlay"], toward=None, label="weiter zur Sauer"),
             dict(from_=by_name["Alsdorf-Oberecken"], toward=None, label="weiter zur Mündung")]
    fx = C["fixed"]
    nz = lambda s: [None if v != v else round(float(v), 2) for v in s]
    decomp = dict(L1=r2(C["L1"]), L2=r2(C["L2"]), a=r2(fx["a"]), b=r2(fx["b"]), c=r2(fx["c"]), nse=round(fx["nse"], 4),
                  free=dict(L1=r2(C["free"]["lag_h"]), L2=r2(C["free"]["lag2_h"]), nse=round(C["free"]["nse"], 4)),
                  meas=nz(fx["parts"]["meas"]), prum=nz(fx["parts"]["prum"]), enz=nz(fx["parts"]["enz"]), c_=nz(fx["parts"]["c"]))
    wcols = ["Wiersdorf", "Brecht", "Bettingen", "Holsthum Prüm", "Prümzurlay", "Sinspelt", "Enzen", "Holsthum Schloss"]
    series = dict(t0Mez=mez(WE.index[0]), stepMin=15, n=len(WE), decomp=decomp,
                  w={c: [None if v != v else int(round(v)) for v in WW[c].reindex(WE.index)] for c in wcols},
                  q={c: [round(float(v), 2) for v in WE[c]] for c in WE.columns},
                  lake=[round(float(v), 0) for v in lake.reindex(WE.index).ffill()])
    return dict(generated=datetime.now(timezone.utc).isoformat(timespec="seconds"), nodes=out_nodes, edges=edges, reaches=reaches,
                stubs=[{"from": s_["from_"], "label": s_["label"]} for s_ in stubs], series=series,
                event=dict(first=mez(WE.index[0]), last=mez(WE.index[-1]), n=len(WE)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=str(HERE / "data" / "wasserportal_event_2026-10.csv"))
    a = ap.parse_args()
    data = build(a.event)
    tpl = (HERE / "lag_map_template.html").read_text(encoding="utf-8")
    out = HERE / "gui" / "lag_map.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(out, round(out.stat().st_size / 1e3), "kB")
    for r in data["reaches"]:
        e = r["ev"]
        print(f"{r['upName']:>16} -> {r['downName']:<18} {r['km']:5.1f} km  Scheitel {e['peak']['lag']} h  Schwerpunkt {e['centroid']} h  "
              f"Fit {e['fit']['all']['lag']} h  Korr {e['corr']['level']['lag']} h  | vorläufig B {r['early']['B']['best'] if r['early'] else None}")
    for n in data["nodes"]:
        if n["name"] in ("Prüm 2", "Prümzurlay", "Hermespand", "Alsdorf-Oberecken", "Stausee Bitburg"):
            print(f"{n['name']:<18} {n['lat']:.4f} N {n['lon']:.4f} E")
