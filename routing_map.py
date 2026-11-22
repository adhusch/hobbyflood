"""Baut gui/routing_map.html: Karte des Routing-Modells (Pegel, Strecken, laufende Welle, c(Q), Güte, Vorhersage) als einzelne HTML-Datei.

  python routing_map.py [--out gui/routing_map.html]       (nach python routing_fit.py)

Liest data/routing_params.json, routing_sim_floods.json, routing_skill_floods.csv, routing_forecast_skill.csv und die Pegeldaten der Pakete.
Flusslauf: Verbindungslinien durch die Pegelpositionen (kommunale Pegel liegen im Abstand von 3 bis 6 km am Fluss); die Hintergrundkarte
(OpenStreetMap-Kacheln) ist optional zuschaltbar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import floods as F
import pkgdata
from build_lag_map import utm32_to_wgs84
from routing_fit import GROUPS

HERE = Path(__file__).parent
Q_GAUGES = ["Prüm 2", "Echtershausen", "Wiersdorf", "Prümzurlay", "Sinspelt", "Giesdorf", "Seffern", "Alsdorf-Oberecken"]
RIVERS = {
    "Prüm": ["Hermespand", "Prüm 2", "Niederprüm", "Pittenbach", "Kinzenburg", "Mauel", "Merkeshausen", "Echtershausen", "Stausee Bitburg", "Wiersdorf",
             "Brecht", "Bettingen", "Holsthum Prüm", "Prümzurlay"],
    "Mehlenbach": ["Niedermehlen", "Pittenbach"],
    "Alfbach": ["Habscheid", "Kinzenburg"],
    "Enz": ["Neurath / Arzfeld", "Neuerburg", "Sinspelt Gemeindehaus", "Sinspelt", "Enzen", "Holsthum Schloss", "Prümzurlay"],
    "Nims": ["Rommersheim", "Giesdorf", "Nimsreuland", "Seffern", "Rittersdorf", "Dockendorf", "Alsdorf-Oberecken"],
}
PATHS = {
    ("EC", "Prüm 2"): ["Prüm 2", "Niederprüm", "Pittenbach", "Kinzenburg", "Mauel", "Merkeshausen", "Echtershausen"],
    ("WI", "Echtershausen"): ["Echtershausen", "Stausee Bitburg", "Wiersdorf"],
    ("PZ", "Wiersdorf"): ["Wiersdorf", "Brecht", "Bettingen", "Holsthum Prüm", "Prümzurlay"],
    ("PZ", "Sinspelt"): ["Sinspelt", "Enzen", "Holsthum Schloss", "Prümzurlay"],
    ("SE", "Giesdorf"): ["Giesdorf", "Nimsreuland", "Seffern"],
    ("AL", "Seffern"): ["Seffern", "Rittersdorf", "Dockendorf", "Alsdorf-Oberecken"],
}


MAP_VARS = ("route_full", "route_tied", "rain_full", "rain_tied")          # Varianten, die die Karte zeigt (rain_fit.py rechnet weitere, siehe README)


def load_rain(floods: list[dict]) -> dict | None:
    """Regenblock (optional): Stationen, Güte aus der Kreuzvalidierung, Hochwasser-Simulationen mit Regen und Stationsregen je Hochwasserfenster."""
    import glob

    import rain as RN
    need = [HERE / "data" / n for n in ("rain_params.json", "rain_cv_summary.csv", "rain_forecast_skill.csv", "rain_sim_floods.json")]
    if not all(p.exists() for p in need):
        return None
    st = pd.read_csv(HERE / "data" / "dwd_rr" / "stationen.csv", dtype={"id": str})
    S = {Path(f).name.split("_")[2]: RN.read_dwd_hourly(f, shift_h=1.0) for f in glob.glob(str(HERE / "data" / "dwd_rr" / "stundenwerte_RR_*_hist.zip"))}
    rs = json.loads(need[3].read_text(encoding="utf-8"))
    for f in floods:
        if f["id"] in rs:
            f["rsim"] = {k: {v: x for v, x in g.items() if v in MAP_VARS or v == "rain_mm_h"} for k, g in rs[f["id"]].items()}
            idx = pd.date_range(pd.Timestamp(f["t0"]), periods=73, freq="1h")
            f["rain_st"] = {k: [None if pd.isna(v) else round(float(v), 1) for v in S[k].reindex(idx)] for k in S}
    params = json.loads(need[0].read_text(encoding="utf-8"))
    for g in params.values():
        g["nse_in"] = {v: x for v, x in g["nse_in"].items() if v in MAP_VARS}
        g["models"] = {v: x for v, x in g["models"].items() if v in MAP_VARS}
    summ, fcs = pd.read_csv(need[1]), pd.read_csv(need[2])
    summ, fcs = summ[summ.variant.isin(MAP_VARS)], fcs[fcs.variant.isin(MAP_VARS)]
    return dict(stations=[dict(id=r.id, name=r.name, lat=r.lat, lon=r.lon, hoehe_m=int(r.hoehe_m)) for r in st.itertuples()],
                params=params, summary=summ.round(4).to_dict("records"), fcskill=fcs.round(4).where(fcs.notna(), None).to_dict("records"))


def load_chain(floods: list[dict]) -> dict | None:
    """Kettenvorhersage (chain_fc.py): Modellsätze, Hochwasserfenster je Hochwasser (f["fcw"]), Güte je Vorlauf und Scheiteltreffer als Reihen."""
    need = [HERE / "data" / n for n in ("chain_models.json", "chain_floods.json", "chain_skill.csv", "chain_peak.csv")]
    if not all(p.exists() for p in need):
        return None
    cm = json.loads(need[0].read_text(encoding="utf-8"))
    fw = {x["id"]: x for x in json.loads(need[1].read_text(encoding="utf-8"))}
    used = set()
    for f in floods:
        if f["id"] in fw:
            f["fcw"] = fw[f["id"]]
            used.add(fw[f["id"]]["models"])
    sk, pk = pd.read_csv(need[2]), pd.read_csv(need[3])
    tag = {"cv_2004_2022": "cv", "alt_1989_2003": "alt", "cv_starkregen": "storm"}
    skill, peak = {}, {}
    for (st, t, v), df in sk.groupby(["set", "target", "variant"]):
        df = df.sort_values("lead_h")
        d = skill.setdefault(tag[st], {}).setdefault(t, {})
        d[v] = dict(skill=[round(float(x), 3) for x in df.skill], nse=[round(float(x), 3) for x in df.nse])
        d["events"] = int(df.events.iloc[0])
    for (st, t, v), df in pk.groupby(["set", "target", "variant"]):
        df = df.sort_values("before_h")
        d = peak.setdefault(tag[st], {}).setdefault(t, {})
        d[v] = dict(med=[round(float(x), 3) for x in df.ratio_med], q25=[round(float(x), 3) for x in df.ratio_q25], q75=[round(float(x), 3) for x in df.ratio_q75],
                    hit=[round(float(x), 3) for x in df.hit] if "hit" in df else None, events=int(df.events.iloc[0]))
        d["events"] = int(df.events.iloc[0])
    import rain_fit as RF2
    return dict(sets={k: v for k, v in cm["sets"].items() if k in used}, consts=cm["consts"], blocks=[list(b) for b in RF2.BLOCKS], skill=skill, peak=peak)


def load_uncert(floods: list[dict]) -> dict | None:
    """Unsicherheit (uncert.py), W-Q-Beziehung (rating.py) und Kennwerte der HVZ für den Vorhersage-Tab. Für die Karte nur die Tabellensätze der
    gezeigten Hochwasser (Regen: ohne den Block des Hochwassers; ohne Regen: aus 2004-2022), Vorläufe bis 12 h stündlich, dann alle 2 h,
    Quantile auf 0,01 gerundet (1 %)."""
    need = [HERE / "data" / n for n in ("uncert_tables.json", "rating_curves.json", "pegelkennwerte.json", "uncert_band.csv", "uncert_warn.csv",
                                        "uncert_peak.csv", "uncert_exceed.csv", "uncert_warn_events.csv", "uncert_warn_phases.csv")]
    if not all(p.exists() for p in need):
        return None
    U = json.loads(need[0].read_text(encoding="utf-8"))
    leads = np.array(U["leads_h"])
    keep = np.where((leads <= 12) | (np.isclose(leads % 2, 0)))[0]
    used = {f["fcw"]["models"] for f in floods if f.get("fcw") and f["fcw"]["rain"]}

    def slim(tab: dict) -> dict:
        return {k: dict(q=[[[round(x, 2) for x in tab[k]["q"][c][j]] for j in keep] for c in range(len(tab[k]["q"]))], peak=tab[k]["peak"],
                        n=tab[k]["n"]) for k in tab}
    out = dict(leads_h=leads[keep].tolist(), lev=U["lev"], classes=U["classes"], peak_buckets_h=U["peak_buckets_h"], peak_lev=U["peak_lev"],
               peak_rise=U["peak_rise"], peak_min_mq=U["peak_min_mq"], offset=U["offset"], mq=U["mq"],
               regen={k: slim(v) for k, v in U["regen"].items() if k in used}, kette={"alle": slim(U["kette"]["alle"])},
               vorhersage={k: slim(v) for k, v in U.get("vorhersage", {}).items() if k in used},
               floods={f["id"]: U["floods"][f["id"]] for f in floods if f.get("fcw") and f["id"] in U["floods"]}, warn=U.get("warn", {}))
    rc = json.loads(need[1].read_text(encoding="utf-8"))["pegel"]
    out["rating"] = {g: dict(q=c["q"], w=c["w"], q_meldehoehe=c["q_meldehoehe"], start=c["start"]) for g, c in rc.items()}
    out["kennwerte"] = json.loads(need[2].read_text(encoding="utf-8"))["pegel"]
    # Prüfzahlen je Pegel für „Wie verlässlich?“
    band, warn, peak, ex, wev, wph = (pd.read_csv(p) for p in need[3:])
    chk = {}
    for g in Q_GAUGES:
        b = band[band.pegel == g]
        rows = {}
        for (var, tag, kl), r in b.set_index(["variante", "pruefung", "klasse"]).iterrows():
            if kl in ("alle", "Stufe ab 8 MQ", "gemessen ab HQ2/2 (nachträglich ausgewählt)", "Zustand Welle unterwegs", "Zustand Anstieg", "Starkregenlage",
                      "HQ10-Hochwasser (48 h vor dem Überschreiten bis Scheitel, nachträglich ausgewählt)"):
                rows.setdefault(var, {}).setdefault(tag, {})[kl] = {"n": int(r.faelle), **{str(h): float(r[f"im80_{h}h"]) for h in (1, 3, 6, 12, 24, 48)}}
        w = warn[warn.pegel == g]
        p = peak[peak.pegel == g]
        # Warnschwelle (HQ10): jedes Hochwasser einzeln, Warnphasen ohne Überschreitung (Fehlalarme) gezählt, knappe (≥ 80 % der Schwelle) extra
        ev = wev[wev.pegel == g].drop(columns=["pegel"]).replace({np.nan: None}).to_dict("records")
        ph = []
        for (var, tag, lev), x in wph[wph.pegel == g].groupby(["variante", "pruefung", "stufe"]):
            fa = x[~x.treffer]
            ph.append(dict(variante=var, pruefung=tag, stufe=float(lev), phasen=len(x), fehlalarme=len(fa), knapp=int((fa.max_vs_schwelle >= 0.8).sum()),
                           ueber_halb=int((fa.max_vs_schwelle >= 0.5).sum()), liste=fa.sort_values("p_max", ascending=False).head(6)[["beginn", "p_max", "max_q_48h", "max_vs_schwelle"]].to_dict("records")))
        chk[g] = dict(band=rows, warn=w.drop(columns=["pegel"]).replace({np.nan: None}).to_dict("records"), peak=p.drop(columns=["pegel"]).to_dict("records"), wev=ev, wph=ph)
    # Zuverlässigkeit der Wahrscheinlichkeiten, über alle Pegel gepoolt (Regen, 2004-2022, Schwelle HQ2/2, 12 h)
    rel = {}
    for (var, tag, thr, w), df in ex.groupby(["variante", "pruefung", "schwelle", "fenster_h"]):
        if thr != "HQ2/2" or w not in (6, 24):
            continue
        tot = {}
        for z in df.zuverl:
            for k, (fr, n) in json.loads(z).items():
                if fr is None:
                    continue
                a = tot.setdefault(k, [0.0, 0])
                a[0] += fr * n
                a[1] += n
        rel[f"{var}|{tag}|{w}"] = {k: [round(v[0] / v[1], 3), v[1]] for k, v in tot.items()}
    out["check"] = chk
    out["reliability"] = rel
    return out


def build(out: Path) -> None:
    net = pd.read_csv(HERE / "pegel_netz.csv", dtype=str).fillna("")
    nodes = []
    for _, r in net.iterrows():
        if not r["easting"]:
            continue
        lat, lon = utm32_to_wgs84(float(r["easting"]), float(r["northing"]))
        nodes.append(dict(name=r["name"], river=r["gewaesser"], art=r["art"], vars=r["messgroessen"], lat=round(lat, 5), lon=round(lon, 5),
                          area=float(r["catchment_km2"]) if r["catchment_km2"] else None, q=r["name"] in Q_GAUGES))
    names = {n["name"] for n in nodes}
    for lst in list(RIVERS.values()) + list(PATHS.values()):
        assert all(x in names for x in lst), [x for x in lst if x not in names]
    params = json.loads((HERE / "data" / "routing_params.json").read_text(encoding="utf-8"))
    sims = json.loads((HERE / "data" / "routing_sim_floods.json").read_text(encoding="utf-8"))
    skill = pd.read_csv(HERE / "data" / "routing_skill_floods.csv")
    Q = pkgdata.load_all("Q")
    fl_all = F.find_floods(pkgdata.load_all("W")["Prümzurlay"], 400)
    floods = []
    for fl in fl_all:
        fid = f"{fl['t_peak']:%Y-%m-%d}"
        if fid not in sims:
            continue
        win = F.window(Q, fl)
        s = sims[fid]
        floods.append(dict(id=fid, t_peak=s["t_peak"], t0=f"{win.index[0]:%Y-%m-%d %H:%M}", clipped=s["clipped"],
                           q_pz=round(float(win["Prümzurlay"].max()), 1),
                           obs={g: [round(float(x), 2 if win[g].max() < 10 else 1) for x in win[g]] for g in Q_GAUGES},
                           sim=s["series"]))
    reaches = []
    for g in GROUPS:
        for k, (up, L) in enumerate(g.links):
            reaches.append(dict(id=f"{g.key}:{up}", group=g.key, link=k, up=up, dn=g.dn, length_km=L, path=PATHS[(g.key, up)], lake=g.key == "WI"))
    rain = load_rain(floods)
    chain = load_chain(floods)
    unc = load_uncert(floods)
    Wd = pkgdata.load_all("W")
    for f, fl in zip(floods, [x for x in fl_all if f"{x['t_peak']:%Y-%m-%d}" in sims]):
        win = F.window(Wd, fl)                                      # damals gemessener Wasserstand (Scheitel im 72-h-Fenster), alte W-Q-Beziehung
        f["w_peak"] = {g: None if not np.isfinite(win[g].max()) else int(round(float(win[g].max()))) for g in Q_GAUGES}
        fw = f.get("fcw")
        if fw and unc:                                              # Beziehung Wasserstand-Abfluss, wie sie beim Hochwasser galt (wie eval_view.py)
            from eval_view import event_rating
            i0 = int(Q.index.get_loc(pd.Timestamp(fw["t_start"])))
            sl = slice(i0, i0 + len(next(iter(fw["q"].values()))))
            fw["rat"] = {}
            for dn in fw["q"]:
                r = event_rating(Q[dn].to_numpy()[sl], Wd[dn].to_numpy()[sl], unc["rating"][dn])
                if r:
                    fw["rat"][dn] = r
    data = dict(nodes=nodes, rivers=RIVERS, reaches=reaches, groups=params, floods=floods, rain=rain, chain=chain, unc=unc,
                skill=skill.round(3).where(skill.notna(), None).to_dict("records"))
    tpl = (HERE / "routing_map_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("/*__FC_CORE__*/", (HERE / "forecast_core.js").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "gui" / "routing_map.html"))
    build(Path(ap.parse_args().out))
