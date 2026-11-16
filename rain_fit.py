"""Routing mit Regen anpassen und prüfen (DWD-Stundenwerte RR, 9 Stationen, data/dwd_rr).

  python rain_fit.py shift            Zeitversatz des Regens gegen die Pegelzeit (MEZ) über den Zielwert wählen
  python rain_fit.py run [--jobs 2]   Anpassung je Gruppe (Routing allein / mit Regen, c(Q) und festes c), blockweise Kreuzvalidierung,
                                      Hochwasser 2011, 2018, 2021 einzeln, Vorhersagebetrieb. Ausgaben data/rain_*.csv/json

Zeitbasis: Die Pegelreihen stehen in fester MEZ (UTC+1), die DWD-Stundenwerte in UTC; Stundenwert gilt für die Stunde vor dem Stempel (Intervallende)
-- das ist die Annahme, die `shift` prüft (Versatz 0, 1, 2 h gegen UTC; erwartet 1 = reine Umrechnung UTC nach MEZ).
Zwischengebiete (Fläche aus pegel_netz.csv) und Regengewichte: Abstandsgewichte (Potenz 2) der Stationen zum Mittelpunkt der Pegel, die das Gebiet
umschließen. Wiersdorf (Seeumfeld, 14 km²) bekommt keinen Regenanteil.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import lateral as L
import pkgdata
import rain as RN
import routing as R
import routing_fit as RF
from build_lag_map import utm32_to_wgs84

HERE = Path(__file__).parent
DT = R.DT_H


@dataclass
class Area:
    key: str                 # Gruppe
    nodes: list[str]         # Pegel, deren Mittelpunkt das Gebiet vertritt
    parts: tuple[str, str, tuple]   # Fläche = A(dn) - A(oberlieger) ...


def net() -> pd.DataFrame:
    n = pd.read_csv(HERE / "pegel_netz.csv", dtype=str).fillna("")
    n = n[n["easting"] != ""].copy()
    ll = [utm32_to_wgs84(float(e), float(nn)) for e, nn in zip(n["easting"], n["northing"])]
    n["lat"], n["lon"] = [x[0] for x in ll], [x[1] for x in ll]
    n["A"] = pd.to_numeric(n["catchment_km2"], errors="coerce")
    return n.set_index("name")


AREAS = {
    "EC": dict(dn="Echtershausen", minus=["Prüm 2"], nodes=["Prüm 2", "Niederprüm", "Pittenbach", "Kinzenburg", "Mauel", "Merkeshausen", "Echtershausen", "Niedermehlen", "Habscheid"]),
    "PZ": dict(dn="Prümzurlay", minus=["Wiersdorf", "Sinspelt"], nodes=["Wiersdorf", "Brecht", "Bettingen", "Holsthum Prüm", "Prümzurlay", "Sinspelt", "Enzen", "Holsthum Schloss"]),
    "SE": dict(dn="Seffern", minus=["Giesdorf"], nodes=["Giesdorf", "Nimsreuland", "Seffern", "Rommersheim"]),
    "AL": dict(dn="Alsdorf-Oberecken", minus=["Seffern"], nodes=["Seffern", "Rittersdorf", "Dockendorf", "Alsdorf-Oberecken"]),
}


def lateral_defs(N: pd.DataFrame) -> dict[str, tuple[float, tuple[float, float]]]:
    out = {}
    for k, a in AREAS.items():
        area = float(N.loc[a["dn"], "A"] - sum(N.loc[m, "A"] for m in a["minus"]))
        out[k] = (area, (float(N.loc[a["nodes"], "lat"].mean()), float(N.loc[a["nodes"], "lon"].mean())))
    return out


def stations() -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    st = pd.read_csv(HERE / "data" / "dwd_rr" / "stationen.csv", dtype={"id": str})
    return {r.id: (r.lat, r.lon) for r in st.itertuples()}, {r.id: r.name for r in st.itertuples()}


def load_stations(shift_h: float) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / "dwd_rr" / "stundenwerte_RR_*_hist.zip"))):
        out[Path(f).name.split("_")[2]] = RN.read_dwd_hourly(f, shift_h=shift_h)
    return out


RADAR_FILL = os.environ.get("PRUEM_RADAR_FILL", "1") != "0"     # vor 2004 (kein Stundenregen der Stationen) angeglichenes Radar als Gebietsregen
RADAR_FILL_BIS = "2004-01-01"
EXCESS_F = (4.0, 10.0, 20.0)                            # Schwellen des Starkregenanteils in mm/h (Spalten x4, x10, x20; aus radar.EXCESS_MM_H)
XCOLS = ("xr",) + tuple(f"x{f:g}" for f in EXCESS_F)    # Radar-Zusatzgrößen je Gebiet: Gebietsmittel und Starkregenanteile (lateral "ml")
RAIN_SOURCE = os.environ.get("PRUEM_RAIN", "station")   # Regenmenge p: "station" (Standard) oder "radar" (RADKLIM-Gebietsmittel, Lücken mit Stationsregen).
# Versuch 14.11.2026: Radar-Gebietsmittel statt Stationsregen war schlechter (7-12 % weniger Regen, HQ10-Scheitel 41 statt 56 %), siehe README.


_MERGE = {}


def _merge() -> pd.DataFrame:
    if "df" not in _MERGE:
        _MERGE["df"] = pd.read_csv(HERE / "data" / "radar" / "areal_merge.csv", index_col=0, parse_dates=True)
    return _MERGE["df"]


def radar_hourly() -> pd.DataFrame | None:
    p = HERE / "data" / "radar" / "areal_hourly.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None


def radar_excess() -> pd.DataFrame | None:
    """Starkregenanteile je Gebiet (mm/h über den Schwellen EXCESS_F, Flächenmittel über die Radarpixel), Spalten <Gebiet>_x<f>."""
    p = HERE / "data" / "radar" / "areal_excess.csv"
    if not p.exists():
        return None
    head = pd.read_csv(p, nrows=0).columns
    cols = [c for c in head if any(c.endswith(f"_x{f:g}") for f in EXCESS_F)]
    return pd.read_csv(p, index_col=0, parse_dates=True, usecols=[head[0]] + cols)


def rain_frames(Qindex: pd.DatetimeIndex, shift_h: float = 1.0, extra: dict | None = None, source: str | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Gebietsregen und Vorregenindex je Gruppe auf dem 15-min-Raster der Pegel. extra: weitere Gebiete, Schlüssel -> dict(dn=Pegel, nodes=[Pegel für
    den Mittelpunkt]); Fläche = Einzugsgebiet des Pegels (Oberläufe ohne Oberlieger, siehe chain_fc.py).
    source "radar": Stundenwerte aus dem Radar-Gebietsmittel (radar.py, echte Einzugsgebietsflächen); wo das Radar fehlt, der Stationsregen."""
    source = source or RAIN_SOURCE
    radar = radar_hourly()
    if source == "radar" and radar is None:
        raise FileNotFoundError("data/radar/areal_hourly.csv fehlt (radar.py) - oder PRUEM_RAIN=station setzen")
    exc = radar_excess() if os.environ.get("PRUEM_RADAR_X", "0") != "0" else None      # Radar-Zusatzgrößen (nur für "ml"; 0 spart Speicher)
    N = net()
    defs = lateral_defs(N)
    for k, a in (extra or {}).items():
        defs[k] = (float(N.loc[a["dn"], "A"]), (float(N.loc[a["nodes"], "lat"].mean()), float(N.loc[a["nodes"], "lon"].mean())))
    xy, _ = stations()
    S = load_stations(shift_h)
    full = pd.date_range(min(s.index.min() for s in S.values()), max(s.index.max() for s in S.values()), freq="1h")
    S = {k: v.reindex(full) for k, v in S.items()}
    frames, info = {}, {}
    for k, (area, ctr) in defs.items():
        w = RN.idw_weights({s: xy[s] for s in S}, ctr)
        h = RN.areal_series(S, w)
        info[k] = dict(area_km2=area, centroid=ctr, weights={s: round(v, 3) for s, v in w.items() if v > 0.01})
        hst = h
        if radar is not None and source == "radar":
            r = radar[k]
            idx = pd.date_range(min(h.index.min(), r.index.min()), max(h.index.max(), r.index.max()), freq="1h")
            r, hs = r.reindex(idx), h.reindex(idx)
            h = r.where(r.notna(), hs)
            both = r.notna() | hs.notna()
            info[k]["radar_share"] = round(float(r.notna()[both].mean()), 4)
        if source == "tag":                                  # Tagessummen der Stationen mit Stundenverlauf aus ERA5 (day_rain.py), ganzer Zeitraum
            h = pd.read_csv(HERE / "data" / "tag" / "areal_tagera.csv", index_col=0, parse_dates=True)[k]
        if source == "merge":                                # Menge von den Stationen, Muster vom Radar (grid_rain.py); wo es fehlt, der Stationsregen
            mg = _merge()[k]
            idx = pd.date_range(min(h.index.min(), mg.index.min()), max(h.index.max(), mg.index.max()), freq="1h")
            mg, hs = mg.reindex(idx), h.reindex(idx)
            h = mg.where(mg.notna(), hs)
            info[k]["merge_share"] = round(float(mg.notna()[mg.notna() | hs.notna()].mean()), 4)
        if radar is not None and source == "station" and RADAR_FILL:
            # Vor Beginn der Stationsdaten (2004) das Radar-Gebietsmittel, angeglichen mit dem Verhältnis Station / Radar des Gebiets aus den gemeinsamen
            # Stunden ab 2004 (das Radar liegt 7 bis 12 % tiefer); Lücken im Radar bis 6 h linear gefüllt, längere bleiben leer
            r = radar[k]
            cut = pd.Timestamp(RADAR_FILL_BIS)
            hh = h[h.index >= cut]
            rc = r.reindex(hh.index)
            both = hh.notna() & rc.notna()
            fac = float(hh[both].sum() / rc[both].sum())
            idx = pd.date_range(r.index.min(), cut - pd.Timedelta(hours=1), freq="1h")
            rr = r.reindex(idx).interpolate(limit=6, limit_area="inside") * fac
            h = pd.concat([rr, hh])
            info[k]["radar_vor_2004"] = dict(faktor=round(fac, 3), ab=str(idx[0]), fehlend=round(float(rr.isna().mean()), 4))
        frames[k] = RN.areal_frame(h, Qindex)
        if radar is not None and exc is not None:             # Radar-Zusatzgrößen; ohne Radar aus dem Stationsmittel (unterschätzt Schauerzellen)
            hr = pd.date_range(hst.index.min(), hst.index.max(), freq="1h")
            st = hst.reindex(hr)
            stamp = Qindex.ceil("1h")
            xr = radar[k].reindex(hr)
            frames[k]["xr"] = xr.where(xr.notna(), st).reindex(stamp).to_numpy()
            for f in EXCESS_F:
                x = exc[f"{k}_x{f:g}"].reindex(hr)
                frames[k][f"x{f:g}"] = x.where(x.notna(), (st - f).clip(lower=0.0)).reindex(stamp).to_numpy()
    return frames, info


def lat_start(info: dict, key: str, kind: int = 0) -> L.Lateral:
    starts = [(0.4, 40.0, 3.0, 24.0), (0.2, 80.0, 1.0, 8.0)]
    p, s, k1, k2 = starts[kind]
    return L.Lateral(key, info[key]["area_km2"], p, float(np.log(s)), float(np.log(k1)), float(np.log(k2)))


def best_fit(g: RF.Group, d: R.Data, Q: pd.DataFrame, info: dict, tied: bool, **kw):
    cands = [R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, tied=tied, cmax=g.cmax, lat=[lat_start(info, g.key, i)], **kw) for i in (0, 1)]
    return min(cands, key=lambda z: z[1])


def shift_test() -> None:
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    for sh in (0.0, 1.0, 2.0):
        frames, info = rain_frames(Q.index, sh)
        g = RF.G["EC"]
        d = R.make_data(Q, W, g.dn, RF.ups(g), rain={"EC": frames["EC"]})
        t0 = time.time()
        m, cost, dx0 = R.fit(g.dn, g.links, d, Q, dx_grid=(20, 8, 3), tied=True, cmax=g.cmax, lat=[lat_start(info, "EC", 0)], max_nfev=40)
        lt = m.lat[0]
        nse = R.nse(d.Y[d.mask], m.route(d.X, d.P)[d.mask])
        print(f"Versatz {sh:+.0f} h: Ereignisse {len(d.peaks)}, Zielwert {cost:.3f}, NSE {nse:.3f}, phi_max {lt.phi_max:.2f} S {np.exp(lt.log_s):.0f} "
              f"K1 {np.exp(lt.log_k1):.1f} K2 {np.exp(lt.log_k2):.1f} ({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ Anpassung und Prüfung
VARS = ("route_full", "route_tied", "rain_full", "rain_tied", "int_full", "acc_full", "both_full")
# Erweiterungen des Abflussbeiwerts (lateral.py): Intensität, Ereignisfeuchte, beides; nur mit c(Q) (Variante "full")
EXT = {"int_full": ("int",), "acc_full": ("acc",), "both_full": ("int", "acc")}
BLOCKS = [(2004, 2007), (2008, 2011), (2012, 2015), (2016, 2018), (2019, 2022)]
KEYS = ("EC", "PZ", "SE", "AL")
FC_OFF = np.arange(-24, 1, 2)          # Ausgabezeiten relativ zum Scheitel des Unterliegers (h)
FC_A, FC_H, FC_CORR = 48, 12, 6.0      # Vorgeschichte (h), Horizont (h), Abklingzeit der Fehlerkorrektur (h)
_S: dict = {}


def with_lat(m: R.Model, lat) -> R.Model:
    return R.Model(m.dn, m.links, m.c0, m.tied, m.cmax, list(lat), m.amax)


def fit_variants(g: RF.Group, d: R.Data, Q, info, start: dict | None, max_nfev: int = 40) -> dict[str, R.Model]:
    """start: Modelle je Variante als Warmstart (route_* aus der Routing-Anpassung, rain_* aus der Gesamtanpassung der Gruppe)."""
    out = {}
    for tied in (False, True):
        tag = "tied" if tied else "full"
        m0, cost0, _ = R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, tied=tied, start=start["route_" + tag], cmax=g.cmax, max_nfev=max_nfev)
        out["route_" + tag] = m0
        if start.get("rain_" + tag) is not None:
            m1, _, _ = R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, tied=tied, start=start["rain_" + tag], cmax=g.cmax, max_nfev=max_nfev)
        else:
            cands = [R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, tied=tied, start=with_lat(m0, [lat_start(info, g.key, i)]), cmax=g.cmax,
                           max_nfev=max_nfev) for i in (0, 1)]
            m1 = min(cands, key=lambda z: z[1])[0]
        out["rain_" + tag] = m1
    for v, terms in EXT.items():
        st = start.get(v)
        if st is None:
            st = with_lat(out["rain_full"], [out["rain_full"].lat[0].with_terms(terms)])
        out[v] = R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, tied=False, start=st, cmax=g.cmax, max_nfev=max_nfev)[0]
    return out


def event_rows(d: R.Data, m: R.Model, spin: int) -> list[dict]:
    sim = m.route(d.X, d.P)
    rows = []
    for i in range(len(d.peaks)):
        o, s = d.Y[i, spin:], sim[i, spin:]
        pk = float(o.max())
        rows.append(dict(nse=R.nse(o, s), peak_obs=pk, peak_sim=float(s.max()), peak_err_pct=100 * (float(s.max()) - pk) / pk,
                         peak_dt_h=(int(s.argmax()) - int(o.argmax())) * DT, sse=float((((o - s) / pk) ** 2).sum()),
                         sst=float((((o - o.mean()) / pk) ** 2).sum())))
    return rows


def forecast_events(m: R.Model, g: RF.Group, Q, frame, d: R.Data) -> dict:
    """Vorhersage je Ausgabezeit und Ereignis: Oberlieger bis t0 beobachtet, danach konstant; Regen bis t0 beobachtet, danach 0 (kein Vorhersageregen);
    Fehlerkorrektur am Unterlieger. Rückgabe normierte Vorhersage, Beobachtung und Persistenz [Fälle, Schritte]."""
    a, H = int(FC_A / DT), int(FC_H / DT)
    t0 = (d.peaks[:, None] + (FC_OFF * int(1 / DT))[None, :]).ravel()
    pk = np.repeat(d.Y.max(axis=1), len(FC_OFF))
    idx = t0[:, None] + np.arange(-a, H + 1)[None, :]
    X = {}
    for u in RF.ups(g):
        v = Q[u].to_numpy()[idx].copy()
        v[:, a + 1:] = v[:, a:a + 1]
        X[u] = v
    P = None
    if m.lat:
        lead = int(round(72 / DT))
        ridx = t0[:, None] + np.arange(-a - lead, H + 1)[None, :]
        r = frame["p"].to_numpy()[ridx].copy()
        ap = frame["api"].to_numpy()[ridx].copy()
        aps = frame["api_s"].to_numpy()[ridx].copy()
        r[:, lead + a + 1:] = 0.0
        k = np.arange(1, H + 1)
        ap[:, lead + a + 1:] = ap[:, lead + a:lead + a + 1] * np.exp(-k * DT / RN.TAU_API_H)[None, :]
        aps[:, lead + a + 1:] = aps[:, lead + a:lead + a + 1] * np.exp(-k * DT / RN.TAU_API_S_H)[None, :]
        P = {m.lat[0].name: L.Rain(np.nan_to_num(r), np.nan_to_num(ap), lead, np.nan_to_num(aps))}
    y = m.route(X, P)
    obs = Q[g.dn].to_numpy()[t0[:, None] + np.arange(1, H + 1)[None, :]]
    err0 = Q[g.dn].to_numpy()[t0] - y[:, a]
    lead_h = (np.arange(1, H + 1) * DT)[None, :]
    corr = np.maximum(y[:, a + 1:] + err0[:, None] * np.exp(-lead_h / FC_CORR), 0)
    pers = np.repeat(Q[g.dn].to_numpy()[t0][:, None], H, axis=1)
    return dict(obs=obs / pk[:, None], corr=corr / pk[:, None], pers=pers / pk[:, None])


def glob_fit(key: str) -> dict:
    Q, W, frames, info, base = _S["Q"], _S["W"], _S["frames"], _S["info"], _S["base"]
    g = RF.G[key]
    t = time.time()
    d = R.make_data(Q, W, g.dn, RF.ups(g), rain={key: frames[key]})
    start = {"route_full": R.Model.from_dict(base[key]["full"]), "route_tied": R.Model.from_dict(base[key]["tied"]), "rain_full": None, "rain_tied": None}
    start.update({v: None for v in EXT})
    for v in start:
        if v.startswith("route"):
            start[v].lat = []
    ms = fit_variants(g, d, Q, info, start)
    spin = int(12 / DT)
    ins = {v: R.nse(d.Y[d.mask], m.route(d.X, d.P)[d.mask]) for v, m in ms.items()}
    print(f"[global] {key} {g.dn}: {len(d.peaks)} Ereignisse, NSE in-sample " + "  ".join(f"{v} {x:.3f}" for v, x in ins.items()) + f" ({time.time() - t:.0f} s)", flush=True)
    return dict(key=key, models=ms, nse_in=ins, n=len(d.peaks), times=[str(x)[:16] for x in d.times])


def cv_fold(args) -> dict:
    key, b = args
    Q, W, frames, info, GL = _S["Q"], _S["W"], _S["frames"], _S["info"], _S["GL"]
    g = RF.G[key]
    t = time.time()
    d_all = R.make_data(Q, W, g.dn, RF.ups(g), rain={key: frames[key]})
    yrs = pd.DatetimeIndex(d_all.times).year.to_numpy()
    test = (yrs >= BLOCKS[b][0]) & (yrs <= BLOCKS[b][1])
    if not test.any():
        return dict(key=key, block=b, rows=[], fc={})
    tp = d_all.peaks[test]
    near = (np.abs(d_all.peaks[:, None] - tp[None, :]) <= int(3.5 * 24 / DT)).any(axis=1)
    dtr, dte = d_all.subset(~test & ~near), d_all.subset(test)
    start = dict(GL[key]["models"])
    ms = fit_variants(g, dtr, Q, info, start, max_nfev=25)
    spin = int(12 / DT)
    rows, fcs = [], {}
    for v, m in ms.items():
        for i, r in enumerate(event_rows(dte, m, spin)):
            rows.append(dict(group=key, dn=g.dn, variant=v, block=b, time=str(dte.times[i])[:16], **r))
        fcs[v] = forecast_events(m, g, Q, frames[key], dte)
    print(f"[CV] {key} Block {BLOCKS[b][0]}-{BLOCKS[b][1]}: Training {len(dtr.peaks)}, Test {len(dte.peaks)} ({time.time() - t:.0f} s)", flush=True)
    return dict(key=key, block=b, rows=rows, fc=fcs, models={v: m.to_dict() for v, m in ms.items()})


def flood_sims(res: list[dict], Q, W, frames) -> None:
    """Simulation der Hochwasser mit Regenabdeckung (PZ-Scheitel +-36 h) mit den Modellen des Blocks, in dem das Hochwasser liegt (nicht im Training)."""
    import floods as F
    out = {}
    a, b, lead = int(48 / DT), int(36 / DT), int(round(72 / DT))
    for fl in F.find_floods(W["Prümzurlay"], 400):
        yr = fl["t_peak"].year
        blk = [i for i, (y0, y1) in enumerate(BLOCKS) if y0 <= yr <= y1]
        ipz = int(Q.index.get_loc(fl["t_peak"]))
        if not blk or not frames["EC"]["p"].iloc[ipz - a - lead: ipz + b + 1].notna().all():
            continue
        fid = f"{fl['t_peak']:%Y-%m-%d}"
        out[fid] = {}
        for k in KEYS:
            x = next((r for r in res if r["key"] == k and r["block"] == blk[0] and r.get("models")), None)
            if x is None:
                continue
            g = RF.G[k]
            idx = ipz + np.arange(-a, b + 1)
            X = {u: Q[u].to_numpy()[idx][None, :] for u in RF.ups(g)}
            ridx = ipz + np.arange(-a - lead, b + 1)
            P = {k: L.Rain(np.nan_to_num(frames[k]["p"].to_numpy()[ridx])[None, :], np.nan_to_num(frames[k]["api"].to_numpy()[ridx])[None, :], lead,
                           np.nan_to_num(frames[k]["api_s"].to_numpy()[ridx])[None, :])}
            sl = slice(a - int(36 / DT), None)
            d = {}
            for v, md in x["models"].items():
                m = R.Model.from_dict(md)
                d[v] = [round(float(z), 2) for z in m.route(X, P)[0][sl]]
            d["rain_mm_h"] = [round(float(z), 2) for z in frames[k]["p"].to_numpy()[ridx][lead:][sl][::4]]
            out[fid][k] = d
    (HERE / "data" / "rain_sim_floods.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def run(jobs: int) -> None:
    import multiprocessing as mp
    t_all = time.time()
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    frames, info = rain_frames(Q.index, 1.0)
    base = json.loads((HERE / "data" / "routing_params.json").read_text(encoding="utf-8"))
    _S.update(Q=Q, W=W, frames=frames, info=info, base=base)
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        G = {r["key"]: r for r in pool.map(glob_fit, KEYS)}
        _S["GL"] = G
    with ctx.Pool(jobs) as pool:
        res = list(pool.imap_unordered(cv_fold, [(k, b) for b in range(len(BLOCKS)) for k in KEYS]))
    ev = pd.DataFrame([r for x in res for r in x["rows"]])
    ev.to_csv(HERE / "data" / "rain_cv_events.csv", index=False, float_format="%.4g")
    # Zusammenfassung je Gruppe und Variante
    summ = []
    for (k, v), df in ev.groupby(["group", "variant"]):
        summ.append(dict(group=k, variant=v, n=len(df), nse_pooled=1 - df.sse.sum() / df.sst.sum(), nse_median=df.nse.median(),
                         peak_err_med_abs_pct=df.peak_err_pct.abs().median(), peak_dt_med_abs_h=df.peak_dt_h.abs().median()))
    pd.DataFrame(summ).to_csv(HERE / "data" / "rain_cv_summary.csv", index=False, float_format="%.4g")
    # Vorhersagegüte gepoolt
    rows = []
    for k in KEYS:
        parts = [x for x in res if x["key"] == k and x["fc"]]
        if not parts:
            continue
        O = np.concatenate([x["fc"]["route_full"]["obs"] for x in parts])
        Pe = np.concatenate([x["fc"]["route_full"]["pers"] for x in parts])
        mse_p = ((Pe - O) ** 2).mean(axis=0)
        for v in VARS:
            Pv = np.concatenate([x["fc"][v]["corr"] for x in parts])
            mse = ((Pv - O) ** 2).mean(axis=0)
            for h in range(O.shape[1]):
                rows.append(dict(group=k, dn=RF.G[k].dn, variant=v, lead_h=(h + 1) * DT, nrmse=float(np.sqrt(mse[h])),
                                 skill_vs_persistence=float(1 - mse[h] / mse_p[h]), n=int(len(O))))
    pd.DataFrame(rows).to_csv(HERE / "data" / "rain_forecast_skill.csv", index=False, float_format="%.4g")
    out = {}
    for k in KEYS:
        out[k] = dict(dn=RF.G[k].dn, **info[k], n_events=G[k]["n"], nse_in=G[k]["nse_in"],
                      models={v: m.to_dict() for v, m in G[k]["models"].items()})
    (HERE / "data" / "rain_params.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    flood_sims(res, Q, W, frames)
    print(f"fertig in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shift", "run", "info"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "shift":
        shift_test()
    elif a.cmd == "run":
        run(a.jobs)
    elif a.cmd == "info":
        Q = pkgdata.load_all("Q")
        frames, info = rain_frames(Q.index, 1.0)
        for k, v in info.items():
            f = frames[k]
            print(k, v, "Regen gültig:", f"{f['p'].notna().mean():.1%}", "API gültig:", f"{f['api'].notna().mean():.1%}")
