"""Live-Vorhersage für Prüm, Enz und Nims mit Wettervorhersage (ICON-D2 und ICON-D2-EPS) und Kette über alle acht Pegel.

  python live.py                 Daten holen (Internet) und Seite bauen: gui/live.html
  python live.py fetch           nur holen, nach data/live/raw_<Zeit>/
  python live.py build [ordner]  nur bauen, aus dem jüngsten (oder angegebenen) Rohdatenordner, ohne Internet

Quellen (alle frei zugänglich; Abrufe sparsam halten, Nutzungsbedingungen der Anbieter beachten):
  Pegel        hochwasser.rlp.de/api/v1/measurement-site/<nummer>: die letzten 5 Tage W und Q in 15 min, für Prümzurlay und
               Alsdorf-Oberecken auch die amtliche Vorhersage der HVZ (p10 bis p90), hier nur zum Vergleich gezeigt.
  Regen        DWD Open Data: Stundenwerte RR der Stationen (recent, bis gestern) und 10-Minuten-Werte (now, heute).
  Vorhersage   Open-Meteo: DWD ICON-D2 (2 km, 15 min, 48 h, alle 3 h) und ICON-D2-EPS (20 Läufe, stündlich, 48 h) an den Stationsorten.

Rechnung:
  - Gebietsregen wie bei der Anpassung (Stationsgewichte je Gebiet, je Stunde auf die vorhandenen Stationen renormiert). ICON wird an denselben
    Stationsorten gelesen und mit dem Faktor Station / ICON je Station (verif_rain.py, 2024-2026) angeglichen, dann genauso gewichtet.
    Stunden zwischen der letzten Stationsmeldung und jetzt: ICON-D2 der jüngsten Läufe.
  - Kette (chain_fc, Modellsatz aus allen Ereignissen 2004-2022, Variante kette_vorhersage: Regen nach jetzt aus der Vorhersage, Umschalten
    Winter / Starkregen; die Scheitel-Regel gilt nur noch in der Vergleichsrechnung ohne Regenvorhersage) für den ICON-D2-Hauptlauf und jeden der 20 Ensemble-Läufe.
  - Unsicherheit: Mischung aus den 21 Läufen (gleich gewichtet) und dem Restfehler des Abflussmodells bei bekanntem Regen (uncert.py,
    Variante kette_vorhersage mit gemessenem Regen, Klasse aus Zustand und Abflussstufe je Lauf). Band = Quantile der Mischung; Überschreitung
    = Mittel über die Läufe der Wahrscheinlichkeit, die Schwelle bis dahin zu erreichen; Höchststand in 48 h aus gewichteten Pfaden.
  Zum Vergleich wird die Kette ohne Regenvorhersage (Regen nach jetzt = 0, wie bisher) mitgerechnet.
Zeiten: intern MEZ ohne Sommerzeit (wie die LfU-Daten der Anpassung), Anzeige MEZ.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import chain_fc as C  # noqa: E402
import rain as RN  # noqa: E402
import rain_fit as RF2  # noqa: E402
import rain_thr as RTH  # noqa: E402
import rating as RT  # noqa: E402
import uncert as UN  # noqa: E402

DT = C.DT
NUM = {"P2": "26280050", "EC": "26280300", "WI": "26280402", "PZ": "26280504", "SI": "26280650", "GI": "26280708", "SE": "26280800", "AL": "26280854"}
MEZ = pd.Timedelta(hours=1)
HISTORY_D = 40                                                   # Regen-Vorlauf für den Vorregenindex (7 Tage Zeitkonstante, gültig ab 7 Tagen)
UA = "pruem-hochwasser-live/1.0 (privat, Forschung)"


# ------------------------------------------------------------------ Abruf
def _get(url: str, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:                                    # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f"Abruf fehlgeschlagen: {url}: {e}") from e
            time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def station_table() -> pd.DataFrame:
    return pd.read_csv(HERE / "data" / "dwd_rr" / "stationen.csv", dtype={"id": str})


def icon_urls(st: pd.DataFrame) -> dict[str, str]:
    lat, lon = ",".join(f"{x:.4f}" for x in st.lat), ",".join(f"{x:.4f}" for x in st.lon)
    return {
        "icon_d2": f"https://api.open-meteo.com/v1/dwd-icon?latitude={lat}&longitude={lon}&minutely_15=precipitation&hourly=precipitation"
                   f"&models=icon_d2&past_days=3&forecast_days=3&timezone=GMT",
        "icon_d2_eps": f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&hourly=precipitation&models=icon_d2_eps"
                       f"&past_days=1&forecast_days=3&timezone=GMT",
        # Ausblick 7 Tage: nach dem Ende von ICON-D2 (48 h) ICON-EU (7 km, bis 120 h), danach ICON global (13 km, bis 180 h)
        "icon_eu": f"https://api.open-meteo.com/v1/dwd-icon?latitude={lat}&longitude={lon}&hourly=precipitation&models=icon_eu"
                   f"&past_days=1&forecast_days=8&timezone=GMT",
        "icon_global": f"https://api.open-meteo.com/v1/dwd-icon?latitude={lat}&longitude={lon}&hourly=precipitation&models=icon_global"
                       f"&past_days=1&forecast_days=8&timezone=GMT",
        "meta_d2": "https://api.open-meteo.com/data/dwd_icon_d2/static/meta.json",
        "meta_d2_eps": "https://ensemble-api.open-meteo.com/data/dwd_icon_d2_eps/static/meta.json",
    }


NB_KM = 30.0                                                     # Versatz der Nachbarschaftsläufe
_D = 0.7071
NB_DIRS = {"N": (0, 1), "NO": (_D, _D), "O": (1, 0), "SO": (_D, -_D), "S": (0, -1), "SW": (-_D, -_D), "W": (-1, 0), "NW": (-_D, _D)}   # (Ost, Nord)
RUC_MAX_H = 30.0                                                 # längste glaubhafte Reichweite von ICON-D2-RUC ab jetzt (Läufe bis 27 h)
RUC_MODELS = ("dwd_icon_d2_ruc", "icon_d2_ruc", "dwd_icon_d2_ruc_native")


def fetch() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = HERE / "data" / "live" / f"raw_{stamp}"
    (out / "dwd").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    hvz = dict(abgerufen_utc=now, quelle="https://www.hochwasser.rlp.de/api/v1/measurement-site/<nummer>", pegel={})
    for k, n in NUM.items():
        hvz["pegel"][n] = json.loads(_get(f"https://www.hochwasser.rlp.de/api/v1/measurement-site/{n}"))
    (out / "hvz.json").write_text(json.dumps(hvz), encoding="utf-8")
    st = station_table()
    base = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/"
    alt = pd.to_datetime(st.bis.astype(str), format="%Y%m%d") < pd.Timestamp.now() - pd.Timedelta(days=365)
    for sid in st.id[~alt]:                                       # seit über einem Jahr eingestellte Stationen (Bitburg, 2018) nicht abfragen
        for rel in (f"hourly/precipitation/recent/stundenwerte_RR_{sid}_akt.zip", f"10_minutes/precipitation/now/10minutenwerte_nieder_{sid}_now.zip"):
            try:
                (out / "dwd" / rel.split("/")[-1]).write_bytes(_get(base + rel))
            except RuntimeError as e:                             # Datei fehlt (Station ausgefallen): überspringen
                print("  ", e)
    data = {}
    for k, u in icon_urls(st).items():
        try:
            data[k] = json.loads(_get(u))
        except RuntimeError as e:
            if k not in ("icon_eu", "icon_global"):               # ohne ICON-EU / global entfällt nur der Ausblick
                raise
            print("  ", e)
    # ICON-D2-RUC (stündlich neu, kurze Reichweite): Kennung bei Open-Meteo nicht dokumentiert, daher mehrere Möglichkeiten; fehlt es, entfällt nur die Linie
    lat, lon = ",".join(f"{v:.4f}" for v in st.lat), ",".join(f"{v:.4f}" for v in st.lon)
    for mid, ep in [(m, e) for m in RUC_MODELS for e in ("forecast", "dwd-icon")]:
        try:
            j = json.loads(_get(f"https://api.open-meteo.com/v1/{ep}?latitude={lat}&longitude={lon}&hourly=precipitation&models={mid}"
                                f"&past_days=0&forecast_days=2&timezone=GMT"))
        except RuntimeError:
            continue
        if isinstance(j, list) and len(j) == len(st) and any(v is not None for v in j[0].get("hourly", {}).get("precipitation", [])):
            data["icon_ruc"], data["icon_ruc_modell"] = j, f"{mid} ({ep})"
            break
    print("   ICON-RUC:", data.get("icon_ruc_modell", "nicht verfügbar"))
    # Nachbarschaft: ICON-D2 an um NB_KM versetzten Punkten (Gewitterzellen liegen in der Vorhersage oft richtig, aber um einige zehn km versetzt);
    # fehlt ein Abruf, entfällt nur diese Richtung
    nb = {}
    for name, (dx, dy) in NB_DIRS.items():
        la = ",".join(f"{v + dy * NB_KM / 111.2:.4f}" for v in st.lat)
        lo = ",".join(f"{v + dx * NB_KM / (111.2 * np.cos(np.radians(la0))):.4f}" for v, la0 in zip(st.lon, st.lat))
        try:
            j = json.loads(_get(f"https://api.open-meteo.com/v1/dwd-icon?latitude={la}&longitude={lo}&hourly=precipitation&models=icon_d2"
                                f"&past_days=0&forecast_days=3&timezone=GMT", tries=2))
        except RuntimeError as e:
            print("  ", e)
            continue
        if isinstance(j, list) and len(j) == len(st):
            nb[name] = j
    if nb:
        data["icon_nb"] = nb
    print(f"   Nachbarschaft: {len(nb)} von {len(NB_DIRS)} Richtungen")
    (out / "icon.json").write_text(json.dumps(dict(abgerufen_utc=now, stationen=st[["id", "lat", "lon"]].values.tolist(), data=data)), encoding="utf-8")
    (out / "abruf.json").write_text(json.dumps(dict(dwd=now, hvz=now, icon=now), indent=1), encoding="utf-8")
    print("Rohdaten in", out)
    return out


# ------------------------------------------------------------------ Einlesen
def _mez(t) -> pd.DatetimeIndex:
    """UTC (mit oder ohne Zeitzone) -> MEZ ohne Zone."""
    t = pd.to_datetime(t, utc=True)
    return pd.DatetimeIndex(t.tz_convert(None)) + MEZ


def read_hvz(raw: Path) -> dict:
    j = json.loads((raw / "hvz.json").read_text(encoding="utf-8"))
    out = {}
    for k, n in NUM.items():
        site = j["pegel"][n]
        rec = {}
        for var in ("Q", "W"):
            if site.get(var):
                m = site[var]["measurements"]
                rec[var] = pd.Series([x["y"] for x in m], index=_mez([x["x"] for x in m]), dtype=float).sort_index()
                pr = site[var].get("predictions")
                if pr and pr.get("p50"):
                    rec[var + "_pred"] = {p: pd.Series([x["y"] for x in pr[p]], index=_mez([x["x"] for x in pr[p]]), dtype=float)
                                          for p in ("p10", "p50", "p90") if pr.get(p)}
                    rec[var + "_pred_time"] = str(_mez([pr["time"]])[0])
        out[k] = rec
    return out


def read_dwd_10min_now(path: Path) -> pd.Series:
    """10-Minuten-Werte (heute) zu Stundensummen, Stempel = Ende der Stunde (wie Stundenwerte), UTC ohne Zone. Nur vollständige Stunden."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.startswith("produkt")][0]
        df = pd.read_csv(io.StringIO(z.read(name).decode("latin-1")), sep=";", skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    t = pd.to_datetime(df["MESS_DATUM"].str.strip(), format="%Y%m%d%H%M")
    v = pd.to_numeric(df["RWS_10"].str.strip(), errors="coerce")
    v[v < 0] = np.nan
    s = pd.Series(v.to_numpy(), index=t).sort_index()
    g = s.resample("1h", label="right", closed="right")
    h = g.sum(min_count=6)
    h[g.count() < 6] = np.nan
    return h


def station_obs(raw: Path) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Stundenwerte je Station in MEZ (Stempel = Ende der Stunde): historisch (data/dwd_rr), recent und now aus dem Rohdatenordner.
    Rückgabe auch die letzte Meldung je Station."""
    import glob
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / "dwd_rr" / "stundenwerte_RR_*_hist.zip"))):
        out[Path(f).name.split("_")[2]] = RN.read_dwd_hourly(f, shift_h=1.0)
    for f in sorted((raw / "dwd").glob("stundenwerte_RR_*_akt.zip")):
        k = f.name.split("_")[2]
        s = RN.read_dwd_hourly(f, shift_h=1.0)
        out[k] = s if k not in out else s.combine_first(out[k])
    for f in sorted((raw / "dwd").glob("10minutenwerte_nieder_*_now.zip")):
        k = f.name.split("_")[2]
        s = read_dwd_10min_now(f)
        s.index = s.index + MEZ
        out[k] = s if k not in out else out[k].combine_first(s)
    last = {k: str(v.dropna().index.max()) for k, v in out.items() if v.notna().any()}
    return out, last


def read_icon(raw: Path) -> dict:
    """ICON-D2 (15 min und stündlich) und ICON-D2-EPS (stündlich je Lauf) je Station; Index MEZ, Werte mm je Intervall (Stempel = Ende)."""
    j = json.loads((raw / "icon.json").read_text(encoding="utf-8"))
    ids = [s[0] for s in j["stationen"]]
    d2, eps = {}, {}
    for sid, loc in zip(ids, j["data"]["icon_d2"]):
        m = loc["minutely_15"]
        d2[sid] = pd.Series(pd.to_numeric(pd.Series(m["precipitation"]), errors="coerce").to_numpy(), index=_mez(m["time"]))
    for sid, loc in zip(ids, j["data"]["icon_d2_eps"]):
        h = loc["hourly"]
        idx = _mez(h["time"])
        keys = ["precipitation"] + sorted(k for k in h if k.startswith("precipitation_member"))
        eps[sid] = pd.DataFrame({k: pd.to_numeric(pd.Series(h[k]), errors="coerce").to_numpy() for k in keys}, index=idx)
    longer = {}
    for key in ("icon_eu", "icon_global", "icon_ruc"):                        # stündlich, Stempel = Ende der Stunde wie beim Ensemble
        if key in j["data"]:
            longer[key] = {}
            for sid, loc in zip(ids, j["data"][key]):
                h = loc["hourly"]
                longer[key][sid] = pd.Series(pd.to_numeric(pd.Series(h["precipitation"]), errors="coerce").to_numpy(), index=_mez(h["time"]))
    nb = {}
    for name, locs in (j["data"].get("icon_nb") or {}).items():
        nb[name] = {sid: pd.Series(pd.to_numeric(pd.Series(loc["hourly"]["precipitation"]), errors="coerce").to_numpy(), index=_mez(loc["hourly"]["time"]))
                    for sid, loc in zip(ids, locs)}
    meta = {}
    for k in ("meta_d2", "meta_d2_eps"):
        m = j["data"].get(k) or {}
        if "last_run_initialisation_time" in m:
            meta[k] = str(pd.Timestamp(m["last_run_initialisation_time"], unit="s") + MEZ)
    # Lauf des Ensembles aus den Daten: letzte Stunde mit Werten minus 48 h (die Meta-Datei des Ensembles ist nicht immer aktuell)
    e0 = next(iter(eps.values()))
    last = e0["precipitation"].dropna().index.max()
    meta["eps_end"] = str(last)
    meta["d2_end"] = str(next(iter(d2.values())).dropna().index.max())
    return dict(d2=d2, eps=eps, ids=ids, meta=meta, abgerufen=j["abgerufen_utc"], eu=longer.get("icon_eu"), glob=longer.get("icon_global"),
                ruc=longer.get("icon_ruc"), ruc_modell=j["data"].get("icon_ruc_modell"), nb=nb)


# ------------------------------------------------------------------ Aufbau
def area_info() -> dict:
    """Fläche, Schwerpunkt und Stationsgewichte je Gebiet wie rain_fit.rain_frames (mit den Oberläufen aus chain_fc.HW)."""
    N = RF2.net()
    defs = RF2.lateral_defs(N)
    for k, a in C.HW.items():
        defs[k] = (float(N.loc[a["dn"], "A"]), (float(N.loc[a["nodes"], "lat"].mean()), float(N.loc[a["nodes"], "lon"].mean())))
    xy, _ = RF2.stations()
    return {k: dict(area_km2=area, centroid=ctr, weights=RN.idw_weights(xy, ctr)) for k, (area, ctr) in defs.items()}


def hourly_to_grid(h: pd.Series, grid: pd.DatetimeIndex) -> np.ndarray:
    """Stundensumme (Stempel = Ende) -> Intensität mm/h auf dem 15-min-Raster (wie rain.areal_frame)."""
    return h.reindex(grid.ceil("1h")).to_numpy()


def build_frames(p: dict[str, np.ndarray], Qn: dict[str, np.ndarray]) -> dict[str, dict]:
    """p: Gebiet -> Intensität mm/h auf dem Raster (bis zum Ende der Vorhersage); Vorregenindizes fortlaufend (ab Rasterbeginn); Gebietsfeuchte aus
    dem gemessenen Abfluss (Qn: Pegelname -> Reihe auf demselben Raster)."""
    out = {}
    for k, v in p.items():
        out[k] = dict(p=v, api=RN.api_series(v, DT, RN.TAU_API_H), api_s=RN.api_series(v, DT, RN.TAU_API_S_H))
    return C.add_wetness(out, Qn)


def mixture(members: list[np.ndarray], tabs: list[np.ndarray], d: float, levels=UN.LEV) -> np.ndarray:
    """Quantile der Mischung über die Läufe (gleich gewichtet): members je [H] Abfluss, tabs je [H, Stufen] Quantile des Restfehlers.
    Rückgabe [H, len(levels)] in m³/s; je Vorlauf Bisektion auf log(Q + d)."""
    M = np.stack([np.maximum(m, 0) for m in members])           # [Läufe, H]
    T = np.stack(tabs)                                          # [Läufe, H, Stufen]
    lo = np.log(M.min(0) + d) + T[:, :, 0].min(0) - 3.0
    hi = np.log(M.max(0) + d) + T[:, :, -1].max(0) + 3.0
    out = np.empty((M.shape[1], len(levels)))
    for j, lev in enumerate(levels):
        a, b = lo.copy(), hi.copy()
        for _ in range(40):
            mid = 0.5 * (a + b)
            x = mid[None, :] - np.log(M + d)                    # Fehler, der den Wert exp(mid) - d ergibt, je Lauf
            F = UN.cdf_fast(x, T).mean(0)
            a = np.where(F < lev, mid, a)
            b = np.where(F >= lev, mid, b)
        out[:, j] = np.exp(0.5 * (a + b)) - d
    return np.maximum(out, 0)


def exceed_mix(members: list[np.ndarray], tabs: list[np.ndarray], thr: float, d: float) -> np.ndarray:
    """P(Schwelle bis zum Vorlauf erreicht) [H]: je Lauf laufendes Maximum (Rang über die Vorläufe gekoppelt), dann Mittel über die Läufe."""
    P = [UN.exceed_prob(np.maximum(m, 0)[None, :], t[None, :, :], thr, d)[0] for m, t in zip(members, tabs)]
    return np.mean(P, axis=0)


def paths_max(members: list[np.ndarray], tabs: list[np.ndarray], d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gewichtete Pfade (Lauf x Quantilstufe, Rang über die Vorläufe gleich): Höchstwert und Zeitpunkt in 48 h je Pfad, Gewichte."""
    lev = UN.LEV
    edges = np.r_[0, (lev[1:] + lev[:-1]) / 2, 1]
    wl = np.diff(edges)
    mx, am, w = [], [], []
    for m, t in zip(members, tabs):
        traj = (np.maximum(m, 0)[:, None] + d) * np.exp(t) - d     # [H, Stufen]
        mx.append(traj.max(0))
        am.append(traj.argmax(0) + 1)
        w.append(wl / len(members))
    return np.concatenate(mx), np.concatenate(am), np.concatenate(w)


def wquant(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(q, c, x[o])


def sim_data(Qn: dict, p_d2: dict, i0: int, offs: dict) -> dict:
    """Alles, was die Seite für die Szenario-Rechnung im Browser braucht (forecast_core.js): Modellsatz, Konstanten und das jetzige Fenster im
    Format der Hochwasserfenster (chain_floods.json): Abfluss ab t0 - 48 h, Regen ab t0 - 48 h - 72 h bis t0 + 48 h (nach t0 der ICON-D2-Hauptlauf)."""
    doc = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    a, b = i0 - C.A, i0 + C.H
    r0 = a - C.LEAD
    fr = build_frames(p_d2, Qn)
    fl = dict(rain=True,
              q={C.DN[k]: [None if not np.isfinite(v) else round(float(v), 4) for v in Qn[C.DN[k]][a:b + 1]] for k in C.ORDER},
              p={k: [round(float(v), 4) for v in p_d2[k][r0:b + 1]] for k in C.RAINK},
              api0={k: [float(fr[k]["api"][r0 - 1]), float(fr[k]["api_s"][r0 - 1])] for k in C.RAINK},
              storm={k: "0" * C.A + ("1" if C.is_storm(k, np.array([i0]))[0] else "0") for k in C.ORDER})
    fl["niveau"] = dict(C.LEVEL)
    return dict(consts=doc["consts"], set=doc["sets"]["alle"], fl=fl, offset={k: float(offs[k]) for k in C.ORDER})


OUTLOOK_H = 168                                                  # Ausblick in Stunden


def outlook(models: dict, hv: dict, t0: pd.Timestamp, obs_filled: dict, w_live: dict, d2h: dict, icon: dict, info: dict, corr48: dict,
            offs: dict, rc: dict) -> dict | None:
    """Ausblick über 7 Tage, eine Linie je Pegel ohne Unsicherheitsband: dieselbe Kette (kette_vorhersage), Regen je Station aus ICON-D2, nach
    dessen Ende aus ICON-EU, danach aus ICON global (jeweils das nächste Modell der ICON-Kette; EU und global ohne Angleichfaktor, dafür gibt es
    keine Prüfung). Die Fehlerkorrektur (beta) klingt nach 48 h weiter ab; bis 48 h gilt der Median-Abgleich des ICON-D2-Laufs (corr48), danach
    bleibt dessen letzter Faktor stehen. Nicht geprüft: Für Vorläufe über 48 h gibt es keine Rückrechnung."""
    if not icon.get("eu"):
        return None
    H0, H7 = C.H, int(OUTLOOK_H / DT)
    grid = pd.date_range(t0 - pd.Timedelta(days=HISTORY_D), t0 + pd.Timedelta(hours=OUTLOOK_H), freq="15min")
    i0 = int(grid.get_loc(t0))
    hours = pd.date_range(grid[0].ceil("1h"), grid[-1].ceil("1h"), freq="1h")
    tH = t0.ceil("1h")
    Qn = {}
    for k in NUM:
        q = hv[k]["Q"].reindex(grid).to_numpy().copy()
        q[i0 + 1:] = np.nan
        q[:i0 + 1] = pd.Series(q[:i0 + 1]).interpolate(limit=4, limit_area="inside").to_numpy()
        Qn[C.DN[k]] = q
    ends, st_fc = {}, {}
    for sid in next(iter(w_live.values())):
        parts = [("ICON-D2", d2h[sid]), ("ICON-EU", icon["eu"][sid])] + ([("ICON global", icon["glob"][sid])] if icon.get("glob") else [])
        s = pd.Series(np.nan, index=hours)
        for name, x in parts:
            x = x.reindex(hours)
            use = s.isna() & x.notna() & (hours > tH)
            s[use] = x[use]
            if use.any():
                ends[name] = max(ends.get(name, hours[0]), hours[use].max())
        s[hours <= tH] = obs_filled[sid].reindex(hours)[hours <= tH]
        st_fc[sid] = s
    p = {k: np.nan_to_num(hourly_to_grid(RN.areal_series(st_fc, w_live[k]), grid)) for k in C.RAINK}
    m7 = C.from_json(C.to_json(models))
    for ms in m7.values():
        for m in ms.values():
            b = np.asarray(m.beta, float)
            r = float(np.clip((abs(b[-1]) / max(abs(b[-97]), 1e-9)) ** (1 / 96), 0.0, 1.0)) if len(b) >= 97 and b[-97] != 0 else 0.0
            m.beta = np.concatenate([b, b[-1] * r ** np.arange(1, H7 - len(b) + 1)])
    keep = {k: C._S.get(k) for k in ("pm", "month")}
    out = {}
    try:
        C.H = H7
        C._S.update(pm=C.storm_index({k: dict(p=np.where(grid > t0, 0.0, p[k])) for k in C.RAINK}, info), month=grid.month.to_numpy())
        fr = build_frames(p, Qn)
        for target in C.ORDER:
            dn = C.DN[target]
            raw = C.forecast(m7, target, Qn, fr, np.array([i0]), "kette_vorhersage")[target][0][C.A + 1:]
            d = offs[target]
            c48 = np.asarray(corr48[target])                       # ICON-D2-Lauf nach der Fehlerstatistik, 192 Schritte
            f = (c48 + d) / (np.maximum(raw[:H0], 0) + d)
            q = np.concatenate([c48, np.maximum((np.maximum(raw[H0:], 0) + d) * f[-1] - d, 0)])[3::4]      # stündlich
            cw = _chain_weights(target, info)
            pa = sum(w * p[x] for w, x in zip(*cw))
            day = pd.Series(pa[i0 + 1:] * DT, index=grid[i0 + 1:]).resample("1D").sum()
            out[target] = dict(q=[round(float(x), 3) for x in q], w=[round(float(x), 1) for x in RT.w_of_q(rc[dn], q)],
                               regen_tag=[[str(t)[:10], round(float(v), 1)] for t, v in day.items()])
    finally:
        C.H = H0
        C._S.update(**keep)
    return dict(pegel=out, ende={k: round((v - t0) / pd.Timedelta(hours=1), 1) for k, v in ends.items()}, stunden=OUTLOOK_H)


def build(raw: Path, out_html: Path) -> dict:
    t_start = time.time()
    hv = read_hvz(raw)
    obs_st, last_st = station_obs(raw)
    icon = read_icon(raw)
    fac = json.loads((HERE / "data" / "verif" / "rain_factors.json").read_text(encoding="utf-8"))["stationen"]
    K = UN.kennwerte()
    rc = json.loads((HERE / "data" / "rating_curves.json").read_text(encoding="utf-8"))["pegel"]
    rc_all = rc
    U = json.loads((HERE / "data" / "uncert_tables.json").read_text(encoding="utf-8"))
    models = C.from_json(json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]["alle"])
    # Niveaufaktor je Pegel (niveau.py): Die Kette ist an den Abflüssen bis 2021 angepasst; seit dem Wechsel der Wasserstand-Abfluss-Beziehung
    # (Prümzurlay Januar 2022, Nims) zeigt der Pegel bei gleichem Zufluss ein anderes Niveau. Der Faktor multipliziert die Simulation vor der Fehlerkorrektur.
    nv_path = HERE / "data" / "niveau.json"
    niveau = {k: v["faktor"] for k, v in json.loads(nv_path.read_text(encoding="utf-8"))["pegel"].items() if v.get("angewendet")} if nv_path.exists() else {}
    C.LEVEL.clear()
    C.LEVEL.update(niveau)

    # Ausgabezeit: letzter 15-min-Zeitpunkt, zu dem alle acht Pegel einen Abfluss haben
    t0 = min(hv[k]["Q"].dropna().index.max() for k in NUM)
    grid = pd.date_range(t0 - pd.Timedelta(days=HISTORY_D), t0 + pd.Timedelta(hours=C.H_H), freq="15min")
    i0 = int(grid.get_loc(t0))
    n = len(grid)
    Qn = {}
    for k in NUM:
        q = hv[k]["Q"].reindex(grid).to_numpy().copy()
        q[i0 + 1:] = np.nan
        # einzelne Lücken (bis 1 h) linear füllen, sonst bleibt die Reihe leer und die Vorhersage entfällt
        q[:i0 + 1] = pd.Series(q[:i0 + 1]).interpolate(limit=4, limit_area="inside").to_numpy()
        Qn[C.DN[k]] = q

    # Stationsregen: Stunden bis zur letzten Meldung je Station; Stationen ohne Meldung in den letzten 24 h fallen weg
    hours = pd.date_range(grid[0].ceil("1h"), grid[-1].ceil("1h"), freq="1h")
    t_obs_end = {}
    live_ids = []
    for sid, s in obs_st.items():
        s = s.reindex(hours)
        last = s.dropna().index.max() if s.notna().any() else None
        if last is not None and last >= t0 - pd.Timedelta(hours=24):
            live_ids.append(sid)
            t_obs_end[sid] = last
    # ICON-D2: 15-min-Summen -> Stundensummen (Stempel Ende); Ensemble stündlich; je Station mit Faktor angeglichen
    d2h = {sid: s.resample("1h", label="right", closed="right").sum(min_count=4) * fac.get(sid, {"faktor": 1.0})["faktor"] for sid, s in icon["d2"].items()}
    eps = {sid: df * fac.get(sid, {"faktor": 1.0})["faktor"] for sid, df in icon["eps"].items()}
    members = list(next(iter(eps.values())).columns)
    info = area_info()
    tH = t0.ceil("1h")
    # gemessen bis zur letzten Meldung, danach bis jetzt ICON-D2 (jüngste Läufe), danach Vorhersage
    filled = {}
    for sid in live_ids:
        s = obs_st[sid].reindex(hours).copy()
        gap = (s.index > t_obs_end[sid]) & (s.index <= tH)
        s[gap] = d2h[sid].reindex(s.index[gap]).to_numpy()
        s[s.index > tH] = np.nan
        filled[sid] = s
    w_live = {k: {s: v for s, v in a["weights"].items() if s in live_ids} for k, a in info.items()}
    p_obs = {k: hourly_to_grid(RN.areal_series(filled, w_live[k]), grid) for k in C.RAINK}
    fut = grid > t0

    def areal_fc(series_by_station: dict[str, pd.Series], k: str) -> np.ndarray:
        h = RN.areal_series({s: series_by_station[s].reindex(hours) for s in w_live[k]}, w_live[k])
        return hourly_to_grid(h, grid)

    runs = {"ICON-D2": {k: areal_fc(d2h, k) for k in C.RAINK}}
    for m in members:
        runs["EPS " + ("Kontrolle" if m == "precipitation" else m.replace("precipitation_member", ""))] = {k: areal_fc({s: eps[s][m] for s in eps}, k) for k in C.RAINK}
    p_runs = {}
    for name, pr in runs.items():
        p_runs[name] = {k: np.where(fut, np.nan_to_num(pr[k]), np.nan_to_num(p_obs[k])) for k in C.RAINK}
    p_none = {k: np.where(fut, 0.0, np.nan_to_num(p_obs[k])) for k in C.RAINK}
    # ICON-RUC als zusätzlicher Lauf (nicht im Band und nicht in der Warnwahrscheinlichkeit): RUC-Regen, soweit er reicht, danach ICON-D2;
    # ohne Angleichfaktor, weil es für RUC keine Prüfung gegen die Stationen gibt
    p_ruc, ruc_end = None, None
    if icon.get("ruc"):
        comb, ends = {}, []
        for sid in d2h:
            x = icon["ruc"][sid].reindex(hours)
            use = x.notna() & (hours > tH)
            comb[sid] = pd.Series(np.where(use, x, d2h[sid].reindex(hours)), index=hours)
            if use.any():
                ends.append(hours[use].max())
        if ends:
            ruc_end = round((min(ends) - t0) / pd.Timedelta(hours=1), 1)
            if ruc_end > RUC_MAX_H:                                # ICON-D2-RUC reicht höchstens gut einen Tag: sonst hat die Schnittstelle ein anderes Modell geliefert
                print(f"   ICON-RUC verworfen: Daten reichen {ruc_end:.0f} h weit ({icon.get('ruc_modell')}), das ist nicht ICON-D2-RUC")
                ruc_end = None
            else:
                p_ruc = {k: np.where(fut, np.nan_to_num(areal_fc(comb, k)), np.nan_to_num(p_obs[k])) for k in C.RAINK}

    # Nachbarschaft: ICON-D2 an versetzten Punkten, mit denselben Angleichfaktoren; nicht im Band und nicht in der Warnwahrscheinlichkeit
    p_nb = {}
    for name, ser in (icon.get("nb") or {}).items():
        h = {sid: ser[sid] * fac.get(sid, {"faktor": 1.0})["faktor"] for sid in ser}
        p_nb[name] = {k: np.where(fut, np.nan_to_num(areal_fc(h, k)), np.nan_to_num(p_obs[k])) for k in C.RAINK}

    # Zustand: Starkregen-Einteilung und Monat für chain_fc.is_storm
    C._S.update(pm=C.storm_index({k: dict(p=p_none[k]) for k in C.RAINK}, info), month=grid.month.to_numpy())
    feats = {}
    UN._S.update(Qn=Qn, rain={k: sum(w * p_none[x] for w, x in zip(*_chain_weights(k, info))) for k in C.ORDER})
    for k in C.ORDER:
        feats[k] = UN.features_at(k, np.array([i0]))
    res = {}
    T_v, T_r = U["vorhersage"]["alle"], U["regen"]["alle"]
    for target in C.ORDER:
        dn = C.DN[target]
        d = U["offset"][target]
        st = int(UN.state_class(feats[target], target)[0])
        out = {}
        for name, pr in p_runs.items():
            S = C.forecast(models, target, Qn, build_frames(pr, Qn), np.array([i0]), "kette_vorhersage")
            out[name] = S[target][0]
        S0 = C.forecast(models, target, Qn, build_frames(p_none, Qn), np.array([i0]), "kette_regen")[target][0]
        A, H = C.A, C.H
        mem = [out[nm][A + 1:] for nm in out]
        wet = int(Qn[dn][i0] >= UN.WET_EDGE * U["mq"][target])
        lev_of = lambda s: int(np.digitize(max(s[A], np.max(s[A + 1:])) / U["mq"][target], UN.LEVEL_EDGES)) * 2 + wet   # noqa: E731
        # Starkregenlage je Lauf (Regen 24 h vor bis 24 h nach der Ausgabezeit, Vorhersage des Laufs): dann die Starkregentabelle
        cw = _chain_weights(target, info)
        heavy_run = {nm: bool(UN.heavy_flags(sum(w * p_runs[nm][x] for w, x in zip(*cw)), np.array([i0]))[1][0]) for nm in out}
        heavy0 = bool(feats[target]["hp"][0])
        tabs = [_interp(T_v[target]["q"][st * UN.NSUB + lev_of(out[nm]) + UN.NB * heavy_run[nm]]) for nm in out]
        band = mixture(mem, tabs, d)
        # Läufe nach der Fehlerstatistik bereinigt (Median des Restfehlers je Lauf und Vorlauf): gleicht die Drift des Modells bei Niedrigwasser aus
        corr = {nm: np.maximum((np.maximum(m, 0) + d) * np.exp(t[:, 5]) - d, 0) for nm, m, t in zip(out, mem, tabs)}
        tab0 = _interp(T_r[target]["q"][st * UN.NSUB + lev_of(S0) + UN.NB * heavy0])
        band0 = np.maximum((np.maximum(S0[A + 1:], 0)[:, None] + d) * np.exp(tab0) - d, 0)
        mx, am, w = paths_max(mem, tabs, d)
        q0 = float(Qn[dn][i0])
        thr = []
        rcv = rc[dn]
        kw = K[dn]
        th_list = ([("MH", "Meldehöhe", kw["meldehoehe"], rcv["q_meldehoehe"])] if kw["meldehoehe"] else []) + \
                  [(f"HW{x}", f"HW {x}", kw["W"][f"HW{x}"], kw["Q"][f"HQ{x}"]) for x in ("2", "10", "20", "50", "100")]
        for tid, tname, wv, qv in sorted(th_list, key=lambda z: z[2]):
            if q0 >= qv:
                thr.append(dict(id=tid, name=tname, w=wv, q=qv, now=True))
                continue
            P = exceed_mix(mem, tabs, qv, d)
            P0 = UN.exceed_prob(np.maximum(S0[A + 1:], 0)[None, :], tab0[None], qv, d)[0]
            first = lambda P, lv: next((int(l + 1) for l in range(H) if round(P[l] * 20) / 20 >= lv), None)   # noqa: E731
            thr.append(dict(id=tid, name=tname, w=wv, q=qv, now=False, p=[round(float(P[l - 1]), 3) for l in (24, 48, 96, 192)],
                            p0=[round(float(P0[l - 1]), 3) for l in (24, 48, 96, 192)], ab10=first(P, 0.1), ab50=first(P, 0.5)))
        # Warnschwelle (uncert.WARN, Standard HQ10): darunter nur Einordnung, keine Warnung
        wid = "HW" + UN.warn_name(target)[2:]
        for t in thr:
            t["warn"] = t["id"] == wid
            t["info"] = t["w"] < kw["W"][wid]
        wt = next(t for t in thr if t["warn"])
        p48 = None if wt["now"] else round(wt["p"][3] * 20) / 20                      # auf 5 % gerundet wie in der Anzeige
        wstat = "über" if wt["now"] else "Warnung" if p48 >= UN.WARN_LEVELS[1] else "möglich" if p48 >= UN.WARN_LEVELS[0] else "keine"
        warn = dict(id=wid, name=wt["name"], w=wt["w"], q=wt["q"], status=wstat, p48=None if wt["now"] else wt["p"][3],
                    ab10=wt.get("ab10"), ab50=wt.get("ab50"))
        wq = lambda q: [round(float(x), 1) for x in RT.w_of_q(rcv, np.asarray(q))]            # noqa: E731
        hist0 = i0 - 72 * 4
        obs_hist = Qn[dn][hist0:i0 + 1]
        mx_q = wquant(mx, w, [0.1, 0.5, 0.9])
        mx_t = wquant(am.astype(float), w, [0.1, 0.5, 0.9])
        ruc = None
        if p_ruc is not None:
            sr = C.forecast(models, target, Qn, build_frames(p_ruc, Qn), np.array([i0]), "kette_vorhersage")[target][0]
            hr = bool(UN.heavy_flags(sum(w_ * p_ruc[x] for w_, x in zip(*cw)), np.array([i0]))[1][0])
            cr = np.maximum((np.maximum(sr[A + 1:], 0) + d) * np.exp(_interp(T_v[target]["q"][st * UN.NSUB + lev_of(sr) + UN.NB * hr])[:, 5]) - d, 0)
            cr = cr[:int(ruc_end * 4)]                             # die Linie endet, wo der RUC-Regen endet (danach käme nur noch ICON-D2)
            ruc = dict(q=[round(float(x), 3) for x in cr], w=wq(cr), ende_h=ruc_end, modell=icon.get("ruc_modell"),
                       regen_mm=round(float(sum(w_ * p_ruc[x][i0 + 1:i0 + 1 + int(ruc_end * 4)].sum() for w_, x in zip(*cw)) * DT), 1),
                       regen_d2_mm=round(float(sum(w_ * p_runs["ICON-D2"][x][i0 + 1:i0 + 1 + int(ruc_end * 4)].sum() for w_, x in zip(*cw)) * DT), 1))
        # Nachbarschaft: je Richtung Höchststand (nach der Fehlerstatistik bereinigt), Regen und Wahrscheinlichkeit der Warnschwelle in 48 h
        nb_res = None
        if p_nb:
            rows_nb = []
            for name, pn in p_nb.items():
                sn = C.forecast(models, target, Qn, build_frames(pn, Qn), np.array([i0]), "kette_vorhersage")[target][0]
                rn = sum(w_ * pn[x] for w_, x in zip(*cw))
                tn = _interp(T_v[target]["q"][st * UN.NSUB + lev_of(sn) + UN.NB * bool(UN.heavy_flags(rn, np.array([i0]))[1][0])])
                cn = np.maximum((np.maximum(sn[A + 1:], 0) + d) * np.exp(tn[:, 5]) - d, 0)
                pw = None if wt["now"] else float(UN.exceed_prob(np.maximum(sn[A + 1:], 0)[None, :], tn[None], wt["q"], d)[0][-1])
                rows_nb.append(dict(richtung=name, regen48=round(float(rn[i0 + 1:i0 + 1 + H].sum() * DT), 1), q_max=round(float(cn.max()), 3),
                                    w_max=wq([cn.max()])[0], p48=None if pw is None else round(pw, 3)))
            nb_res = dict(km=NB_KM, laeufe=rows_nb)
        # Regenschwellen: Regen in 3 bis 24 h bis zur Vorwarnung (10 %) und Warnung (50 %), dazu die größten vorhergesagten Summen
        rs = None
        if not wt["now"]:
            th = RTH.thresholds(models, target, Qn, p_none, i0, info, T_v[target]["q"], st, U["mq"][target], d, wt["q"])
            ms = {nm: RTH.max_sums(sum(w_ * p_runs[nm][x] for w_, x in zip(*cw)), i0) for nm in p_runs}
            eps_names = [nm for nm in p_runs if nm != "ICON-D2"]
            rs = dict(dauer_h=list(RTH.DURS_H), r10=[th[D]["r10"] and round(th[D]["r10"], 1) for D in RTH.DURS_H],
                      r50=[th[D]["r50"] and round(th[D]["r50"], 1) for D in RTH.DURS_H], max_mm=RTH.R_GRID[-1],
                      d2=[round(ms["ICON-D2"][D], 1) for D in RTH.DURS_H],
                      eps=[[round(float(x), 1) for x in np.quantile([ms[nm][D] for nm in eps_names], [0.1, 0.5, 0.9])] for D in RTH.DURS_H],
                      nb=[round(max(RTH.max_sums(sum(w_ * pn[x] for w_, x in zip(*cw)), i0)[D] for pn in p_nb.values()), 1) for D in RTH.DURS_H] if p_nb else None)
        res[target] = dict(
            nachbar=nb_res, regenschwellen=rs, ruc=ruc, dn=dn, state=st, dry=not wet, mq=U["mq"][target], q0=q0, w0=float(RT.w_of_q(rcv, np.array([q0]))[0]), w1=float(RT.w_of_q(rcv, np.array([Qn[dn][i0 - 4]]))[0]),
            obs_q=[None if not np.isfinite(x) else round(float(x), 3) for x in obs_hist], obs_w=[None if not np.isfinite(x) else v for x, v in zip(obs_hist, wq(np.nan_to_num(obs_hist)))],
            band_q=[[round(float(x), 3) for x in band[:, j]] for j in range(band.shape[1])], band_w=[wq(band[:, j]) for j in range(band.shape[1])],
            band0_q=[[round(float(x), 3) for x in band0[:, j]] for j in range(band0.shape[1])], band0_w=[wq(band0[:, j]) for j in range(band0.shape[1])],
            runs_w={nm: wq(cor) for nm, cor in corr.items()}, runs_q={nm: [round(float(x), 3) for x in cor] for nm, cor in corr.items()},
            runs_raw_q={nm: [round(float(x), 3) for x in out[nm][A + 1:]] for nm in out}, max48=dict(q=[round(float(x), 2) for x in mx_q], w=wq(mx_q), t_steps=[int(round(x)) for x in mx_t]),
            up_open=bool(feats[target]["up_open"][0]), thresholds=thr, warn=warn,
            heavy=dict(runs=int(sum(heavy_run.values())), of=len(heavy_run), past=heavy0),
            ups=[dict(dn=C.DN[u], peak=_peak_info(Qn[C.DN[u]], i0)) for u in C.UPS.get(target, [])],
            rain_area=_chain_rain(target, info, p_obs, p_runs, grid, i0))
        hvz = hv[target]
        if "W_pred" in hvz:
            res[target]["hvz"] = dict(time=hvz["W_pred_time"], **{p: [[str(t), float(v)] for t, v in s.items()] for p, s in hvz["W_pred"].items()})
    try:
        aus = outlook(models, hv, t0, filled, w_live, d2h, icon, info, {k: res[k]["runs_q"]["ICON-D2"] for k in res}, U["offset"], rc_all)
    except Exception as e:                                         # noqa: BLE001  Ausblick ist Zusatz: scheitert er, bleibt die 48-h-Seite
        print("   Ausblick entfällt:", repr(e))
        aus = None
    meta = dict(niveau={C.DN[k]: v for k, v in niveau.items()}, t0=str(t0), grid0=str(grid[i0 - 72 * 4]), abruf=json.loads((raw / "abruf.json").read_text(encoding="utf-8")) if (raw / "abruf.json").exists() else {},
                stationen_letzte=dict((k, str(v)) for k, v in t_obs_end.items()), stationen=live_ids, icon=icon["meta"], members=len(p_runs),
                gauges_last={C.DN[k]: str(hv[k]["Q"].dropna().index.max()) for k in NUM}, faktoren={k: v["faktor"] for k, v in fac.items()})
    data = dict(meta=meta, res=res, ausblick=aus, sim=sim_data(Qn, p_runs["ICON-D2"], i0, U["offset"]), lev=UN.LEV.tolist(), kennwerte=K, rating={g: dict(q=c["q"], w=c["w"]) for g, c in rc.items()},
                verif=_verif_summary(), check=_check_summary(), warncheck=_warn_check())
    tpl = (HERE / "live_template.html").read_text(encoding="utf-8")
    tpl = tpl.replace("/*__FC_CORE__*/", (HERE / "forecast_core.js").read_text(encoding="utf-8"))
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"{out_html} ({out_html.stat().st_size / 1024:.0f} kB, Ausgabe {t0} MEZ, {len(p_runs)} Läufe, {time.time() - t_start:.0f} s)")
    return data


def _interp(T: list) -> np.ndarray:
    """Fehlerquantile [gespeicherte Vorläufe, Stufen] linear auf jeden 15-min-Vorlauf 1..H (wie UQ.qAt in forecast_core.js)."""
    T = np.asarray(T)
    L = np.arange(1, C.H + 1)
    return np.stack([np.interp(L, UN.LEADS, T[:, k]) for k in range(T.shape[1])], axis=1)


def _chain_weights(k: str, info: dict) -> tuple[list[float], list[str]]:
    ks = [x for x in C.chain(k) if x in C.RAINK]
    a = np.array([info[x]["area_km2"] for x in ks])
    return list(a / a.sum()), ks


def _chain_rain(target: str, info: dict, p_obs: dict, p_runs: dict, grid: pd.DatetimeIndex, i0: int) -> dict:
    """Gebietsregen des ganzen Einzugsgebiets (flächengewichtet über die Gebiete der Kette), stündlich: 72 h gemessen, 48 h je Lauf."""
    w, ks = _chain_weights(target, info)
    def hourly(p):
        v = sum(wi * np.nan_to_num(p[x]) for wi, x in zip(w, ks))
        s = pd.Series(v, index=grid).iloc[i0 - 72 * 4 + 1:]
        return s.resample("1h", label="right", closed="right").mean()       # mm/h im Stundenmittel = mm je Stunde
    obs = hourly(p_obs)
    obs = obs[obs.index <= grid[i0]]
    runs = {nm: hourly(pr) for nm, pr in p_runs.items()}
    fut = {nm: s[s.index > grid[i0]] for nm, s in runs.items()}
    M = np.stack([v.to_numpy() for v in fut.values()])
    names = list(fut)
    cum = M.cumsum(1)
    return dict(t_obs=[str(t) for t in obs.index], obs=[round(float(x), 2) for x in obs.to_numpy()], t_fc=[str(t) for t in next(iter(fut.values())).index],
                d2=[round(float(x), 2) for x in fut["ICON-D2"].to_numpy()], q=np.round(np.quantile(M[1:], [0.1, 0.5, 0.9], axis=0), 2).tolist(),
                cum_q=np.round(np.quantile(cum[1:], [0.1, 0.5, 0.9], axis=0), 1).tolist(), cum_d2=np.round(cum[names.index("ICON-D2")], 1).tolist(),
                sum48=dict(d2=round(float(cum[names.index("ICON-D2"), -1]), 1), eps=np.round(np.quantile(cum[1:, -1], [0.1, 0.5, 0.9]), 1).tolist()))


def _peak_info(q: np.ndarray, i0: int) -> dict:
    past = q[i0 - C.A:i0 + 1]
    passed = bool(C.peak_passed(past[None, :])[0]) if np.isfinite(past).all() else False
    j = int(np.nanargmax(past))
    return dict(passed=passed, steps_ago=int(C.A - j), q=round(float(past[j]), 2))


def _verif_summary() -> dict | None:
    p = HERE / "data" / "verif" / "rain_verif.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    g = d[d.ort.str.startswith("Gebiet")]
    out = {}
    for lead, x in g.groupby("vorlauf"):
        out[lead] = dict(r6=round(float(x[x.summe_h == 6].r.mean()), 2), r24=round(float(x[x.summe_h == 24].r.mean()), 2),
                         hit10=round(float(x[x.summe_h == 6]["treffer>=10"].mean()), 2), fa10=round(float(x[x.summe_h == 6]["fehlalarm>=10"].mean()), 2),
                         ratio=round(float(x[x.summe_h == 24].sum_ratio.mean()), 2))
    return out


def _check_summary() -> dict | None:
    p = HERE / "data" / "uncert_band.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    x = b[(b.variante == "kette_vorhersage") & (b.pruefung == "cv_2004_2022") & (b.klasse == "alle")]
    return {r["pegel"]: {h: round(float(r[f"im80_{h}h"]), 3) for h in (6, 24)} for _, r in x.iterrows()} if len(x) else None


def _warn_check() -> dict | None:
    """Rückblick auf die Warnschwelle (uncert.py): je Pegel jedes Hochwasser über der Schwelle mit Vorwarnzeiten (ohne Regenvorhersage, mit
    gemessenem Regen als perfekter Vorhersage) und Fehlalarme je Stufe, 2004-2022 kreuzvalidiert."""
    pe, pp = HERE / "data" / "uncert_warn_events.csv", HERE / "data" / "uncert_warn_phases.csv"
    if not (pe.exists() and pp.exists()):
        return None
    ev, ph = pd.read_csv(pe), pd.read_csv(pp)
    ev = ev[ev.pruefung == "cv_2004_2022"]
    ph = ph[ph.pruefung == "cv_2004_2022"]
    out = {}
    for g in sorted(set(ev.pegel) | set(ph.pegel)):
        e = ev[(ev.pegel == g) & ev.variante.isin(["kette_regen", "kette_vorhersage"])]
        rows = {}
        for _, r in e.sort_values("beginn").iterrows():
            x = rows.setdefault(r.beginn, dict(beginn=r.beginn, q=round(float(r.scheitel_q), 1), thr=round(float(r.schwelle_q), 1)))
            x[r.variante] = {**{f"{a}{lv}": (None if pd.isna(r[f"{c}_{lv}_h"]) else float(r[f"{c}_{lv}_h"])) for a, c in (("v", "vorwarn"), ("e", "erste"), ("l", "letzte")) for lv in (10, 50)},
                             "pmax": None if pd.isna(r.p_max_48h) else float(r.p_max_48h)}
        fa = {}
        for (var, lev), x in ph[ph.pegel == g].groupby(["variante", "stufe"]):
            f = x[~x.treffer]
            fa[f"{var}|{lev:g}"] = dict(n=len(f), knapp=int((f.max_vs_schwelle >= 0.8).sum()))
        out[g] = dict(events=list(rows.values()), fa=fa)
    return out


def latest_raw() -> Path:
    c = sorted((HERE / "data" / "live").glob("raw_*"))
    if not c:
        raise SystemExit("Keine Rohdaten: erst python live.py fetch")
    return c[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "fetch", "build"])
    ap.add_argument("raw", nargs="?", default=None)
    ap.add_argument("--out", default=str(HERE / "gui" / "live.html"))
    a = ap.parse_args()
    if a.cmd in ("all", "fetch"):
        raw = fetch()
    if a.cmd in ("all", "build"):
        build(Path(a.raw) if a.raw else (raw if a.cmd == "all" else latest_raw()), Path(a.out))
