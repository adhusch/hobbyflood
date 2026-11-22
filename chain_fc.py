"""Kettenvorhersage über das ganze Pegelnetz: Ausgabe alle 15 min, Vorlauf in 15-min-Schritten bis 48 h.

  python chain_fc.py run [--jobs 2]     Modelle je Zeitblock anpassen (ohne den Block), Vorhersage aller Pegel kreuzvalidiert auswerten,
                                         Modelle und Hochwasserfenster für die Karte schreiben. Ausgaben data/chain_*.csv/json

Netz (oben nach unten):
  Oberläufe   Prüm 2 (P2), Sinspelt (SI, Enz), Giesdorf (GI, Nims): keine Oberlieger. Vorhersage entweder Persistenz (Q bleibt) oder
              Regen-Abfluss-Modell Q = c0 + Zwischengebietsantwort (lateral.py) auf den Gebietsregen des ganzen Einzugsgebiets.
  Strecken    Echtershausen <- Prüm 2, Wiersdorf <- Echtershausen (Stausee), Prümzurlay <- Wiersdorf + Sinspelt, Seffern <- Giesdorf,
              Alsdorf-Oberecken <- Seffern; mit Regen für die Zwischengebiete (außer Wiersdorf).

Vorhersage zur Ausgabezeit t0 (alle Messwerte bis t0 bekannt):
  einzeln       wie bisher: Oberlieger bis t0 gemessen, danach konstant; Routing ohne Regen.
  kette         Oberläufe konstant, jeder Unterlieger aus der Vorhersage seines Oberliegers (nach t0); ohne Regen.
  kette_regen_1 Kette, Oberläufe und Zwischengebiete aus Regen, gemessen bis t0, danach 0 (keine Regenvorhersage); ein Modellsatz für alle Ereignisse.
  kette_regen   wie kette_regen_1, aber mit Umschalten zwischen zwei Modellsätzen (Winterhochwasser / Starkregenhochwasser), siehe unten.
  kette_ideal   wie kette_regen, aber mit dem gemessenen Regen auch nach t0 (perfekte Regenvorhersage, Obergrenze, nicht betrieblich).
  kette_vorhersage  wie kette_ideal (Regen nach t0 aus frames, im Betrieb die Regenvorhersage), seit 14.11.2026 ohne Scheitel-Regel, also dieselbe Rechnung; mit gemessenem Regen nach t0
                gerechnet liefert sie die Restfehler des Abflussmodells bei bekanntem Regen (uncert.py), im Betrieb rechnet live.py damit je
                Ensemble-Lauf. Nicht Teil der Kreuzvalidierung in run().
  Umschalten: Starkregenhochwasser, wenn der Monat der Ausgabezeit Mai bis September ist UND der größte 1-h-Gebietsregen über dem Einzugsgebiet des
  Zielpegels in den 48 h bis zur Ausgabezeit mindestens 4 mm/h war; sonst Winterhochwasser. Gilt für die ganze Kette des Zielpegels. Die Modelle
  für Winterhochwasser sind an den Winterereignissen angepasst, die für Starkregenhochwasser an den (wenigen) Starkregenereignissen mit Strafterm
  zum Gesamtmodell (routing._run, prior). Die Trainingsereignisse werden mit derselben Regel zur Scheitelzeit eingeteilt.
  Seit 14.11.2026 gilt die Scheitel-Regel nur noch ohne Regenvorhersage (kette, kette_regen): Mit Regen als Vorhersage (kette_vorhersage, Live-Betrieb)
  verschlechterte sie alle Hochwasser 2004-2022 deutlich (NSE bei 24 h Vorlauf an Alsdorf-Oberecken 0,03 statt 0,30, Prümzurlay 0,08 statt 0,25,
  bei 48 h -1,0 statt 0,0 bis 0,1), ohne den Scheitel besser zu treffen; ohne Regenvorhersage hilft sie (24 h: Echtershausen -1,55 statt -1,95),
  weil dort der kommende Regen fehlt und die Vorhersage zu früh fällt. Tabelle: data/hold_test.csv.
  Scheitel-Regel (HOLD_RULE): Ein Pegel mit Oberliegern zeigt keinen Scheitel, solange nicht an allen seinen direkten Oberliegern ein Scheitel gemessen
  ist (peak_passed). Steigt seine Vorhersage über den jetzigen Wert, bleibt sie nach ihrem Höchstwert auf diesem stehen („mindestens“), statt zu fallen.
  Gilt die Kette hinunter, nicht für kette_ideal (dort ist der künftige Regen bekannt).
  Je Pegel Fehlerkorrektur: Abweichung Messung - Modell zur Ausgabezeit e0, Vorhersage = Modell + beta(Vorlauf) * e0. beta je Pegel und Modell aus den
  Trainingsereignissen: Regressionssteigung des Modellfehlers zur Zeit t + Vorlauf auf den Fehler zur Zeit t (Fehler je Ereignis durch den Scheitel
  geteilt), auf 0 bis 1 begrenzt und mit dem Vorlauf nicht steigend. Die korrigierte Vorhersage geht in den Unterlieger.
Anpassung (seit 12.11.2026): an Hochwassern und an Regenereignissen ohne Hochwasser (48-h-Gebietsregen ab 25 mm), Abflussbeiwert mit
Gebietsfeuchte aus dem Abfluss am Bezugspegel zu Beginn des Fensters (lateral.py, bf); vorher nur an Hochwassern, das gab nach Trockenphasen
mit bekanntem Regen 3- bis 4-mal zu viel Abfluss.
Sättigungsüberschuss (seit 13.11.2026, REGIONAL_SAT): Nach der Anpassung je Pegel bekommt jedes Zwischengebiet den Baustein "sat" (lateral.py,
SCS-Verfahren, schneller Weg) mit vier gemeinsamen Parametern (rs, Sv, g_ev, Kf), angepasst an allen Hochwasser- und Regenfenstern aller sieben
Regengebiete (ohne den Prüfblock), Grundmodelle fest. Je Pegel gab es nur ein bis zwei Extremhochwasser, an den Oberläufen nur 2021; einzeln
angepasst schaltete die Anpassung jede Erweiterung für große Ereignisse ab (extreme_fit.py). Mit gemessenem Regen erreichte die Kette bei den
Hochwassern über HQ10 vorher 18 % des Scheitels (Median, 24 h vor dem Überschreiten), mit dem Baustein 39 %, der Scheitel kommt nicht mehr 7 h zu spät.
Dazu (13.11.2026, zweiter Schritt) nichtlineare Speicher in allen Zwischengebieten (lateral.py, "nlr"; Speicherzeit sinkt, je voller der Speicher;
zwei weitere gemeinsame Parameter) und für Wiersdorf absolute Fehler in der Anpassung (WTS_EXP_KEY): Die Strecke über den Stausee dämpfte große
Wellen mit gemessenem Zufluss auf die Hälfte (2018: 116 statt 218 m³/s), weil die vielen kleinen Wellen die Wellengeschwindigkeit bei hohem Abfluss
bestimmten.
Prüfung: Blöcke wie rain_fit.py (2004-07, 2008-11, 2012-15, 2016-18, 2019-22); Modelle ohne den Block und ohne Ereignisse im Abstand von 3,5 Tagen.
Ausgabezeiten je Ereignis des Zielpegels 48 h vor bis 12 h nach seinem Scheitel, alle 15 min (241 je Ereignis); Vorlauf 15 min bis 48 h.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lateral as L
import pkgdata
import rain as RN
import rain_fit as RF2
import routing as R
import routing_fit as RF

HERE = Path(__file__).parent
DT = R.DT_H
HW = {"P2": dict(dn="Prüm 2", nodes=["Hermespand", "Prüm 2"]),
      "SI": dict(dn="Sinspelt", nodes=["Neurath / Arzfeld", "Neuerburg", "Sinspelt"]),
      "GI": dict(dn="Giesdorf", nodes=["Rommersheim", "Giesdorf"])}
DN = {"P2": "Prüm 2", "SI": "Sinspelt", "GI": "Giesdorf", "EC": "Echtershausen", "WI": "Wiersdorf", "PZ": "Prümzurlay", "SE": "Seffern",
      "AL": "Alsdorf-Oberecken"}
UPS = {"EC": ["P2"], "WI": ["EC"], "PZ": ["WI", "SI"], "SE": ["GI"], "AL": ["SE"]}
ORDER = ("P2", "SI", "GI", "EC", "WI", "PZ", "SE", "AL")
RAINK = ("P2", "SI", "GI", "EC", "PZ", "SE", "AL")         # Gebiete mit Regen (Wiersdorf: Seeumfeld ohne)
VARIANTS = ("einzeln", "kette", "kette_regen_1", "kette_regen", "kette_ideal")
NORAIN = ("einzeln", "kette")
SWITCH = ("kette_regen", "kette_ideal", "kette_vorhersage")  # Varianten mit Umschalten Winter / Starkregen
FUTURE_RAIN = ("kette_ideal", "kette_vorhersage")            # Regen nach t0 aus frames (gemessen bzw. Vorhersage) statt 0
P_STORM, STORM_MONTHS, STORM_WIN_H, W_PRIOR = 4.0, (5, 9), 48.0, 1.0
HOLD_RULE = True                                            # kein Scheitel am Unterlieger ohne gemessenen Scheitel an den Oberliegern
HOLD_TAU_H = float(__import__("os").environ.get("PRUEM_HOLD_TAU", "inf"))   # Abklingzeit der Scheitel-Regel in h (inf: Vorhersage bleibt auf dem Höchstwert stehen)
# Abflussbeiwert mit Gebietsfeuchte aus dem Abfluss (lateral.py, "bfa", additiv) und seit 18.11.2026 mit der Bodenfeuchte als Faktor ("smf");
# PRUEM_SM=0 schaltet sie ab. Additiv ("sma") setzte die Anpassung das Gewicht stets auf 0; als Faktor (Schwelle 86-103 % nFK, steil) wurde die
# Güte der Oberläufe in allen 5 Prüfblöcken besser (NSE z. B. Giesdorf 0,02 -> 0,15, Sinspelt 0,35 -> 0,40 im Block 2004-07).
LAT_TERMS = ("bfa", "smf") if os.environ.get("PRUEM_SM", "1") == "1" else ("bfa",)
RAIN_EVENTS, RAIN_EVENT_MM = True, 25.0                     # Anpassung zusätzlich an Regenereignissen ohne Hochwasser (48-h-Gebietsregen ab 25 mm)
BASE_CACHE = os.environ.get("PRUEM_BASE")                    # Grundmodelle aus einem früheren Lauf übernehmen (siehe base_models)
# Faktor auf die Wellengeschwindigkeit c(Q) je Routing-Strecke: {Unterlieger: {Oberlieger: f}}, z. B. {"PZ": {"WI": 0.8}}. Leer oder 1 = wie
# angepasst. Gedacht für die im Ereignis gemessene Laufzeit des Scheitels (kommunale Pegel); derzeit nirgends gesetzt. Grenzen CELER_MIN..MAX.
CELER: dict[str, dict[str, float]] = {}
CELER_MIN, CELER_MAX = 0.5, 2.0
# Kurzfristkorrektur eines Oberliegers (kommunal_nowcast.py, nur live): Pegel -> Funktion, die dessen Vorhersage [Fälle, H] (Abfluss) anhebt.
# Wirkt nur, wenn der Pegel Oberlieger des Ziels ist; die Vorhersage des Pegels selbst hebt live.py an (leer = aus).
UP_LIFT: dict = {}
LIVE_DATA = os.environ.get("PRUEM_LIVE", "1") == "1"         # Pegel aus Downloads der Live-Seite mit ins Training (live_training.py, pkgdata)
LEVEL: dict[str, float] = {}                                # Niveaufaktor je Pegel auf die Simulation (leer = aus; der Live-Betrieb setzt ihn aus data/niveau.json)
# Live-Abflüsse kommen mit der heutigen Abflusskurve; die Modelle sind an den Abflüssen bis 2022 angepasst. Oberlieger und Gebietsfeuchte deshalb durch
# den Niveaufaktor teilen, bevor sie in das Modell des Unterliegers gehen (bis 22.11.2026 bekam Alsdorf-Oberecken Seffern 1,3-fach). PRUEM_LEVEL_IN=0: aus.
LEVEL_IN = LEVEL if os.environ.get("PRUEM_LEVEL_IN", "1") == "1" else {}
ML_NET = True                                               # lernender Abflussbeiwert (lateral "ml", ml_fit.py) nach der gemeinsamen Anpassung
REGIONAL_SAT = True                                         # gemeinsamer Sättigungsüberschuss und nichtlineare Speicher in allen Zwischengebieten
SAT_SPEC = dict(rs=(0.97, 0.0, 1.0), log_sev=(float(np.log(170.0)), float(np.log(10.0)), float(np.log(3000.0))), g_ev=(0.2, 0.0, 10.0),
                log_kf=(float(np.log(2.5)), float(np.log(0.25)), float(np.log(12.0))),
                log_oref=(float(np.log(4.0)), float(np.log(0.02)), float(np.log(50.0))), a_nl=(0.7, 0.0, 2.0))   # Start, untere, obere Grenze
SAT_TERMS = ("satx", "nlrx")
WTS_EXP_KEY = {"WI": 1.0}    # Wiersdorf (Stausee): absolute Fehler, damit die großen Wellen die Wellengeschwindigkeit bei hohem Abfluss bestimmen
WTS_EXP = 0.0                                               # Fehler je Fenster durch max(Scheitel, q95)^(1-g) * q95^g: g=0 relativ (jedes Fenster gleich), g>0 große Hochwasser zählen mehr
C0_SHIFT = False                                            # versucht, nicht übernommen: Modell über der Messung -> Sockel c0 dauerhaft abziehen (siehe _forecast)
PEAK_LAG, PEAK_DROP = 2, 0.01                               # Scheitel gemessen: Höchstwert >= 30 min zurück, jetzt >= 1 % darunter, über 30 min nicht gestiegen
A_H, H_H, LEAD_H = 48.0, 48.0, 72.0                       # Vorgeschichte, Horizont, Einschwingen Regen (h)
A, H, LEAD = int(A_H / DT), int(H_H / DT), int(LEAD_H / DT)
OFF = np.arange(int(-48 / DT), int(12 / DT) + 1)            # Ausgabezeiten relativ zum Scheitel des Zielpegels (Schritte)
BLOCKS = RF2.BLOCKS
_S: dict = {}


def chain(k: str) -> list[str]:
    """Alle Pegel, die für die Vorhersage von k gebraucht werden, in Rechenreihenfolge."""
    need, todo = set(), [k]
    while todo:
        x = todo.pop()
        need.add(x)
        todo += UPS.get(x, [])
    return [x for x in ORDER if x in need]


# ------------------------------------------------------------------ Modelle
def hw_model(dn: str, key: str, c0: float, lat: L.Lateral) -> R.Model:
    return R.Model(dn, [], c0, False, R.C_MAX, [lat])


def _dummy(d: R.Data) -> R.Data:
    d.X = {"_": np.zeros_like(d.Y)}
    return d


def fit_hw(key: str, d: R.Data, info: dict, max_nfev: int = 40) -> R.Model:
    """Regen-Abfluss-Modell eines Oberlaufs: c0 + Zwischengebietsantwort; zwei Startwerte, Zielwert wie routing.objective."""
    base = float(np.median(d.Y[:, :8].mean(axis=1)))
    best = None
    for i in (0, 1):
        m0 = hw_model(DN[key], key, base, RF2.lat_start(info, key, i).with_terms(LAT_TERMS))
        m, cost = R._run(m0, d, 0.03, max_nfev)
        if best is None or cost < best[1]:
            best = (m, cost)
    return best[0]


def rain_events(key: str, Q: pd.DataFrame, frame: pd.DataFrame, flood: np.ndarray) -> np.ndarray:
    """Regenereignisse ohne Hochwasser im Gebiet key: 48-h-Gebietsregen mindestens RAIN_EVENT_MM (Abstand 5 Tage), Anker = größter Abfluss am Pegel
    bis 48 h nach dem Regenmaximum, nicht innerhalb von 3,5 Tagen um einen Hochwasserscheitel. Damit lernt die Anpassung auch, wie wenig nach
    Trockenphasen abfließt (vorher nur Hochwasser: Abflussbeiwert zu hoch, siehe lateral.py, Erweiterung bf)."""
    from scipy.signal import find_peaks
    p = np.nan_to_num(np.asarray(frame["p"]))
    r48 = pd.Series(p).rolling(int(48 / DT), min_periods=1).sum().to_numpy() * DT
    pk, _ = find_peaks(r48, height=RAIN_EVENT_MM, distance=int(5 * 24 / DT))
    q = Q[DN[key]].to_numpy()
    out = []
    for t in pk:
        seg = q[t:t + int(48 / DT)]
        if not np.isfinite(seg).any():
            continue
        a = t + int(np.nanargmax(seg))
        if flood.size == 0 or np.min(np.abs(flood - a)) > int(3.5 * 24 / DT):
            out.append(a)
    out = np.array(out, int)
    return out[(out - int(60 / DT) - int(72 / DT) >= 0) & (out + int(24 / DT) < len(Q))]


def data_for(key: str, Q, W, frames) -> R.Data:
    """Anpassungsfenster: Hochwasser (Scheitel über dem 95-%-Quantil) und Regenereignisse ohne Hochwasser; Fehler je Fenster durch
    max(Scheitel, 95-%-Quantil) geteilt."""
    if key == "WI":
        d = R.make_data(Q, W, "Wiersdorf", ["Echtershausen"])
        yrs = pd.DatetimeIndex(d.times).year.to_numpy()
        d = d.subset((yrs >= 2004) & (yrs <= 2022))
        g = WTS_EXP_KEY.get(key, WTS_EXP)
        if g:
            d.wts = d.wts ** (1.0 - g) * float(np.min(d.wts)) ** g
        return d
    dn = DN[key]
    flood = R.find_events(Q, dn, pre_h=60, post_h=24)
    peaks = np.sort(np.r_[flood, rain_events(key, Q, frames[key], flood)]) if RAIN_EVENTS else flood
    floor = float(Q[dn].quantile(0.95)) if RAIN_EVENTS else 1.0
    if key in HW:
        d = _dummy(R.make_data(Q, W, dn, [], peaks=peaks, rain={key: frames[key]}, wts_floor=floor))
    else:
        g = RF.G[key]
        d = R.make_data(Q, W, g.dn, RF.ups(g), peaks=peaks, rain={key: frames[key]}, wts_floor=floor)
    g = WTS_EXP_KEY.get(key, WTS_EXP)
    if g:
        d.wts = d.wts ** (1.0 - g) * max(1.0, floor) ** g
    return d


WET_REF = {"P2": "Prüm 2", "SI": "Sinspelt", "GI": "Giesdorf", "EC": "Echtershausen", "PZ": "Prümzurlay", "SE": "Seffern", "AL": "Alsdorf-Oberecken"}


def wet_mq() -> dict[str, float]:
    """MQ der Bezugspegel für die Gebietsfeuchte (Kennwerte der HVZ, data/pegelkennwerte.json)."""
    kw = json.loads((HERE / "data" / "pegelkennwerte.json").read_text(encoding="utf-8"))["pegel"]
    return {k: float(kw[dn]["Q"]["MQ"]) for k, dn in WET_REF.items()}


SM_FILE = HERE / "data" / "bodenfeuchte" / "bf_0-60.csv"
SM_FALLBACK = 100.0                                         # % nFK, falls ein Wert fehlt (etwa Mittel 2001-2026); Rain.sm
SM_CURRENT: dict | None = None                              # Bodenfeuchte des laufenden Live-Rasters (live.build setzt sie; Regenfenster ohne Index)


def soil_moisture(index: pd.DatetimeIndex, table: pd.DataFrame | None = None) -> dict[str, np.ndarray] | None:
    """Bodenfeuchte je Regengebiet auf dem Raster index (MEZ): Tageswert des Vortags (live am Morgen verfügbar), höchstens 3 Tage fortgeschrieben."""
    if table is None:
        if not SM_FILE.exists():
            return None
        table = pd.read_csv(SM_FILE, index_col=0, parse_dates=True)
    day = pd.DatetimeIndex(index).normalize() - pd.Timedelta(days=1)
    t = table.reindex(pd.date_range(table.index.min(), max(table.index.max(), day.max()), freq="1D")).ffill(limit=3)
    return {k: t[k].reindex(day).to_numpy(float) for k in RAINK if k in t}


# Untergrenze der Gebietsfeuchte aus der Bodenfeuchte (bf_untergrenze.py): B_eff = max(B, typischer Grundabfluss bei dieser Bodenfeuchte im Winter,
# unteres Viertel 2004-2022). Anlass: Herbst 2026, Oberboden gesättigt, Grundwasser leer. Aus mit PRUEM_BFMIN=0.
BF_FLOOR_FILE = HERE / "data" / "bodenfeuchte" / "bf_untergrenze.json"
BF_FLOOR: dict[str, tuple[list, list]] = {}
if os.environ.get("PRUEM_BFMIN", "1") == "1" and BF_FLOOR_FILE.exists():
    BF_FLOOR = {k: (v["sm"], v["b"]) for k, v in json.loads(BF_FLOOR_FILE.read_text(encoding="utf-8"))["tabelle"].items()}


# Anhebung der Gebietsfeuchte bei niedrigem Grundabfluss (22.11.2026), nur in der Vorhersage, nicht in der Anpassung:
# B_eff = B + BF_LIFT * max(0, Bt - B), Bt = typischer Grundabfluss (Median von B im Winter 2004-2022) bei der Bodenfeuchte des Vortags
# (data/bodenfeuchte/bf_typisch.json, bf_untergrenze.py 0.5). Anlass: Bei nassem Boden und leerem Grundwasser (B < 0,7 MQ) lag der Anstieg im
# Modell über alle Regenereignisse im Median beim 0,80-Fachen, bei nassem Boden und B ab 1,5 MQ beim 0,94-Fachen. Die Anhebung greift nur unter
# Bt, die Modelle bleiben unverändert; im häufigen Fall (nass, B >= 1,5 MQ) ändert sich deshalb fast nichts. PRUEM_BFLIFT=0: aus.
BF_LIFT = float(os.environ.get("PRUEM_BFLIFT", "0.75"))
BT_FILE = HERE / "data" / "bodenfeuchte" / "bf_typisch.json"
BT_TAB: dict = {}
if BT_FILE.exists() and BF_LIFT > 0:
    BT_TAB = {k: (v["sm"], v["b"]) for k, v in json.loads(BT_FILE.read_text(encoding="utf-8"))["tabelle"].items()}


def bt_typ(k: str, sm) -> np.ndarray:
    """Typischer Grundabfluss / MQ bei der Bodenfeuchte (% nFK; fehlend = SM_FALLBACK)."""
    s = np.asarray(sm, float)
    return np.interp(np.where(np.isfinite(s), s, SM_FALLBACK), *BT_TAB[k])


def bf_floor(k: str, sm) -> np.ndarray:
    """Untergrenze von B für Gebiet k aus der Bodenfeuchte (% nFK; fehlend = SM_FALLBACK), linear zwischen den Stützstellen, außen konstant."""
    s = np.asarray(sm, float)
    return np.interp(np.where(np.isfinite(s), s, SM_FALLBACK), *BF_FLOOR[k])


def add_wetness(frames: dict, Q, sm_table: pd.DataFrame | None = None, sm: dict | None = None) -> dict:
    """Spalten bf (Gebietsfeuchte = Abfluss am Bezugspegel / MQ) und sm (Bodenfeuchte, soil_moisture) je Regengebiet; Q: DataFrame oder dict
    Pegelname -> Reihe auf demselben Raster; sm: fertige Reihen je Gebiet auf dem Raster (Live-Betrieb), sonst aus der Tabelle über Q.index."""
    mq = wet_mq()
    if sm is None and isinstance(Q, pd.DataFrame):
        sm = soil_moisture(Q.index, sm_table)
    elif sm is None and SM_CURRENT:                         # Live: Raster beginnen alle gleich, Länge wie die Abflussreihen
        n = len(np.asarray(next(iter(Q.values()))))
        sm = {k: v[:n] for k, v in SM_CURRENT.items()}
    for k in RAINK:
        q = np.asarray(Q[WET_REF[k]], float) / LEVEL_IN.get(k, 1.0)     # Gebietsfeuchte auf der Skala der Anpassung (MQ bis 2022)
        extra = dict(bf=q / mq[k])
        if sm is not None and k in sm:
            extra["sm"] = sm[k]
            if k in BF_FLOOR:
                extra["bf"] = np.maximum(extra["bf"], bf_floor(k, sm[k]))     # NaN im Abfluss bleibt NaN
            if k in BT_TAB:
                extra["bt"] = bt_typ(k, sm[k])
        if isinstance(frames[k], pd.DataFrame):
            frames[k] = frames[k].assign(**extra)
        else:
            frames[k] = dict(frames[k], **extra)
    return frames


def error_beta(m: R.Model, d: R.Data, spin_h: float = 12.0) -> np.ndarray:
    """Fehlerkorrektur je Vorlauf (1 bis H Schritte): Regressionssteigung e(t + L) auf e(t) über die Trainingsfenster, e normiert mit dem Scheitel."""
    e = (d.Y - m.route(d.X, d.P)) / d.wts[:, None]
    s0, T = int(spin_h / DT), e.shape[1]
    beta = np.empty(H)
    for L in range(1, H + 1):
        a, b = e[:, s0:T - L], e[:, s0 + L:]
        beta[L - 1] = (a * b).sum() / max((a * a).sum(), 1e-12)
    # Ein Fehler kann mit wachsendem Vorlauf nur verblassen: beta nicht steigend (bei langen Vorläufen wenige Wertepaare, sonst Ausreißer nach oben)
    return np.minimum.accumulate(np.clip(beta, 0.0, 1.0))


def storm_index(frames: dict, info: dict) -> dict[str, np.ndarray]:
    """Je Zielpegel: größter 1-h-Gebietsregen (mm/h, flächengewichtet über alle Gebiete der Kette) in den 48 h bis zum Zeitpunkt."""
    out = {}
    for k in ORDER:
        ks = [x for x in chain(k) if x in RAINK]
        a = np.array([info[x]["area_km2"] for x in ks])
        a = a / a.sum()
        p = sum(w * np.nan_to_num(np.asarray(frames[x]["p"])) for w, x in zip(a, ks))
        p1 = pd.Series(p).rolling(int(1 / DT), min_periods=1).mean()
        out[k] = p1.rolling(int(STORM_WIN_H / DT), min_periods=1).max().to_numpy()
    return out


def is_storm(k: str, t) -> np.ndarray:
    """Starkregenhochwasser zur Zeit t (Indizes im Abflussraster), nur aus Daten bis t."""
    t = np.asarray(t)
    mo = _S["month"][t]
    return (mo >= STORM_MONTHS[0]) & (mo <= STORM_MONTHS[1]) & (_S["pm"][k][t] >= P_STORM)


def rain_type(k: str) -> str:
    """Modell, das in den Regenvarianten für Pegel k genutzt wird."""
    return "hw" if k in HW else ("route_full" if k == "WI" else "rain_full")


def fit_set(block: int | None) -> dict:
    """Alle Modelle der Kette, angepasst ohne den Zeitblock (block=None: alle Ereignisse 2004-2022)."""
    Q, W, frames, info, start = _S["Q"], _S["W"], _S["frames"], _S["info"], _S["start"]
    out = {}
    for key in ORDER:
        d = data_for(key, Q, W, frames)
        if block is not None:
            yrs = pd.DatetimeIndex(d.times).year.to_numpy()
            test = (yrs >= BLOCKS[block][0]) & (yrs <= BLOCKS[block][1])
            near = (np.abs(d.peaks[:, None] - d.peaks[test][None, :]) <= int(3.5 * 24 / DT)).any(axis=1)
            d = d.subset(~test & ~near)
        if key in HW:
            ms = {"hw": fit_hw(key, d, info)}
        else:
            g = RF.G[key]
            ms = {"route_full": R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, start=start[key]["route_full"], cmax=g.cmax, max_nfev=25)[0]}
            if key != "WI":
                ms["rain_full"] = R.fit(g.dn, g.links, d, Q, dx_grid=RF.DX_GRID, start=start[key]["rain_full"], cmax=g.cmax, max_nfev=25)[0]
        for v, m in ms.items():
            m.beta = error_beta(m, d)
        # zwei Modellsätze: Winterhochwasser (Anpassung an den Winterereignissen) und Starkregenhochwasser (wenige Ereignisse, Strafterm zum Gesamtmodell)
        mt = rain_type(key)
        st = is_storm(key, d.peaks)
        for reg, sel in (("winter", ~st), ("storm", st)):
            m_all = ms[mt]
            if sel.sum() < 4:
                m = R.Model.from_dict(m_all.to_dict())
            else:
                ds = d.subset(sel)
                if key in HW:
                    m = R._run(m_all, ds, 0.03, 40, None if reg == "winter" else (m_all, W_PRIOR))[0]
                else:
                    g = RF.G[key]
                    m = R.fit(g.dn, g.links, ds, Q, dx_grid=RF.DX_GRID, start=m_all, cmax=g.cmax, max_nfev=25,
                              prior=None if reg == "winter" else (m_all, W_PRIOR))[0]
            m.beta = error_beta(m, d.subset(sel)) if sel.sum() >= 4 else m_all.beta
            ms[f"{mt}_{reg}"] = m
        out[key] = ms
    return out


def _training(key: str, block: int | None) -> R.Data:
    d = data_for(key, _S["Q"], _S["W"], _S["frames"])
    if block is not None:
        yrs = pd.DatetimeIndex(d.times).year.to_numpy()
        test = (yrs >= BLOCKS[block][0]) & (yrs <= BLOCKS[block][1])
        near = (np.abs(d.peaks[:, None] - d.peaks[test][None, :]) <= int(3.5 * 24 / DT)).any(axis=1)
        d = d.subset(~test & ~near)
    return d


def _lat_shared(lt, P: dict):
    lt2 = lt.with_terms(tuple(t for t in lt.terms if t not in ("int", "fast", "sat", "satx", "nlr", "nlrx")) + SAT_TERMS)
    for a, x in P.items():
        setattr(lt2, a, float(x))
    return lt2


def with_sat(models: dict, P: dict) -> dict:
    """Kopie der Modelle mit Sättigungsüberschuss und nichtlinearen Speichern (gemeinsame Parameter P) in jedem Zwischengebiet."""
    out = {}
    for k, ms in models.items():
        out[k] = {}
        for v, m in ms.items():
            m2 = R.Model.from_dict(m.to_dict())
            m2.beta = np.array(m.beta, float)
            if m2.lat:
                m2.lat = [_lat_shared(lt, P) for lt in m2.lat]
            out[k][v] = m2
    return out


def fit_regional_sat(models: dict, block: int | None, nstart: int = 2) -> tuple[dict, dict]:
    """Gemeinsame Parameter (SAT_SPEC: Sättigungsüberschuss und nichtlineare Speicher) an allen Regengebieten, Grundmodelle fest, ohne den Block;
    die ganze Zwischengebietsantwort wird je Auswertung neu gerechnet. Modelle mit den Bausteinen und neu bestimmter Fehlerkorrektur zurück,
    dazu die Parameter."""
    from scipy.optimize import least_squares
    stack = []
    for k in RAINK:
        d = _training(k, block)
        m = models[k][rain_type(k)]
        rest = m.route(d.X, d.P) - sum(L.response(d.P[lt.name], lt) for lt in m.lat)
        stack.append((d, rest, m.lat))
    names = list(SAT_SPEC)
    lo = np.array([v[1] for v in SAT_SPEC.values()])
    hi = np.array([v[2] for v in SAT_SPEC.values()])

    def resid(th):
        P = dict(zip(names, th))
        return np.concatenate([((d.Y - rest - sum(L.response(d.P[lt.name], _lat_shared(lt, P)) for lt in lats)) / d.wts[:, None])[d.mask]
                               for d, rest, lats in stack])
    best, rng = None, np.random.default_rng(2)
    for j in range(nstart):
        th0 = np.array([v[0] for v in SAT_SPEC.values()]) if j == 0 else lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
        r = least_squares(resid, np.clip(th0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=60)
        if best is None or r.cost < best.cost:
            best = r
    P = {n: float(x) for n, x in zip(names, best.x)}
    out = with_sat(models, P)
    for k in RAINK:                                                       # Fehlerkorrektur mit den Bausteinen neu (wie fit_set)
        d = _training(k, block)
        mt = rain_type(k)
        st = is_storm(k, d.peaks)
        out[k][mt].beta = error_beta(out[k][mt], d)
        for reg, sel in (("winter", ~st), ("storm", st)):
            out[k][f"{mt}_{reg}"].beta = error_beta(out[k][f"{mt}_{reg}"], d.subset(sel)) if sel.sum() >= 4 else out[k][mt].beta
    return out, P


# ------------------------------------------------------------------ Vorhersage
def rain_window(frame: pd.DataFrame | dict, t0: np.ndarray, ideal: bool) -> L.Rain:
    """Regenfenster [Fälle, LEAD + A + H + 1] ab t0 - A - LEAD. Ohne ideal: Regen nach t0 = 0, Vorregenindizes klingen ab."""
    ridx = t0[:, None] + np.arange(-A - LEAD, H + 1)[None, :]
    r = np.asarray(frame["p"])[ridx].copy()
    ap = np.asarray(frame["api"])[ridx].copy()
    aps = np.asarray(frame["api_s"])[ridx].copy()
    xc = [c for c in RF2.XCOLS if c in frame]
    ex = np.nan_to_num(np.stack([np.asarray(frame[c], dtype="float32")[ridx] for c in xc], axis=-1)) if xc else None
    if not ideal:
        j = LEAD + A
        k = np.arange(1, H + 1) * DT
        r[:, j + 1:] = 0.0
        if ex is not None:
            ex[:, j + 1:] = 0.0
        ap[:, j + 1:] = ap[:, j:j + 1] * np.exp(-k / RN.TAU_API_H)[None, :]
        aps[:, j + 1:] = aps[:, j:j + 1] * np.exp(-k / RN.TAU_API_S_H)[None, :]
    bf = np.nan_to_num(np.asarray(frame["bf"])[ridx[:, LEAD]]) if "bf" in frame else None   # Gebietsfeuchte zu Beginn des Abflussfensters (t0 - A)
    if bf is not None and BF_LIFT > 0 and "bt" in frame:          # Anhebung bei niedrigem Grundabfluss (nur Vorhersage, siehe BF_LIFT)
        ok = np.isfinite(np.asarray(frame["bf"], float)[ridx[:, LEAD]])   # ohne gemessenen Abfluss keine Anhebung (wie forecast_core.js)
        bt = np.nan_to_num(np.asarray(frame["bt"])[ridx[:, LEAD]])
        bf = np.where(ok, bf + BF_LIFT * np.maximum(bt - bf, 0.0), bf)
    # Bodenfeuchte, Vortag des Fensteranfangs; fehlt sie ganz, der Ersatzwert (nie ein Abbruch im Betrieb)
    sm = (np.nan_to_num(np.asarray(frame["sm"])[ridx[:, LEAD]], nan=SM_FALLBACK) if "sm" in frame else np.full(len(t0), SM_FALLBACK))
    return L.Rain(np.nan_to_num(r), np.nan_to_num(ap), LEAD, np.nan_to_num(aps), bf, ex, sm)


def to_json(models: dict) -> dict:
    return {k: {v: dict(m.to_dict(), beta=[round(float(x), 4) for x in m.beta]) for v, m in ms.items()} for k, ms in models.items()}


def from_json(d: dict) -> dict:
    out = {}
    for k, ms in d.items():
        out[k] = {}
        for v, md in ms.items():
            m = R.Model.from_dict(md)
            m.beta = np.array(md["beta"])
            out[k][v] = m
    return out


def forecast(models: dict, target: str, Qn: dict, frames: dict, t0: np.ndarray, variant: str, storm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Vorhersage aller Pegel der Kette zur Ausgabezeit t0 (Indizes). Rückgabe je Pegelschlüssel die Reihe [Fälle, A + H + 1]:
    Messung bis t0 (Spalte A), danach korrigierte Vorhersage. Bei den Varianten mit Umschalten bestimmt storm (je Fall) den Modellsatz;
    ohne Angabe aus is_storm(target, t0)."""
    if variant in SWITCH:
        st = is_storm(target, t0) if storm is None else np.asarray(storm, bool)
        out = None
        for reg, sel in (("winter", ~st), ("storm", st)):
            if not sel.any():
                continue
            S = _forecast(models, target, Qn, frames, t0, variant, "_" + reg)
            if out is None:
                out = {k: np.empty_like(v) for k, v in S.items()}
            for k, v in S.items():
                out[k][sel] = v[sel]
        return out
    return _forecast(models, target, Qn, frames, t0, variant, "")


def peak_passed(obs: np.ndarray) -> np.ndarray:
    """Scheitel gemessen (nur Werte bis t0 = Spalte A): Höchstwert der letzten 48 h liegt mindestens 30 min zurück, der Abfluss liegt jetzt mindestens
    1 % darunter und ist in den letzten 30 min nicht gestiegen, und der Höchstwert ist ein echter Anstieg (mindestens 20 % über dem kleinsten Wert davor).
    Verzögerung nach dem echten Scheitel über alle Ereignisse der Oberlieger im Median 45 min (Wiersdorf 30 min), 90 % innerhalb von 2,5 h."""
    past = obs[:, :A + 1]
    jm = np.argmax(past, axis=1)
    m = past[np.arange(len(past)), jm]
    lo = np.where(np.arange(A + 1)[None, :] <= jm[:, None], past, np.inf).min(axis=1)
    return (jm <= A - PEAK_LAG) & (obs[:, A] <= (1 - PEAK_DROP) * m) & (obs[:, A] <= obs[:, A - PEAK_LAG]) & (m >= 1.2 * np.maximum(lo, 1e-6))


def hold_after_max(fc: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Vorhersage [Fälle, H]: steigt sie über den jetzigen Wert q0, fällt sie nach ihrem Höchstwert höchstens so schnell wie eine Abklingkurve
    mit HOLD_TAU_H Stunden (Höchstwert * exp(-Zeit seit dem Höchstwert / HOLD_TAU_H)); unendlich = bleibt stehen (strenge Regel), 0 = Regel aus."""
    if HOLD_TAU_H <= 0:
        return fc
    jm = np.argmax(fc, axis=1)
    mx = fc[np.arange(len(fc)), jm]
    dtm = (np.arange(fc.shape[1])[None, :] - jm[:, None]) * DT
    mask = (mx > q0)[:, None] & (dtm > 0)
    floor = mx[:, None] * (np.exp(-np.maximum(dtm, 0) / HOLD_TAU_H) if np.isfinite(HOLD_TAU_H) else 1.0)
    return np.where(mask, np.maximum(fc, floor), fc)


def celer_model(m, k: str):
    """Modell m des Unterliegers k mit den Faktoren aus CELER auf c(Q) (log c + log f an allen Stützstellen); ohne Faktor unverändert."""
    f = CELER.get(k) or {}
    up_code = {v: kk for kk, v in DN.items()}
    fac = {i: min(max(float(f.get(up_code.get(lk.up, ""), 1.0)), CELER_MIN), CELER_MAX) for i, lk in enumerate(m.links)}
    if all(v == 1.0 for v in fac.values()):
        return m
    import copy
    m2 = copy.copy(m)
    m2.links = [copy.copy(lk) for lk in m.links]
    for i, lk in enumerate(m2.links):
        lk.lc = np.asarray(lk.lc, float) + np.log(fac[i])
    return m2


# Verhältniskorrektur (Versuch 18.11.2026, Sturm: Sinspelt stieg 3-4-mal stärker als das Modell): Anstieg gemessen / Anstieg Modell über den
# niedrigsten Stand der letzten RATIO_WIN_H; der künftige Anstieg des Modells wird mit diesem Faktor (gedämpft, begrenzt) multipliziert, die
# übliche Fehlerkorrektur gilt danach für den Rest. Nur wenn beide deutlich gestiegen sind.
RATIO_UPDATE = os.environ.get("PRUEM_RATIO", "0") == "1"
RATIO_KEYS = tuple(os.environ.get("PRUEM_RATIO_KEYS", "P2,SI,GI").split(","))
RATIO_WIN_H, RATIO_MIN_MQ, RATIO_LO, RATIO_HI, RATIO_W = 24.0, 0.3, 0.5, 4.0, float(os.environ.get("PRUEM_RATIO_W", "0.7"))


def ratio_update(obs: np.ndarray, sim: np.ndarray, k: str) -> tuple[np.ndarray, np.ndarray]:
    mq = _S.get("mq_ratio", {}).get(k)
    if mq is None:
        mq = float(np.nanmean(_S["Q"][DN[k]])) if "Q" in _S else 1.0
        _S.setdefault("mq_ratio", {})[k] = mq
    w = int(RATIO_WIN_H / DT)
    base = np.nanmin(obs[:, A - w:A + 1], axis=1)
    ro, rs = obs[:, A] - base, sim[:, A] - base
    ok = (ro >= RATIO_MIN_MQ * mq) & (rs >= 0.1 * mq)
    f = np.where(ok, np.clip(ro / np.maximum(rs, 1e-6), RATIO_LO, RATIO_HI), 1.0)
    f = 1.0 + RATIO_W * (f - 1.0)
    sim2 = sim.copy()
    sim2[:, A:] = base[:, None] + f[:, None] * (sim[:, A:] - base[:, None])
    return sim2, obs[:, A] - sim2[:, A]


def _forecast(models: dict, target: str, Qn: dict, frames: dict, t0: np.ndarray, variant: str, suffix: str) -> dict[str, np.ndarray]:
    idx = t0[:, None] + np.arange(-A, H + 1)[None, :]
    S = {}
    for k in chain(target):
        obs = Qn[DN[k]][idx]
        if k in HW:
            if variant in NORAIN:
                S[k] = np.concatenate([obs[:, :A + 1], np.repeat(obs[:, A:A + 1], H, axis=1)], axis=1)
                continue
            m = models[k]["hw" + suffix]
            sim = m.route({"_": np.zeros_like(obs)}, {k: rain_window(frames[k], t0, variant in FUTURE_RAIN)})
        else:
            if variant == "einzeln":
                X = {}
                for u in UPS[k]:
                    v = Qn[DN[u]][idx].copy() / LEVEL_IN.get(u, 1.0)
                    v[:, A + 1:] = v[:, A:A + 1]
                    X[DN[u]] = v
            else:
                X = {DN[u]: S[u] / LEVEL_IN.get(u, 1.0) for u in UPS[k]}   # Oberlieger auf die Skala der Anpassung (heutige Abflusskurve / Niveaufaktor)
            m = celer_model(models[k][("route_full" if (variant in NORAIN or k == "WI") else "rain_full") + suffix], k)
            P = {k: rain_window(frames[k], t0, variant in FUTURE_RAIN)} if m.lat else None
            sim = m.route(X, P)
        if k in LEVEL:                                         # Niveaufaktor (niveau.py): heutige Wasserstand-Abfluss-Beziehung des Pegels
            sim = sim * LEVEL[k]
        err0 = obs[:, A] - sim[:, A]
        if C0_SHIFT:
            # Versuch (nicht übernommen): Liegt das Modell über der Messung, den Sockel c0 dauerhaft abziehen. Bei Niedrigwasser steigt die Vorhersage
            # sonst ohne Regen zum Modellsockel an (Prümzurlay im trockenen Herbst 2026: von 1,4 auf etwa 4 m³/s); mit Abzug fällt sie dafür viel zu
            # schnell (auf 0,4 m³/s in 48 h), weil dem Modell ein langsamer Grundwasseranteil fehlt. Die Seiten zeigen deshalb die Vorhersage nach
            # der Fehlerstatistik (Median der Restfehler je Zustand, Abflussstufe und Vorlauf), die diese Drift ausgleicht.
            shift = np.maximum(np.minimum(err0, 0.0), -max(m.c0, 0.0))
            sim = sim + shift[:, None]
            err0 = err0 - shift
        if RATIO_UPDATE and k in RATIO_KEYS and variant not in NORAIN:
            sim, err0 = ratio_update(obs, sim, k)
        fc = np.maximum(sim[:, A + 1:] + err0[:, None] * m.beta[None, :], 0.0)
        if HOLD_RULE and variant not in FUTURE_RAIN and k in UPS:      # nur ohne Regenvorhersage (hold_test.py, 14.11.2026)
            passed = np.ones(len(t0), bool)
            for u in UPS[k]:
                passed &= peak_passed(Qn[DN[u]][idx])
            if (~passed).any():
                fc[~passed] = hold_after_max(fc[~passed], obs[~passed, A])
        if k != target and k in UP_LIFT:
            fc = UP_LIFT[k](fc)
        S[k] = np.concatenate([obs[:, :A + 1], fc], axis=1)
    return S


def events_for(target: str, Q: pd.DataFrame, frames: dict, years: tuple[int, int], need_rain: bool, W: pd.DataFrame | None = None) -> np.ndarray:
    """Scheitel des Zielpegels (über dem 95-%-Quantil) in den Jahren, mit vollständigen Pegel- und (falls verlangt) Regenfenstern."""
    pk = R.find_events(Q, DN[target], pre_h=60, post_h=24)
    yrs = Q.index[pk].year
    pk = pk[(yrs >= years[0]) & (yrs <= years[1])]
    lo, hi = OFF.min() - A, OFF.max() + H
    keep = []
    for p in pk:
        if p + lo - LEAD < 0 or p + hi >= len(Q):
            continue
        ok = all(np.isfinite(Q[DN[k]].to_numpy()[p + lo:p + hi + 1]).all() for k in chain(target))
        if ok and W is not None and "WI" in chain(target):          # Wiersdorf bei 250 cm abgeschnitten (ältere Hochwasser): Eingang und Prüfung unbrauchbar
            ok = not R.clipped_window(W["Wiersdorf"].to_numpy()[p + lo:p + hi + 1][None, :], Q["Wiersdorf"].to_numpy()[p + lo:p + hi + 1][None, :])[0]
        if ok and need_rain:
            ok = all(np.isfinite(frames[k][c].to_numpy()[p + lo - LEAD:p + hi + 1]).all() for k in chain(target) if k in RAINK for c in ("p", "api", "api_s"))
        if ok:
            keep.append(p)
    return np.array(keep, dtype=int)


def evaluate(models: dict, target: str, events: np.ndarray, variants, Qn: dict, frames: dict) -> dict:
    """Summen je Variante und Vorlauf (normiert mit dem Ereignisscheitel) und Scheitelvorhersage je Ereignis und Ausgabezeit vor dem Scheitel."""
    acc = {v: dict(sse=np.zeros(H), so=np.zeros(H), so2=np.zeros(H), n=0) for v in variants + ("pers",)}
    acc_st = {v: dict(sse=np.zeros(H), so=np.zeros(H), so2=np.zeros(H), n=0) for v in variants + ("pers",)}
    peaks = {v: [] for v in variants + ("pers",)}
    labels = [bool(is_storm(target, [p])[0]) if "pm" in _S else False for p in events]
    for p, lab in zip(events, labels):
        t0 = p + OFF
        pk = float(Qn[DN[target]][p])
        obs = Qn[DN[target]][t0[:, None] + np.arange(1, H + 1)[None, :]] / pk
        res = {v: forecast(models, target, Qn, frames, t0, v)[target][:, A + 1:] / pk for v in variants}
        res["pers"] = np.repeat(Qn[DN[target]][t0][:, None] / pk, H, axis=1)
        before = OFF <= 0                                        # Ausgabe vor dem Scheitel: Scheitel liegt im Horizont
        for v, f in res.items():
            for a in ([acc[v], acc_st[v]] if lab else [acc[v]]):
                a["sse"] += ((f - obs) ** 2).sum(axis=0)
                a["so"] += obs.sum(axis=0)
                a["so2"] += (obs ** 2).sum(axis=0)
                a["n"] += len(t0)
            # Scheitel der Vorhersage im Horizont gegen den gemessenen Scheitel, je Ausgabezeit vor dem Scheitel
            fb = f[before]
            peaks[v].append(np.stack([fb.max(axis=1), (fb.argmax(axis=1) + 1 + OFF[before]) * DT], axis=1))
    return dict(acc=acc, acc_storm=acc_st, storm=labels, peaks={v: np.stack(x) for v, x in peaks.items()} if events.size else {})


def base_models(b: int | None) -> tuple[dict, dict | None]:
    """Grundmodelle eines Satzes (ohne Netz) und gemeinsame Sättigungsgrößen. Mit PRUEM_BASE=<chain_models.json eines früheren Laufs ohne Netz>
    werden sie von dort übernommen statt neu angepasst (die Anpassung ist deterministisch; spart etwa 40 min, wenn nur das Netz neu gerechnet wird)."""
    if BASE_CACHE:
        doc = json.loads(Path(BASE_CACHE).read_text(encoding="utf-8"))
        key = "alle" if b is None else str(b)
        assert not any("ml" in lt["terms"] for ms in doc["sets"][key].values() for m in ms.values() for lt in m.get("lat") or []), "PRUEM_BASE enthält schon ein Netz"
        return from_json(doc["sets"][key]), doc["sat"][key]
    models = fit_set(b)
    sat = None
    if REGIONAL_SAT:
        models, sat = fit_regional_sat(models, b)
    return models, sat


def block_job(b: int) -> dict:
    t = time.time()
    models, sat = base_models(b)
    ml = None
    if ML_NET and sat:
        import ml_fit
        models, ml = ml_fit.fit_models(models, sat, b, f"Block {BLOCKS[b][0]}-{BLOCKS[b][1]}")
    Q, frames = _S["Q"], _S["frames"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    out = dict(block=b, models=to_json(models), eval={}, sat=sat, ml=ml)
    for target in ORDER:
        ev = events_for(target, Q, frames, BLOCKS[b], need_rain=True, W=_S["W"])
        out["eval"][target] = dict(events=[str(Q.index[p])[:16] for p in ev], **evaluate(models, target, ev, VARIANTS, Qn, frames))
    print(f"[Block {BLOCKS[b][0]}-{BLOCKS[b][1]}] fertig ({time.time() - t:.0f} s)", flush=True)
    return out


def global_job(_: int) -> dict:
    """Modelle aus allen Ereignissen 2004-2022; damit die großen Hochwasser vor 2004 (ohne Regen) als unabhängige Prüfung."""
    t = time.time()
    models, sat = base_models(None)
    ml = None
    if ML_NET and sat:
        import ml_fit
        models, ml = ml_fit.fit_models(models, sat, None, "alle")
    Q, frames = _S["Q"], _S["frames"]
    Qn = {c: Q[c].to_numpy() for c in Q.columns}
    out = dict(block=None, models=to_json(models), eval={}, sat=sat, ml=ml)
    for target in ORDER:
        ev = events_for(target, Q, frames, (1989, 2003), need_rain=False, W=_S["W"])
        out["eval"][target] = dict(events=[str(Q.index[p])[:16] for p in ev], **evaluate(models, target, ev, NORAIN, Qn, frames))
    print(f"[alle 2004-2022, Prüfung 1989-2003] fertig ({time.time() - t:.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------ Auswertung und Ausgaben
def summarize(parts: list[dict], variants, tag: str, storm_only: bool = False) -> tuple[list[dict], list[dict]]:
    """Güte je Vorlauf und Scheiteltreffer je Ausgabezeit vor dem Scheitel; storm_only: nur Starkregenereignisse."""
    skill, peak = [], []
    ak = "acc_storm" if storm_only else "acc"
    for target in ORDER:
        xs = [p["eval"][target] for p in parts if p["eval"].get(target) and p["eval"][target]["events"]]
        if storm_only:
            xs = [x for x in xs if any(x["storm"])]
        if not xs:
            continue
        sel = [np.array(x["storm"], bool) if storm_only else np.ones(len(x["events"]), bool) for x in xs]
        n_ev = int(sum(s_.sum() for s_ in sel))
        tot = {v: {k: sum(x[ak][v][k] for x in xs) for k in ("sse", "so", "so2", "n")} for v in variants + ("pers",)}
        for v in variants + ("pers",):
            a = tot[v]
            sst = a["so2"] - a["so"] ** 2 / a["n"]
            for h in range(H):
                skill.append(dict(set=tag, target=target, dn=DN[target], variant=v, lead_h=(h + 1) * DT, nse=float(1 - a["sse"][h] / sst[h]),
                                  skill=float(1 - a["sse"][h] / tot["pers"]["sse"][h]), nrmse=float(np.sqrt(a["sse"][h] / a["n"])), events=n_ev,
                                  cases=int(a["n"])))
            pk = np.concatenate([x["peaks"][v][s_] for x, s_ in zip(xs, sel)])   # [Ereignisse, Ausgabezeiten vor dem Scheitel, (Verhältnis, Zeit)]
            before = OFF[OFF <= 0]
            for j, o in enumerate(before):
                r, dtp = pk[:, j, 0], pk[:, j, 1]
                peak.append(dict(set=tag, target=target, dn=DN[target], variant=v, before_h=float(-o * DT), ratio_med=float(np.median(r)),
                                 ratio_q25=float(np.quantile(r, 0.25)), ratio_q75=float(np.quantile(r, 0.75)), abs_err_med=float(np.median(np.abs(r - 1))),
                                 within20=float(np.mean(np.abs(r - 1) <= 0.2)), dt_med_h=float(np.median(dtp)),
                                 hit=float(np.mean((np.abs(r - 1) <= 0.15) & (np.abs(dtp) <= 2.0))), events=int(len(r))))
    return skill, peak


def flood_windows(Q: pd.DataFrame, W: pd.DataFrame, frames: dict, sets: dict) -> dict:
    """Je großes Hochwasser (Prümzurlay W > 400 cm) die Fenster für die Vorhersage in der Karte: Abfluss aller acht Pegel ab Scheitel Prümzurlay
    - 96 h bis + 60 h, Regen (15 min) ab - 168 h, Vorregenindizes am Anfang; dazu der Modellsatz (Block ohne das Hochwasser, sonst alle 2004-2022)."""
    import floods as F
    out = []
    a0, b0 = int(96 / DT), int(60 / DT)
    r0 = a0 + LEAD
    for fl in F.find_floods(W["Prümzurlay"], 400):
        ip = int(Q.index.get_loc(fl["t_peak"]))
        yr = fl["t_peak"].year
        blk = [i for i, (y0, y1) in enumerate(BLOCKS) if y0 <= yr <= y1]
        rain_ok = all(np.isfinite(frames[k][c].to_numpy()[ip - r0:ip + b0 + 1]).all() for k in RAINK for c in ("p", "api", "api_s"))
        rec = dict(id=f"{fl['t_peak']:%Y-%m-%d}", t_peak=str(fl["t_peak"]), t_start=str(Q.index[ip - a0]), i_peak=a0,
                   models=str(blk[0]) if blk else "alle", rain=bool(rain_ok),
                   q={DN[k]: [None if not np.isfinite(v) else round(float(v), 3) for v in Q[DN[k]].to_numpy()[ip - a0:ip + b0 + 1]] for k in ORDER})
        if rain_ok:
            rec["storm"] = {k: "".join("1" if x else "0" for x in is_storm(k, np.arange(ip - a0, ip + b0 + 1))) for k in ORDER}
            rec["rain_start"] = str(Q.index[ip - r0])
            rec["p"] = {k: [round(float(v), 4) for v in frames[k]["p"].to_numpy()[ip - r0:ip + b0 + 1]] for k in RAINK}
            rec["api0"] = {k: [float(frames[k]["api"].to_numpy()[ip - r0 - 1]), float(frames[k]["api_s"].to_numpy()[ip - r0 - 1])] for k in RAINK}
            if all("sm" in frames[k] for k in RAINK):              # Bodenfeuchte auf dem Abflussraster (Wert am Fensteranfang i - A)
                rec["sm"] = {k: [round(float(np.nan_to_num(v, nan=SM_FALLBACK)), 2) for v in np.asarray(frames[k]["sm"])[ip - a0:ip + b0 + 1]] for k in RAINK}
        out.append(rec)
    return out


def setup_state() -> None:
    """Daten, Regenfenster, Startmodelle und Starkregen-Index in _S (für run() und extreme_fit.py)."""
    Q, W = pkgdata.load_all("Q", live=LIVE_DATA), pkgdata.load_all("W", live=LIVE_DATA)
    frames, info = RF2.rain_frames(Q.index, 1.0, extra=HW)
    frames = add_wetness(frames, Q)
    rp = json.loads((HERE / "data" / "rain_params.json").read_text(encoding="utf-8"))
    bp = json.loads((HERE / "data" / "routing_params.json").read_text(encoding="utf-8"))
    start = {k: {"route_full": R.Model.from_dict(rp[k]["models"]["route_full"]), "rain_full": R.Model.from_dict(rp[k]["models"]["rain_full"])}
             for k in ("EC", "PZ", "SE", "AL")}
    for k in start:                                          # Zwischengebiete mit Gebietsfeuchte
        m = start[k]["rain_full"]
        m.lat = [lt.with_terms(tuple(sorted(set(lt.terms) | set(LAT_TERMS)))) for lt in m.lat]
    start["WI"] = {"route_full": R.Model.from_dict(bp["WI"]["full"])}
    _S.update(Q=Q, W=W, frames=frames, info=info, start=start, pm=storm_index(frames, info), month=Q.index.month.to_numpy())


def run(jobs: int) -> None:
    import multiprocessing as mp
    t_all = time.time()
    setup_state()
    Q, W, frames, info = _S["Q"], _S["W"], _S["frames"], _S["info"]
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        res = list(pool.imap_unordered(lambda_job, [("g", 0)] + [("b", b) for b in range(len(BLOCKS))]))
    blocks = sorted([r for r in res if r["block"] is not None], key=lambda r: r["block"])
    glob = next(r for r in res if r["block"] is None)
    s1, p1 = summarize(blocks, VARIANTS, "cv_2004_2022")
    s3, p3 = summarize(blocks, VARIANTS, "cv_starkregen", storm_only=True)
    s2, p2 = summarize([glob], NORAIN, "alt_1989_2003")
    pd.DataFrame(s1 + s3 + s2).to_csv(HERE / "data" / "chain_skill.csv", index=False, float_format="%.4g")
    pd.DataFrame(p1 + p3 + p2).to_csv(HERE / "data" / "chain_peak.csv", index=False, float_format="%.4g")
    ev = {r["block"] if r["block"] is not None else "alt": {t: x["events"] for t, x in r["eval"].items()} for r in res}
    sets = {str(r["block"]): r["models"] for r in blocks}
    sets["alle"] = glob["models"]
    sat = {str(r["block"]) if r["block"] is not None else "alle": r.get("sat") for r in res}
    mlinfo = {str(r["block"]) if r["block"] is not None else "alle": r.get("ml") for r in res}
    (HERE / "data" / "chain_models.json").write_text(json.dumps(dict(sets=sets, events=ev, sat=sat, ml=mlinfo, hw={k: dict(**v, area_km2=info[k]["area_km2"],
                                                                                                        weights=info[k]["weights"]) for k, v in HW.items()},
                                                                       consts=dict(A_H=A_H, H_H=H_H, LEAD_H=LEAD_H, WET={k: [WET_REF[k], v] for k, v in wet_mq().items()}, BFMIN={k: [list(v[0]), list(v[1])] for k, v in BF_FLOOR.items()}, BTYP={k: [list(v[0]), list(v[1])] for k, v in BT_TAB.items()}, BFLIFT=BF_LIFT,
                                                                                   TAU_API_H=RN.TAU_API_H, TAU_API_S_H=RN.TAU_API_S_H,
                                                                                   P_STORM=P_STORM, STORM_MONTHS=list(STORM_MONTHS), STORM_WIN_H=STORM_WIN_H)),
                                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "data" / "chain_floods.json").write_text(json.dumps(flood_windows(Q, W, frames, sets), ensure_ascii=False, separators=(",", ":")),
                                                     encoding="utf-8")
    print(f"fertig in {time.time() - t_all:.0f} s", flush=True)


def lambda_job(x):
    kind, b = x
    return global_job(b) if kind == "g" else block_job(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    import chain_fc                                   # über das Modul starten: ml_fit importiert chain_fc und braucht denselben Zustand (_S)
    chain_fc.run(a.jobs)
