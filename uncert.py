"""Unsicherheit der Kettenvorhersage aus kreuzvalidierten Fehlern: Band, Überschreitungswahrscheinlichkeiten, Scheitel mit Spanne.

  python uncert.py archive [--jobs 2]   Vorhersagearchiv: jede Stunde eine Ausgabe über die ganzen Jahre (nicht nur um Hochwasser), je Pegel
                                         kette_regen und kette 2004-2022 mit den Modellen ohne den jeweiligen Block, kette 1989-2003 mit den
                                         Modellen aus 2004-2022 (unabhängige Prüfung). Ablage im Zwischenordner (groß, jederzeit neu rechenbar).
  python uncert.py tables               Fehlertabellen, Prüfung (verschachtelt: Tabellen ohne den geprüften Block) und data/uncert_tables.json.

Fehler r = ln((Messung + d) / (Vorhersage + d)) mit d = 10 % von MQ je Pegel und Vorlauf. Die Quantile von r hängen vom Zustand zur
Ausgabezeit ab (Klassen siehe state_class). Band: Vorhersage * exp(Quantil). Überschreitung eines Schwellenwerts innerhalb eines Zeitraums: die
Fehler sind über die Vorläufe stark gekoppelt, deshalb Rang über alle Vorläufe gleich (ein Fall liegt durchgehend z. B. im 80-%-Quantil);
Wahrscheinlichkeit = größter Anteil über die Vorläufe bis zum Zeitraumende. Prüfung auf Zuverlässigkeit und, falls nötig, empirische Nachkalibrierung.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import pkgdata
import rain_fit as RF2
import routing as R

HERE = Path(__file__).parent
SCR = Path("/tmp/claude-0/-home-claude/4bd0f43c-2dd9-566e-bd5b-bb61c4bf2100/scratchpad/uncert")
STEP = 4                                                       # Ausgabe jede Stunde
LEADS = np.array([1, 2, 3] + list(range(4, C.H + 1, 4)))         # gespeicherte Vorläufe (Schritte): 15, 30, 45 min, dann stündlich bis 48 h
CHUNK = 3000
PEAK_WIN = 48                                                    # Scheitel der Welle: bis 12 h nach dem vorhergesagten Scheitel
_S: dict = {}


def setup() -> None:
    Q, W = pkgdata.load_all("Q", live=C.LIVE_DATA), pkgdata.load_all("W", live=C.LIVE_DATA)
    frames, info = RF2.rain_frames(Q.index, 1.0, extra=C.HW)
    frames = C.add_wetness(frames, Q)
    C._S.update(Q=Q, W=W, frames=frames, info=info, pm=C.storm_index(frames, info), month=Q.index.month.to_numpy())
    rain = {}
    for k in C.ORDER:                                          # Gebietsregen der Kette (flächengewichtet, mm/h)
        ks = [x for x in C.chain(k) if x in C.RAINK]
        a = np.array([info[x]["area_km2"] for x in ks])
        a = a / a.sum()
        rain[k] = sum(w * np.nan_to_num(np.asarray(frames[x]["p"])) for w, x in zip(a, ks))
    _S.update(Q=Q, W=W, frames=frames, Qn={c: Q[c].to_numpy() for c in Q.columns}, rain=rain,
              sets=json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"])


def _ok_windows(bad: np.ndarray, lo: int, hi: int, t0: np.ndarray) -> np.ndarray:
    """Fenster [t0 + lo, t0 + hi] ohne schlechte Werte (kumulierte Summe)."""
    cs = np.r_[0, np.cumsum(bad)]
    return cs[t0 + hi + 1] - cs[t0 + lo] == 0


def issue_times(target: str, years: tuple[int, int], need_rain: bool) -> np.ndarray:
    Q, frames, W = _S["Q"], _S["frames"], _S["W"]
    yrs = Q.index.year.to_numpy()
    n = len(Q)
    t0 = np.arange(C.A + C.LEAD, n - C.H - 1, STEP)
    t0 = t0[(yrs[t0] >= years[0]) & (yrs[t0] <= years[1])]
    ok = np.ones(len(t0), bool)
    for k in C.chain(target):
        ok &= _ok_windows(~np.isfinite(_S["Qn"][C.DN[k]]), -C.A, C.H, t0)
        if need_rain and k in C.RAINK:
            for c in ("p", "api", "api_s"):
                ok &= _ok_windows(~np.isfinite(frames[k][c].to_numpy()), -C.A - C.LEAD, C.H, t0)
    if "WI" in C.chain(target):                                 # Wiersdorf bei 250 cm abgeschnitten (ältere Hochwasser)
        w, q = W["Wiersdorf"].to_numpy(), _S["Qn"]["Wiersdorf"]
        clip = (w >= 249) & (w <= 250) & (q >= 136.0) & (q <= 137.3)
        ok &= _ok_windows(clip, -C.A, C.H, t0)
    return t0[ok]


def archive_one(models: dict, target: str, t0: np.ndarray, variant: str) -> dict:
    Qn = _S["Qn"]
    q = Qn[C.DN[target]]
    out = {x: [] for x in ("fc", "obs", "obsmax", "fcmax", "fcarg", "obspk", "obsarg", "obspkw", "obsargw")}
    for i in range(0, len(t0), CHUNK):
        tt = t0[i:i + CHUNK]
        fc = C.forecast(models, target, Qn, _S["frames"], tt, variant)[target][:, C.A + 1:]
        obs = q[tt[:, None] + np.arange(1, C.H + 1)[None, :]]
        out["fc"].append(fc[:, LEADS - 1].astype(np.float32))
        out["obs"].append(obs[:, LEADS - 1].astype(np.float32))
        out["obsmax"].append(np.maximum.accumulate(obs, axis=1)[:, LEADS - 1].astype(np.float32))
        out["fcmax"].append(fc.max(axis=1).astype(np.float32))
        out["fcarg"].append((fc.argmax(axis=1) + 1).astype(np.int16))
        out["obspk"].append(obs.max(axis=1).astype(np.float32))
        out["obsarg"].append((obs.argmax(axis=1) + 1).astype(np.int16))
        # Scheitel der laufenden Welle: höchster Messwert bis 12 h nach dem vorhergesagten Scheitel (spätere Wellen aus neuem Regen zählen nicht)
        ow = np.where(np.arange(C.H)[None, :] <= (fc.argmax(axis=1) + PEAK_WIN)[:, None], obs, -np.inf)
        out["obspkw"].append(ow.max(axis=1).astype(np.float32))
        out["obsargw"].append((ow.argmax(axis=1) + 1).astype(np.int16))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(features_at(target, t0))
    return res


def features_at(target: str, t0: np.ndarray) -> dict:
    """Zustandsmerkmale zur Ausgabezeit (nur Daten bis t0)."""
    Qn = _S["Qn"]
    q = Qn[C.DN[target]]
    idx = t0[:, None] + np.arange(-C.A, 1)[None, :]
    up_open = np.zeros(len(t0), bool)
    for u in C.UPS.get(target, []):
        for j in range(0, len(t0), CHUNK):
            up_open[j:j + CHUNK] |= ~C.peak_passed(Qn[C.DN[u]][idx[j:j + CHUNK]])
    tgt_passed = np.concatenate([C.peak_passed(q[idx[j:j + CHUNK]]) for j in range(0, len(t0), CHUNK)]) if len(t0) else np.zeros(0, bool)
    r = _S["rain"][target]
    cs = np.r_[0, np.cumsum(r)] * C.DT
    hp, hf = heavy_flags(r, t0)
    return dict(t0=t0.astype(np.int32), q0=q[t0].astype(np.float32), q1h=q[t0 - 4].astype(np.float32), q6h=q[t0 - 24].astype(np.float32),
                storm=C.is_storm(target, t0), up_open=up_open, tgt_passed=tgt_passed,
                rain6=(cs[t0 + 1] - cs[t0 + 1 - 24]).astype(np.float32), rain24=(cs[t0 + 1] - cs[t0 + 1 - 96]).astype(np.float32), hp=hp, hf=hf)


def heavy_flags(r: np.ndarray, t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starkregenlage je Ausgabezeit aus dem Gebietsregen r (mm/h je 15 min): hp nur Regen bis t0 (24 h davor), hf auch die 24 h danach.
    Größte 24-h-Summe bzw. 3-h-Summe, deren Ende im Fenster liegt."""
    cs = np.r_[0, np.cumsum(np.nan_to_num(np.asarray(r, float)))] * C.DT
    n = len(r)
    end = np.arange(n)
    s24 = cs[end + 1] - cs[np.maximum(end + 1 - 96, 0)]
    s3 = cs[end + 1] - cs[np.maximum(end + 1 - 12, 0)]
    hit = (s24 >= HEAVY_24_MM) | (s3 >= HEAVY_3_MM)
    ch = np.r_[0, np.cumsum(hit)]
    t0 = np.asarray(t0)
    hp = ch[np.minimum(t0 + 1, n)] - ch[np.clip(t0 + 1 - 96, 0, n)] > 0
    hf = ch[np.minimum(t0 + 1 + 96, n)] - ch[np.clip(t0 + 1 - 96, 0, n)] > 0
    return hp, hf


def job(spec: tuple) -> str:
    variant, key, years, need_rain = spec
    models = C.from_json(_S["sets"][key])
    t = time.time()
    for target in C.ORDER:
        t0 = issue_times(target, years, need_rain)
        res = archive_one(models, target, t0, variant)
        res["block"] = np.full(len(t0), -1 if key == "alle" else int(key), np.int8)
        np.savez_compressed(SCR / f"{variant}_{key}_{target}.npz", **res)
    return f"{variant} {key} {years}: {time.time() - t:.0f} s"


VARIANTS_ARCH = ["kette_regen", "kette", "kette_vorhersage"]


def archive(jobs: int) -> None:
    import multiprocessing as mp
    SCR.mkdir(parents=True, exist_ok=True)
    setup()
    specs = [(v, str(b), C.BLOCKS[b], True) for v in VARIANTS_ARCH for b in range(len(C.BLOCKS))]
    if "kette" in VARIANTS_ARCH:
        specs.append(("kette", "alle", (1989, 2003), False))
    with mp.get_context("fork").Pool(jobs) as pool:
        for msg in pool.imap_unordered(job, specs):
            print(msg, flush=True)


def load(variant: str, target: str, keys) -> dict:
    parts = [dict(np.load(SCR / f"{variant}_{k}_{target}.npz")) for k in keys]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ------------------------------------------------------------------ Fehlermodell
LEV = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
STATES = ("ruhig", "Anstieg", "Welle unterwegs", "fallend")
LEVELS = ("unter 2 MQ", "2 bis 8 MQ", "ab 8 MQ")              # größerer Wert aus jetzt und Vorhersage-Höchstwert, in MQ (8 MQ etwa HQ2/2)
LEVEL_EDGES = (2.0, 8.0)
WET_EDGE = 0.3                                                  # jetzt unter 0,3 MQ: trocken (Vorhersagen nach Trockenphasen deutlich zu hoch)
WETS = ("trocken", "nicht trocken")
NSUB = len(LEVELS) * len(WETS)                                  # Unterstufe = Stufe * 2 + (0 trocken, 1 sonst)
BASE_CLASSES = tuple(f"{s}, {lv}, {w}" for s in STATES for lv in LEVELS for w in WETS)
NB = len(BASE_CLASSES)                                          # 24 Grundklassen; Klasse + NB = dieselbe in einer Starkregenlage
CLASSES = BASE_CLASSES + tuple(f"{c}, Starkregenlage" for c in BASE_CLASSES)
# Starkregenlage (seit 13.11.2026): Gebietsregen der Kette in 24 h mindestens 40 mm oder in 3 h mindestens 20 mm, im Fenster 24 h vor bis zur
# Ausgabezeit (ohne Regenvorhersage) bzw. 24 h vor bis 24 h nach der Ausgabezeit (mit Regenvorhersage; im Archiv der gemessene Regen = perfekte
# Vorhersage). 2004-2022 gab es 24 solche Lagen, 3 davon brachten HQ10; die Fehler dieser Fälle sind eine eigene Tabelle, über alle Pegel
# zusammengefasst (relativer Fehler), weil je Pegel nur eine Handvoll Lagen vorliegt. Das gemessene Hochwasser (etwa „über HQ10“) kann kein
# Auswahlmerkmal sein: Zur Ausgabezeit weiß man es nicht.
HEAVY_24_MM, HEAVY_3_MM = 40.0, 20.0
MIN_CASES = 200
PEAK_BUCKETS = np.array([0.0, 3.0, 6.0, 12.0, 48.0])           # Vorlauf bis zum vorhergesagten Scheitel (h)
PEAK_LEV = np.array([0.1, 0.5, 0.9])
PEAK_RISE, PEAK_MIN_MQ = 1.02, 2.0


def kennwerte() -> dict:
    return json.loads((HERE / "data" / "pegelkennwerte.json").read_text(encoding="utf-8"))["pegel"]


def offset(target: str) -> float:
    return 0.1 * kennwerte()[C.DN[target]]["Q"]["MQ"]


def state_class(a: dict, target: str, use_rain: bool = True) -> np.ndarray:
    """Zustand zur Ausgabezeit (nur Daten bis t0):
       3 fallend          Scheitel am Pegel gemessen und kein Oberlieger steigt noch
       2 Welle unterwegs  Scheitel an allen Oberliegern gemessen, am Pegel noch nicht (nur Pegel mit Oberliegern)
       1 Anstieg          Ereignis läuft: Regen der letzten 24 h >= 10 mm, oder Abfluss über 6 h um mehr als 5 % gestiegen, oder über 3 MQ
       0 ruhig            sonst.
    Ohne Regen (Variante kette, ältere Hochwasser) zählt nur der Abfluss."""
    mq = kennwerte()[C.DN[target]]["Q"]["MQ"]
    up = a["up_open"] if target in C.UPS else np.zeros(len(a["q0"]), bool)
    active = (a["q0"] > 1.05 * a["q6h"]) | (a["q0"] > 3 * mq)
    if use_rain:
        active |= a["rain24"] >= 10
    cls = np.where(active, 1, 0)
    if target in C.UPS:
        cls = np.where(~up & ~a["tgt_passed"], 2, cls)
    cls = np.where(a["tgt_passed"] & ~up, 3, cls)
    return cls.astype(np.int8)


def level_class(a: dict, target: str) -> np.ndarray:
    """Unterstufe = Abflussstufe * 2 + Feuchte. Abflussstufe aus dem größeren Wert von jetzt und Vorhersage-Höchstwert (in MQ): 0 unter 2, 1 von 2
    bis 8, 2 ab 8 MQ (nach dem Pinball-Verlust besser als die Stufe nur aus dem jetzigen Abfluss oder keine Stufe). Feuchte: 0, wenn der Abfluss jetzt
    unter 0,3 MQ liegt (trocken; dort sind die Vorhersagen deutlich zu hoch, bei Prümzurlay nach 48 h im Median um den Faktor 2,3), sonst 1."""
    mq = kennwerte()[C.DN[target]]["Q"]["MQ"]
    lv = np.digitize(np.maximum(a["q0"], a["fcmax"]) / mq, LEVEL_EDGES)
    return (lv * len(WETS) + (a["q0"] >= WET_EDGE * mq)).astype(np.int8)


def heavy(a: dict, use_rain: bool = True, future: bool = False) -> np.ndarray:
    """Starkregenlage je Fall: ohne Regen nie, mit Regen bis zur Ausgabezeit aus dem Regen davor, mit Regenvorhersage auch aus dem danach."""
    if not use_rain or "hp" not in a:
        return np.zeros(len(a["q0"]), bool)
    return np.asarray(a["hf"] if future else a["hp"], bool)


def klass(a: dict, target: str, use_rain: bool = True, future: bool = False) -> np.ndarray:
    """Klasse für die Fehlertabellen: Zustand * 6 + Unterstufe, in einer Starkregenlage + NB."""
    base = state_class(a, target, use_rain) * NSUB + level_class(a, target)
    return (base + NB * heavy(a, use_rain, future)).astype(np.int8)


def errors(a: dict, target: str) -> np.ndarray:
    d = offset(target)
    return np.log((a["obs"].astype(float) + d) / (np.maximum(a["fc"].astype(float), 0) + d))


def quant_table(r: np.ndarray, cls: np.ndarray, heavy_T: np.ndarray | None = None) -> np.ndarray:
    """Quantile [Klasse, Vorlauf, Stufe] für alle 2 * NB Klassen. Grundklassen aus den Fällen ohne Starkregenlage dieses Pegels; Starkregenklassen
    aus heavy_T (über alle Pegel zusammengefasst, heavy_table), sonst wie die Grundklasse."""
    base = _quant_base(r[cls < NB], cls[cls < NB])
    return np.concatenate([base, heavy_T if heavy_T is not None else base], axis=0)


def heavy_table(rs: list[np.ndarray], clss: list[np.ndarray]) -> np.ndarray | None:
    """Starkregentabelle [NB, Vorlauf, Stufe] aus den Starkregenfällen aller Pegel (Fehler relativ, daher zusammenfassbar); Stufung wie bei den
    Grundklassen (Klasse, sonst Unterstufe, sonst Abflussstufe, sonst alle Starkregenfälle). None bei weniger als MIN_CASES Fällen."""
    r = np.concatenate([x[c >= NB] for x, c in zip(rs, clss)])
    c = np.concatenate([c[c >= NB] for c in clss]) - NB
    if len(r) < MIN_CASES:
        return None
    return _quant_base(r, c)


def _quant_base(r: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Quantile [NB, Vorlauf, Stufe]; Klassen mit zu wenigen Fällen bekommen die Quantile derselben Unterstufe über alle Zustände, sonst derselben
    Abflussstufe, sonst aller Fälle. Über die Vorläufe leicht geglättet (gleitendes Mittel über 3 gespeicherte Vorläufe ab 2 h) und je Vorlauf
    aufsteigend."""
    pooled = np.quantile(r, LEV, axis=0).T
    by_level = []
    for lv in range(len(LEVELS)):
        s = (cls % NSUB) // len(WETS) == lv
        by_level.append(np.quantile(r[s], LEV, axis=0).T if s.sum() >= MIN_CASES else pooled)
    by_sub = []
    for sub in range(NSUB):
        s = cls % NSUB == sub
        by_sub.append(np.quantile(r[s], LEV, axis=0).T if s.sum() >= MIN_CASES else by_level[sub // len(WETS)])
    out = np.empty((NB, r.shape[1], len(LEV)))
    for c in range(NB):
        s = cls == c
        out[c] = np.quantile(r[s], LEV, axis=0).T if s.sum() >= MIN_CASES else by_sub[c % NSUB]
    sm = out.copy()
    for j in range(5, r.shape[1] - 1):
        sm[:, j] = out[:, j - 1:j + 2].mean(axis=1)
    return np.sort(sm, axis=2)


def cdf_fast(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """P(r <= x) aus Quantilen q[..., Stufe] für x [Fälle, Vorläufe]: zwischen den Stufen linear, außerhalb exponentielle Ränder (stetig,
    F(q05) = 0,05, Abfall so, dass F(q10) = 0,1 auf derselben Exponentialkurve liegt; oben entsprechend)."""
    x = np.asarray(x, float)
    k = (x[..., None] >= q).sum(axis=-1)                       # Anzahl Stufen <= x
    kk = np.clip(k, 1, len(LEV) - 1)
    qa = np.take_along_axis(q, (kk - 1)[..., None], -1)[..., 0]
    qb = np.take_along_axis(q, kk[..., None], -1)[..., 0]
    la, lb = LEV[kk - 1], LEV[kk]
    mid = la + (lb - la) * np.clip((x - qa) / np.maximum(qb - qa, 1e-9), 0, 1)
    q05, q10, q90, q95 = q[..., 0], q[..., 1], q[..., -2], q[..., -1]
    s_lo = np.maximum((q10 - q05) / np.log(2), 1e-3)
    s_hi = np.maximum((q95 - q90) / np.log(2), 1e-3)
    lo = 0.05 * np.exp(np.minimum(x - q05, 0) / s_lo)
    hi = 1 - 0.05 * np.exp(-np.maximum(x - q95, 0) / s_hi)
    return np.where(x < q05, lo, np.where(x > q95, hi, mid))


def exceed_prob(fc: np.ndarray, q: np.ndarray, thr: float, d: float) -> np.ndarray:
    """P(Schwelle bis zum Vorlauf überschritten) [Fälle, Vorläufe]: Rang über die Vorläufe gekoppelt -> laufendes Maximum der Einzelwahrscheinlichkeiten."""
    x = np.log((thr + d) / (np.maximum(fc, 0) + d))
    p = 1 - cdf_fast(x, q)
    return np.maximum.accumulate(p, axis=1)


def peak_cases(a: dict, target: str, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scheitelvorhersage: Welle unterwegs (an allen Oberliegern ein Scheitel gemessen, am Pegel noch nicht), Vorhersage steigt (Höchstwert mehr als
    2 % über jetzt), Ereignis (Höchstwert mindestens 2 MQ), Scheitel vor dem Horizontende. Gemessener Scheitel: höchster Wert bis 12 h nach dem
    vorhergesagten (eine spätere Welle aus neuem Regen ist nicht dieser Scheitel).
    Rückgabe Auswahl, Fach (Vorlauf bis zum vorhergesagten Scheitel), ln(gemessen/vorhergesagt) der Scheitelhöhe, Zeitfehler (h, gemessen - vorhergesagt)."""
    d = offset(target)
    mq = kennwerte()[C.DN[target]]["Q"]["MQ"]
    sel = ((cls % NB) // NSUB == 2) & (a["fcmax"] > PEAK_RISE * a["q0"]) & (a["fcmax"] >= PEAK_MIN_MQ * mq) & (a["fcarg"] < C.H - 4)
    lead = a["fcarg"] * C.DT
    b = np.clip(np.searchsorted(PEAK_BUCKETS, lead, side="left") - 1, 0, len(PEAK_BUCKETS) - 2)
    lnr = np.log((a["obspkw"] + d) / (a["fcmax"] + d))
    dt = (a["obsargw"].astype(float) - a["fcarg"]) * C.DT
    return sel, b, lnr, dt


def peak_table(a: dict, target: str, cls: np.ndarray) -> dict:
    sel, b, lnr, dt = peak_cases(a, target, cls)
    if sel.sum() < 30:                                        # Oberläufe (kein Oberlieger) oder zu wenige Fälle: keine Scheiteltabelle
        return None
    out = dict(lnr=[], dt=[], n=[])
    for k in range(len(PEAK_BUCKETS) - 1):
        s = sel & (b == k)
        for extra in ((max(k - 1, 0), k + 1 if k == 0 else k), (k - 1, k + 1)):   # zu wenige Fälle: erst mit dem kürzeren Nachbarfach, dann
            if s.sum() >= 30:                                                     # mit beiden, sonst alle zusammen
                break
            s = sel & (b >= extra[0]) & (b <= extra[1])
        if s.sum() < 30:
            s = sel
        out["lnr"].append(np.quantile(lnr[s], PEAK_LEV).tolist())
        out["dt"].append(np.quantile(dt[s], PEAK_LEV).tolist())
        out["n"].append(int((sel & (b == k)).sum()))
    return out


WARN: dict[str, str] = {}      # Warnschwelle je Pegel, abweichend vom Standard (z. B. {"PZ": "HQ20"}); Vorgabe 13.11.2026: überall einheitlich HQ10
WARN_DEFAULT = "HQ10"
WARN_LEVELS = (0.1, 0.5)       # Warnstufen: P(Warnschwelle in 48 h) ab 10 % „möglich“, ab 50 % „Warnung“
WARN_MERGE_H = 12              # Unterbrechungen der Warnphase bis 12 h zählen zur selben Phase


def warn_name(target: str) -> str:
    return WARN.get(target, WARN_DEFAULT)


def thresholds(target: str) -> dict:
    """Schwellen in m³/s: halbes HQ2, Meldehöhe (falls vergeben), HQ2 (Prüfung der Methode an kleineren Hochwassern), HQ10, HQ20 (Warnschwellen)."""
    k = kennwerte()[C.DN[target]]
    rc = json.loads((HERE / "data" / "rating_curves.json").read_text(encoding="utf-8"))["pegel"][C.DN[target]]
    out = {"HQ2/2": 0.5 * k["Q"]["HQ2"]}
    if rc.get("q_meldehoehe"):
        out["Meldehöhe"] = rc["q_meldehoehe"]
    out["HQ2"] = k["Q"]["HQ2"]
    out["HQ10"] = k["Q"]["HQ10"]
    out["HQ20"] = k["Q"]["HQ20"]
    return out


WINDOWS_H = (6, 12, 24, 48)


def evaluate(train: dict, test: dict, target: str, use_rain: bool = True, future: bool = False, heavy_T: np.ndarray | None = None) -> dict:
    """Band, Überschreitung und Scheitel für die Testfälle mit Tabellen aus den Trainingsfällen (Starkregentabelle heavy_T aus allen Pegeln)."""
    d = offset(target)
    ctr, cte = klass(train, target, use_rain, future), klass(test, target, use_rain, future)
    T = quant_table(errors(train, target), ctr, heavy_T)
    PT = peak_table(train, target, ctr)
    r = errors(test, target)
    q = T[cte]
    res = dict(cls=cte, cover80=(r >= q[..., 1]) & (r <= q[..., -2]), cover90=(r >= q[..., 0]) & (r <= q[..., -1]),
               below50=r <= q[..., 5], hw_obs=np.maximum(test["q0"], test["obspk"]) >= 0.5 * kennwerte()[C.DN[target]]["Q"]["HQ2"], exceed={})
    for name, thr in thresholds(target).items():
        P = exceed_prob(test["fc"].astype(float), q, thr, d)
        for w in WINDOWS_H:
            j = int(np.where(LEADS == w * 4)[0][0])
            res["exceed"][(name, w)] = dict(p=P[:, j].astype(np.float32), y=test["obsmax"][:, j] > thr, ok=test["q0"] <= thr,
                                            det=np.maximum.accumulate(test["fc"], axis=1)[:, j] > thr)
    sel, b, lnr, dt = peak_cases(test, target, cte)
    if PT is None:
        z = np.zeros(len(sel), bool)
        res["peak"] = dict(sel=z, bucket=b, in_h=z, in_t=z, err_med=lnr * 0, dt_med=dt * 0, err_raw=lnr * 0, dt_raw=dt * 0)
        return res
    lo_r = np.array([PT["lnr"][k][0] for k in b]); hi_r = np.array([PT["lnr"][k][2] for k in b]); md_r = np.array([PT["lnr"][k][1] for k in b])
    lo_t = np.array([PT["dt"][k][0] for k in b]); hi_t = np.array([PT["dt"][k][2] for k in b]); md_t = np.array([PT["dt"][k][1] for k in b])
    res["peak"] = dict(sel=sel, bucket=b, in_h=(lnr >= lo_r) & (lnr <= hi_r), in_t=(dt >= lo_t) & (dt <= hi_t),
                       err_med=np.exp(lnr - md_r) - 1, dt_med=dt - md_t, err_raw=np.exp(lnr) - 1, dt_raw=dt)
    return res


def _cat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _cat_res(rs: list[dict]) -> dict:
    out = {k: np.concatenate([r[k] for r in rs]) for k in ("cls", "cover80", "cover90", "below50", "hw_obs")}
    out["exceed"] = {key: {k: np.concatenate([r["exceed"][key][k] for r in rs]) for k in rs[0]["exceed"][key]} for key in rs[0]["exceed"]}
    out["peak"] = {k: np.concatenate([r["peak"][k] for r in rs]) for k in rs[0]["peak"]}
    out["t0"] = np.concatenate([r["t0"] for r in rs])
    return out


def warning_times(t0: np.ndarray, p: np.ndarray, q_obs: np.ndarray, thr: float, level: float = 0.5) -> tuple[list[float], int, int]:
    """Ereignisweise: Überschreitungsbeginn (erste Stunde über der Schwelle nach mindestens 48 h darunter); Vorwarnzeit = Abstand zur ersten
    Ausgabezeit, ab der P(Überschreitung in 48 h) bis zum Beginn ununterbrochen >= level war (0, wenn nicht vorher gewarnt).
    Fehlalarme: Warnphasen (zusammenhängend P >= level, Abfluss unter der Schwelle), nach deren Beginn binnen 48 h keine Überschreitung kam."""
    pos = {int(t): i for i, t in enumerate(t0)}
    above = np.nan_to_num(q_obs, nan=-1.0) > thr
    cs = np.r_[0, np.cumsum(above)]
    lo, hi = int(t0.min()) + 192, int(t0.max())
    cand = np.arange(lo, hi)
    starts = cand[above[cand] & (cs[cand] - cs[cand - 192] == 0)]
    lead = []
    for s in starts:
        h0 = s - 1 - (s - 1 - int(t0[0])) % 4                 # letzte stündliche Ausgabe vor s
        if h0 not in pos:                                     # keine Ausgabe (Datenlücke): nicht werten
            continue
        t, w = h0, 0.0
        while t in pos and p[pos[t]] >= level and s - t <= 192:
            w = (s - t) * C.DT
            t -= 4
        lead.append(w)
    alarm = (p >= level) & ~above[t0]
    fa, ep = 0, 0
    i = 0
    while i < len(t0):
        if alarm[i] and (i == 0 or not alarm[i - 1] or t0[i] - t0[i - 1] != 4):
            ep += 1
            if not above[t0[i]:t0[i] + 193].any():
                fa += 1
        i += 1
    return lead, ep, fa


def warn_events(t0: np.ndarray, p: np.ndarray, q_obs: np.ndarray, thr: float, times: pd.DatetimeIndex) -> tuple[list[dict], list[dict]]:
    """Ereignis für Ereignis (für seltene Schwellen wie HQ10/HQ20, bei denen Mittelwerte nichts aussagen):
    Ereignisse = Überschreitungsbeginn nach mindestens 48 h darunter, mit Scheitel (72 h), höchster Wahrscheinlichkeit in den 48 h davor und
    Vorwarnzeit je Warnstufe (ununterbrochen über der Stufe bis zum Beginn, wie warning_times).
    Warnphasen je Stufe: Zeiten mit P >= Stufe bei Abfluss unter der Schwelle, Lücken bis WARN_MERGE_H zusammengefasst; Fehlalarm, wenn binnen 48 h
    nach Phasenbeginn keine Überschreitung kam. Zu jeder Phase der höchste gemessene Abfluss in den 48 h danach (zeigt knappe Fehlalarme)."""
    pos = {int(t): i for i, t in enumerate(t0)}
    qo = np.nan_to_num(q_obs, nan=-1.0)
    above = qo > thr
    cs = np.r_[0, np.cumsum(above)]
    lo, hi = int(t0.min()) + 192, int(t0.max())
    cand = np.arange(lo, hi)
    starts = cand[above[cand] & (cs[cand] - cs[cand - 192] == 0)]
    ev = []
    for s in starts:
        h0 = s - 1 - (s - 1 - int(t0[0])) % 4
        if h0 not in pos:
            continue
        before = [p[pos[t]] for t in range(h0, s - 193, -4) if t in pos]
        rec = dict(beginn=str(times[s]), scheitel_q=round(float(qo[s:s + 289].max()), 1), p_max_48h=round(float(max(before)) if before else np.nan, 3))
        win = [(s - t) * C.DT for t in range(h0, s - 193, -4) if t in pos]
        pw = [p[pos[t]] for t in range(h0, s - 193, -4) if t in pos]
        for lev in WARN_LEVELS:
            t, w = h0, 0.0
            while t in pos and p[pos[t]] >= lev and s - t <= 192:
                w = (s - t) * C.DT
                t -= 4
            rec[f"vorwarn_{int(lev * 100)}_h"] = w
            hit = [h for h, x in zip(win, pw) if x >= lev]                # auch unterbrochen: erste und letzte Ankündigung in den 48 h davor
            rec[f"erste_{int(lev * 100)}_h"] = max(hit) if hit else 0.0
            rec[f"letzte_{int(lev * 100)}_h"] = min(hit) if hit else None
        ev.append(rec)
    ph = []
    for lev in WARN_LEVELS:
        alarm = (p >= lev) & ~above[t0]
        idx = np.where(alarm)[0]
        if not len(idx):
            continue
        groups, g = [], [idx[0]]
        for a, b in zip(idx[:-1], idx[1:]):
            if t0[b] - t0[a] <= WARN_MERGE_H * 4:
                g.append(b)
            else:
                groups.append(g)
                g = [b]
        groups.append(g)
        for g in groups:
            ts, te = int(t0[g[0]]), int(t0[g[-1]])
            hit = bool(above[ts:ts + 193].any())
            ph.append(dict(stufe=lev, beginn=str(times[ts]), ende=str(times[te]), dauer_h=(te - ts) * C.DT + 1, p_max=round(float(p[g].max()), 3),
                           treffer=hit, max_q_48h=round(float(qo[ts:ts + 193].max()), 1), max_vs_schwelle=round(float(qo[ts:ts + 193].max() / thr), 2)))
    return ev, ph


VARIANTS_TAB = (("kette_regen", True, False), ("kette", False, False), ("kette_vorhersage", True, True))   # Variante, Regen, Regenvorhersage


def heavy_tables() -> dict:
    """Starkregentabellen je Variante mit Regen und je ausgelassenem Block (und alle), aus den Starkregenfällen aller Pegel."""
    out = {}
    nb = len(C.BLOCKS)
    for variant, use_rain, future in VARIANTS_TAB:
        if not use_rain or not all((SCR / f"{variant}_0_{t}.npz").exists() for t in C.ORDER):
            continue
        hv = {}
        for t in C.ORDER:
            for b in range(nb):
                a = load(variant, t, [str(b)])
                c = klass(a, t, use_rain, future)
                m = c >= NB
                hv[(t, b)] = (errors(a, t)[m], c[m])
        out[variant] = {}
        for key in [str(b) for b in range(nb)] + ["alle"]:
            sel = [v for (t, b), v in hv.items() if key == "alle" or b != int(key)]
            out[variant][key] = heavy_table([x[0] for x in sel], [x[1] for x in sel])
        n = sum(len(v[0]) for v in hv.values())
        print(f"Starkregentabellen {variant}: {n} Fälle über alle Pegel", flush=True)
    return out


def hq10_windows(target: str, t0: np.ndarray) -> np.ndarray:
    """Ausgabezeiten 48 h vor dem Überschreiten von HQ10 (nach mindestens 48 h darunter) bis zum Scheitel (72 h) - nur zur Prüfung, nachträglich."""
    q = np.nan_to_num(_S["Qn"][C.DN[target]], nan=-1.0)
    thr = kennwerte()[C.DN[target]]["Q"]["HQ10"]
    above = q > thr
    cs = np.r_[0, np.cumsum(above)]
    starts = [i for i in np.where(above)[0] if i >= 192 and cs[i] - cs[i - 192] == 0]
    m = np.zeros(len(t0), bool)
    for s0 in starts:
        pk = s0 + int(np.argmax(q[s0:s0 + 289]))
        m |= (t0 >= s0 - 192) & (t0 <= pk)
    return m


def tables() -> None:
    setup()
    Kp = kennwerte()
    HT = heavy_tables()
    rows_cov, rows_ex, rows_pk, rows_ev, rows_wev, rows_wph = [], [], [], [], [], []
    U = dict(leads_h=(LEADS * C.DT).tolist(), lev=LEV.tolist(), classes=list(CLASSES), peak_buckets_h=PEAK_BUCKETS.tolist(), peak_lev=PEAK_LEV.tolist(), peak_rise=PEAK_RISE, peak_min_mq=PEAK_MIN_MQ, level_edges=list(LEVEL_EDGES), wet_edge=WET_EDGE,
             offset={}, thresholds={}, warn={}, warn_levels=list(WARN_LEVELS), regen={}, kette={})
    hl = [1, 3, 6, 12, 24, 48]
    jl = [int(np.where(LEADS == h * 4)[0][0]) for h in hl]
    for target in C.ORDER:
        dn = C.DN[target]
        U["offset"][target] = offset(target)
        U.setdefault("mq", {})[target] = Kp[dn]["Q"]["MQ"]
        U["thresholds"][target] = thresholds(target)
        U["warn"][target] = dict(name=warn_name(target), q=thresholds(target)[warn_name(target)])
        for variant, use_rain, future in VARIANTS_TAB:
            if not (SCR / f"{variant}_0_{target}.npz").exists():
                continue
            ht = HT.get(variant, {})
            parts = [load(variant, target, [str(b)]) for b in range(len(C.BLOCKS))]
            res = []
            for b in range(len(parts)):                         # verschachtelt: Tabellen ohne den geprüften Block
                ev = evaluate(_cat([p for i, p in enumerate(parts) if i != b]), parts[b], target, use_rain, future, ht.get(str(b)))
                ev["t0"] = parts[b]["t0"]
                res.append(ev)
            sets = {"cv_2004_2022": _cat_res(res)}
            if variant == "kette":                              # unabhängig: Tabellen aus 2004-2022, geprüft an 1989-2003
                old = load("kette", target, ["alle"])
                ev = evaluate(_cat(parts), old, target, use_rain)
                ev["t0"] = old["t0"]
                sets["alt_1989_2003"] = _cat_res([ev])
            for tag, r in sets.items():
                groups = [("alle", np.ones(len(r["cls"]), bool))]
                groups += [("Starkregenlage", r["cls"] >= NB), ("keine Starkregenlage", r["cls"] < NB)]
                groups += [("HQ10-Hochwasser (48 h vor dem Überschreiten bis Scheitel, nachträglich ausgewählt)", hq10_windows(target, r["t0"])),
                           ("HQ10-Hochwasser, Starkregenlage", hq10_windows(target, r["t0"]) & (r["cls"] >= NB))]
                groups += [(f"Zustand {STATES[s]}", (r["cls"] % NB) // NSUB == s) for s in range(len(STATES))]
                groups += [(f"Stufe {LEVELS[v]}", (r["cls"] % NSUB) // len(WETS) == v) for v in range(len(LEVELS))]
                groups += [("trocken (jetzt unter 0,3 MQ)", r["cls"] % len(WETS) == 0)]
                groups += [(CLASSES[c], r["cls"] == c) for c in range(len(CLASSES))]
                groups += [("gemessen ab HQ2/2 (nachträglich ausgewählt)", r["hw_obs"])]
                for gname, s in groups:
                    if s.sum() < 50:
                        continue
                    rows_cov.append(dict(pegel=dn, variante=variant, pruefung=tag, klasse=gname, faelle=int(s.sum()),
                                         **{f"im80_{h}h": round(float(r["cover80"][s, j].mean()), 3) for h, j in zip(hl, jl)},
                                         **{f"unter50_{h}h": round(float(r["below50"][s, j].mean()), 3) for h, j in zip(hl, jl)}))
                for (name, w), e in r["exceed"].items():
                    ok = e["ok"]
                    p, y, det = e["p"][ok].astype(float), e["y"][ok], e["det"][ok]
                    yf, df = y.astype(float), det.astype(float)
                    if w == 48 and name == warn_name(target):    # Warnschwelle: Ereignis für Ereignis
                        evs, phs = warn_events(r["t0"][ok], p, _S["Qn"][dn], thresholds(target)[name], _S["Q"].index)
                        base_row = dict(pegel=dn, warnschwelle=name, schwelle_q=round(thresholds(target)[name], 1), variante=variant, pruefung=tag)
                        rows_wev += [{**base_row, **x} for x in evs]
                        rows_wph += [{**base_row, **x} for x in phs]
                        for lev in WARN_LEVELS:
                            lead, ep, fa = warning_times(r["t0"][ok], p, _S["Qn"][dn], thresholds(target)[name], level=lev)
                            pl = [x for x in phs if x["stufe"] == lev]
                            rows_ev.append(dict(pegel=dn, variante=variant, pruefung=tag, schwelle=name, stufe=lev, ereignisse=len(lead),
                                                vorwarn_med_h=float(np.median(lead)) if lead else None,
                                                ab3h=round(float(np.mean(np.array(lead) >= 3)), 2) if lead else None,
                                                ab6h=round(float(np.mean(np.array(lead) >= 6)), 2) if lead else None,
                                                nicht_gewarnt=int(np.sum(np.array(lead) == 0)), warnphasen=len(pl),
                                                fehlalarme=sum(not x["treffer"] for x in pl), warnschwelle=True))
                    if y.sum() < 5:
                        continue
                    base = y.mean()
                    bs = np.mean((p - yf) ** 2)
                    bins = np.array([0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
                    bi = np.digitize(p, bins) - 1
                    rel = {f"p{bins[k]:.2f}-{min(bins[k + 1], 1):.2f}": (round(float(y[bi == k].mean()), 3) if (bi == k).sum() >= 10 else None,
                                                                           int((bi == k).sum())) for k in range(len(bins) - 1)}
                    warn = p >= 0.5
                    rows_ex.append(dict(pegel=dn, variante=variant, pruefung=tag, schwelle=name, fenster_h=w, faelle=int(ok.sum()), ueberschr=int(y.sum()),
                                        brier=round(float(bs), 4), bss_klima=round(float(1 - bs / max(base * (1 - base), 1e-9)), 3),
                                        bss_determ=round(float(1 - bs / max(np.mean((df - yf) ** 2), 1e-9)), 3),
                                        treffer50=round(float(warn[y].mean()), 3), fehlalarm50=round(float(1 - y[warn].mean()), 3) if warn.any() else None,
                                        treffer_det=round(float(det[y].mean()), 3), fehlalarm_det=round(float(1 - y[det].mean()), 3) if det.any() else None,
                                        zuverl=json.dumps(rel)))
                    if w == 48 and name in ("Meldehöhe", "HQ2", "HQ2/2"):   # Prüfung der Methode an kleineren Hochwassern (keine Warnung)
                        lead, ep, fa = warning_times(r["t0"][ok], p, _S["Qn"][dn], thresholds(target)[name])
                        if lead:
                            rows_ev.append(dict(pegel=dn, variante=variant, pruefung=tag, schwelle=name, stufe=0.5, ereignisse=len(lead),
                                                vorwarn_med_h=float(np.median(lead)), ab3h=round(float(np.mean(np.array(lead) >= 3)), 2),
                                                ab6h=round(float(np.mean(np.array(lead) >= 6)), 2), nicht_gewarnt=int(np.sum(np.array(lead) == 0)),
                                                warnphasen=ep, fehlalarme=fa, warnschwelle=False))
                pk = r["peak"]
                s = pk["sel"]
                if s.sum() >= 20:
                    for k in range(-1, len(PEAK_BUCKETS) - 1):
                        ss = s if k < 0 else s & (pk["bucket"] == k)
                        if ss.sum() < 10:
                            continue
                        rows_pk.append(dict(pegel=dn, variante=variant, pruefung=tag,
                                            vorlauf="alle" if k < 0 else f"{PEAK_BUCKETS[k]:g}-{PEAK_BUCKETS[k + 1]:g} h", faelle=int(ss.sum()),
                                            hoehe_in80=round(float(pk["in_h"][ss].mean()), 3), zeit_in80=round(float(pk["in_t"][ss].mean()), 3),
                                            beide_in80=round(float((pk["in_h"] & pk["in_t"])[ss].mean()), 3),
                                            hoehe_fehler_med_roh=round(float(np.median(np.abs(pk["err_raw"][ss]))), 3),
                                            hoehe_fehler_med_korr=round(float(np.median(np.abs(pk["err_med"][ss]))), 3),
                                            zeit_fehler_med_roh_h=round(float(np.median(np.abs(pk["dt_raw"][ss]))), 2),
                                            zeit_fehler_med_korr_h=round(float(np.median(np.abs(pk["dt_med"][ss]))), 2)))
            # Tabellen für die Karte: Regen je ausgelassenem Block (für die Hochwasser der Karte) und alle; ohne Regen aus allen Blöcken 2004-2022
            store = U["regen"] if variant == "kette_regen" else U["kette"] if variant == "kette" else U.setdefault("vorhersage", {})
            keys = [str(b) for b in range(len(parts))] + ["alle"] if variant in ("kette_regen", "kette_vorhersage") else ["alle"]
            for key in keys:
                tr = _cat([p for i, p in enumerate(parts) if key == "alle" or i != int(key)])
                cls = klass(tr, target, use_rain, future)
                T = quant_table(errors(tr, target), cls, ht.get(key))
                store.setdefault(key, {})[target] = dict(q=np.round(T, 3).tolist(), peak=peak_table(tr, target, cls),
                                                         n=[int((cls == c).sum()) for c in range(len(CLASSES))])
        print(f"{dn} fertig", flush=True)
    pd.DataFrame(rows_cov).to_csv(HERE / "data" / "uncert_band.csv", index=False)
    pd.DataFrame(rows_ex).to_csv(HERE / "data" / "uncert_exceed.csv", index=False)
    pd.DataFrame(rows_pk).to_csv(HERE / "data" / "uncert_peak.csv", index=False)
    pd.DataFrame(rows_ev).to_csv(HERE / "data" / "uncert_warn.csv", index=False)
    pd.DataFrame(rows_wev).to_csv(HERE / "data" / "uncert_warn_events.csv", index=False)
    pd.DataFrame(rows_wph).to_csv(HERE / "data" / "uncert_warn_phases.csv", index=False)
    U["floods"] = flood_classes()
    (HERE / "data" / "uncert_tables.json").write_text(json.dumps(U, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def flood_classes() -> dict:
    """Je Hochwasserfenster der Karte, Pegel und Index (wie chain_fc.flood_windows) eine Hexadezimalziffer: Zustand (0-3) + 4 * Starkregenlage aus
    dem Regen bis zur Ausgabezeit + 8 * Starkregenlage einschließlich der 24 h danach (gemessener Regen = perfekte Vorhersage); ohne Regen nur der
    Zustand."""
    fl = json.loads((HERE / "data" / "chain_floods.json").read_text(encoding="utf-8"))
    Q = _S["Q"]
    out = {}
    for f in fl:
        i0 = int(Q.index.get_loc(pd.Timestamp(f["t_start"])))
        n = len(f["q"][C.DN["PZ"]])
        t0 = np.arange(i0, i0 + n)
        rec = {}
        for k in C.ORDER:
            ft = features_at(k, t0)
            st = state_class(ft, k, use_rain=bool(f["rain"]))
            code = st + (4 * ft["hp"] + 8 * ft["hf"] if f["rain"] else 0)
            rec[k] = "".join(format(int(c), "x") for c in code)
        out[f["id"]] = rec
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["archive", "tables"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--variants", default=None, help="Komma-Liste, z. B. kette_vorhersage (Standard: alle)")
    a = ap.parse_args()
    if a.variants:
        VARIANTS_ARCH[:] = a.variants.split(",")
    if a.cmd == "archive":
        archive(a.jobs)
    else:
        tables()
