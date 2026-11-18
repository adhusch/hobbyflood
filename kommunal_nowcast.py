"""Kurzfristkorrektur aus kommunalen Pegeln flussauf: die ersten Stunden der Vorhersage, ohne historische Trainingsdaten.

Steigt eine kommunale Messstelle flussauf, kommt der Anstieg nach der Laufzeit am Landespegel an. Wie stark, lernt das Verfahren im
laufenden Ereignis: Regression ohne Achsenabschnitt der Änderung am Landespegel über H Stunden auf die Änderung an der Messstelle über die
H Stunden davor, aus den letzten 24 Stunden (beta, begrenzt 0-3). Daraus der Stand am Landespegel in H Stunden:

    W(t0 + H) = W(t0) + beta * (W_kom(t0) - W_kom(t0 - H))

Liegt die Modelllinie zur Zeit t0 + H darunter, wird sie angehoben: bis t0 + H linear ansteigend, danach mit TAU_H Stunden abklingend
(wie die Fehlerkorrektur am Pegel). Nur nach oben und nur bei steigender Messstelle; fällt sie oder liegt das Modell höher, bleibt alles wie
es ist. Die Korrektur wird in Wasserstand gerechnet und über die Abflusskurve auf alle Läufe angewendet, Band und Wahrscheinlichkeiten
folgen.

Prüfung am Sturm 16.-18.11.2026 gegen die tatsächlich ausgegebenen Vorhersagen (Ausgaben stündlich, mittlerer Fehler in cm, in Klammern
nur bei Anstieg um mindestens 3 cm):
    Sinspelt (Neuerburg, 2 h):        2 h 1,5 -> 1,1 (4,2 -> 2,8)   4 h 2,8 -> 2,3 (6,4 -> 5,0)   12 h 7,9 -> 7,7
    Prüm 2 (Hermespand, 2 h):         2 h 1,7 -> 1,6 (3,2 -> 2,4)   4 h 3,0 -> 2,8 (5,2 -> 4,0)
    Seffern (Nimsreuland, 2 h):       2 h 1,2 -> 1,1 (2,0 -> 1,2)   4 h 1,9 -> 1,8 (2,8 -> 2,4)
    Giesdorf (Rommersheim, 1 h):      1 h 2,6 -> 2,3 (5,5 -> 3,0)   2 h 3,9 -> 3,8 (5,5 -> 4,6)
    Echtershausen (Kinzenburg, 3 h):  4 h 3,3 -> 3,1 (3,7 -> 3,4)   6 h 3,9 -> 3,4 (4,0 -> 3,4)
Nicht verwendet: Prümzurlay, Wiersdorf, Alsdorf-Oberecken (das Modell war dort in den ersten Stunden schon so gut oder besser).
Ein einziges Ereignis: die Werte sind ein Anhaltspunkt, keine Statistik. Deshalb vorsichtig: nur anheben, Freigabe nur bei gut
korreliertem, deutlichem Anstieg.

Ausfallsicher: fehlen die kommunalen Daten, sind sie veraltet oder passt etwas nicht, bleibt die Vorhersage unverändert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Landespegel -> (Nummer der Messstelle, Name, H in Stunden)
PAIRS = {"SI": ("1034232130", "Neuerburg", 2.0), "P2": ("1034232010", "Hermespand", 2.0), "SE": ("1034232250", "Nimsreuland", 2.0),
         "GI": ("1034232240", "Rommersheim", 1.0), "EC": ("1034232060", "Kinzenburg", 3.0)}
TAU_H = 6.0
WINDOW_H = 24.0
BETA_MAX = 3.0
MIN_PAIRS = 24                 # Viertelstundenpaare im Fenster
MIN_R = 0.5                    # Korrelation der Änderungen im Fenster
MIN_LEARN_CM = 3.0             # größte Änderung an der Messstelle im Fenster
MIN_RISE_CM = 1.0              # aktueller Anstieg an der Messstelle über H
MAX_AGE_MIN = 45               # letzter Wert der Messstelle höchstens so alt
DT_H = 0.25


def _q15(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    s = s[~s.index.duplicated()]
    return s.resample("15min").mean().interpolate(limit=4)


def estimate(kom: pd.Series | None, gauge: pd.Series | None, t0: pd.Timestamp, h_hours: float) -> dict:
    """Vorhergesagte Änderung am Landespegel in h_hours Stunden (cm) aus der Messstelle; dw None, wenn nicht freigegeben (mit Grund)."""
    out = dict(dw=None, grund="")
    try:
        if kom is None or gauge is None or len(kom) == 0 or len(gauge) == 0:
            out["grund"] = "keine Daten"
            return out
        h = int(round(h_hours / DT_H))
        lo = t0 - pd.Timedelta(hours=WINDOW_H + 2 * h_hours + 1)
        k, g = _q15(kom[(kom.index > lo) & (kom.index <= t0)]), _q15(gauge[(gauge.index > lo) & (gauge.index <= t0)])
        if len(k) == 0 or (t0 - k.index[-1]) > pd.Timedelta(minutes=MAX_AGE_MIN):
            out["grund"] = "Messstelle ohne aktuelle Werte"
            return out
        df = pd.concat([k.rename("N"), g.rename("S")], axis=1, sort=True)
        df = df[df.index <= t0]
        dS = df.S.shift(-h) - df.S                      # Änderung am Landespegel in den h Schritten danach
        dN = df.N - df.N.shift(h)                       # Änderung an der Messstelle in den h Schritten davor
        win = (df.index > t0 - pd.Timedelta(hours=WINDOW_H)) & (df.index <= t0 - pd.Timedelta(hours=h_hours))
        x, y = dN[win].to_numpy(), dS[win].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        out.update(n=int(m.sum()))
        if m.sum() < MIN_PAIRS:
            out["grund"] = "zu wenige Werte im Fenster"
            return out
        x, y = x[m], y[m]
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        beta = float(np.clip((x * y).sum() / max((x * x).sum(), 1e-6), 0.0, BETA_MAX))
        dn_now = float(df.N.iloc[-1] - df.N.iloc[-1 - h]) if len(df) > h else np.nan
        out.update(beta=round(beta, 2), r=None if not np.isfinite(r) else round(r, 2), anstieg_cm=None if not np.isfinite(dn_now) else round(dn_now, 1))
        if not (np.isfinite(r) and r >= MIN_R):
            out["grund"] = "Korrelation zu schwach"
        elif np.abs(x).max() < MIN_LEARN_CM:
            out["grund"] = "im Fenster kein deutlicher Anstieg zum Lernen"
        elif not (np.isfinite(dn_now) and dn_now >= MIN_RISE_CM):
            out["grund"] = "Messstelle steigt nicht"
        else:
            out.update(dw=beta * dn_now, grund="")
    except Exception as e:                                            # noqa: BLE001  Zusatz: nie die Vorhersage stören
        out.update(dw=None, grund=f"Fehler: {e!r}"[:160])
    return out


def weights(n_steps: int, h_hours: float, tau_h: float = TAU_H) -> np.ndarray:
    """Gewicht der Korrektur je Viertelstunde (Index 0 = t0 + 15 min): bis H linear auf 1, danach exp(-(t - H) / tau)."""
    k = np.arange(1, n_steps + 1) * DT_H
    return np.where(k <= h_hours, k / h_hours, np.exp(-(k - h_hours) / tau_h))


def lift_w(w_model: np.ndarray, w_now: float, dw: float, h_hours: float) -> tuple[np.ndarray, float]:
    """Modelllinie (Wasserstand je Viertelstunde ab t0 + 15 min) anheben; Rückgabe (Zuschlag je Schritt in cm, Zuschlag bei H)."""
    h = int(round(h_hours / DT_H))
    c = max(0.0, (w_now + dw) - float(w_model[h - 1]))
    return c * weights(len(w_model), h_hours), c


def q_of_w(c: dict, w: np.ndarray) -> np.ndarray:
    """Abfluss aus Wasserstand, wie rating.q_of_w, für Reihen."""
    lq, wn = np.log(np.asarray(c["q"])), np.asarray(c["w"])
    w = np.asarray(w, float)
    s = (wn[-1] - wn[-2]) / (lq[-1] - lq[-2])
    return np.where(w > wn[-1], np.exp(lq[-1] + (w - wn[-1]) / s), np.exp(np.interp(w, wn, lq)))


def read_w(raw) -> dict[str, pd.Series]:
    """Wasserstände der kommunalen Messstellen aus hvz.json (Zeit MEZ wie die Landespegel); bei jedem Fehler leer."""
    import json
    try:
        from live import _mez
        j = json.loads((raw / "hvz.json").read_text(encoding="utf-8")).get("kommunal") or {}
        out = {}
        for kid, _, _ in PAIRS.values():
            m = ((j.get(kid) or {}).get("W") or {}).get("measurements") or []
            if m:
                s = pd.Series([x["y"] for x in m], index=_mez([x["x"] for x in m]), dtype=float).sort_index().dropna()
                out[kid] = s[(s > -1000) & (s < 100000)]
        return out
    except Exception:                                                 # noqa: BLE001
        return {}
