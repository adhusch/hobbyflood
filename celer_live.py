"""Wellenlaufzeit Wiersdorf -> Prümzurlay live aus den kommunalen Pegeln und daraus der Faktor auf c(Q) im Modell (chain_fc.CELER).

Wiersdorf (Landespegel) und die kommunalen Messstellen Brecht, Bettingen und Holsthum Prüm liegen an derselben Strecke ohne nennenswerten
Zwischenzufluss (die Enz mündet erst unterhalb von Holsthum Prüm). Je Teilstrecke wird die Laufzeit aus der Kreuzkorrelation der Anstiege
bestimmt (geglättete Änderung des Wasserstands je Viertelstunde; auf den Wasserständen selbst ist die Korrelation bei jeder Verschiebung
nahe 1 und taugt nicht). Dasselbe Maß auf der vom Modell gerouteten Wiersdorf-Welle (Kaskade der Strecke Wiersdorf -> Prümzurlay) gibt die
Laufzeit des Modells. Faktor = c gemessen / c Modell, gedämpft zur 1 hin, begrenzt.

Prüfung am Sturm 16.-18.11.2026: Teilstrecken r 0,62-0,72, zusammen 1,41-1,62 m/s, Modell 4,6-4,8 h für 24,3 km; Oktoberwelle 2026 (Abgabe
des Stausees): 1,2-1,3 m/s. Uhren der kommunalen Messstellen geprüft (Sinspelt Gemeindehaus gegen Landespegel Sinspelt, 300 m: 4 min).

Freigabe je Teilstrecke: r >= MIN_R, in beiden Reihen ein Anstieg von mindestens MIN_RISE_CM im Fenster, Laufzeit im zulässigen Bereich und
Geschwindigkeit 0,4-4 m/s. Ohne freigegebene Teilstrecke: Faktor 1. Jeder Fehler ergibt Faktor 1 (die Vorhersage hängt nie daran).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
from routing import cascade

# Strecke flussab: (Schlüssel, Name, km bis Prümzurlay); Schlüssel "WI" = Landespegel, sonst Nummer der kommunalen Messstelle
REACH = [("WI", "Wiersdorf", 24.3), ("1034232090", "Brecht", 17.4), ("1034232100", "Bettingen", 10.9), ("1034232110", "Holsthum Prüm", 3.7)]
L_TOTAL = 24.3                       # Wiersdorf -> Prümzurlay (Strecke im Modell)
WINDOW_H = 36                        # Fenster der Kreuzkorrelation bis jetzt
MIN_R = 0.6
MIN_RISE_CM = 10.0
LAG_STEPS = (1, 20)                  # zulässige Laufzeit je Teilstrecke in Viertelstunden (0,25-5 h)
F_MIN, F_MAX = 0.6, 1.6
DT_H = 0.25


def _deriv(s: pd.Series, t_end: pd.Timestamp) -> pd.Series:
    s = s.sort_index()
    s = s[~s.index.duplicated()].resample("15min").mean().interpolate(limit=4)
    s = s[(s.index > t_end - pd.Timedelta(hours=WINDOW_H)) & (s.index <= t_end)]
    return s.rolling(4, center=True, min_periods=2).mean().diff()


def _xcorr(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Verschiebung (h) mit der größten Korrelation von b gegen a (b später), mit Parabel zwischen den Viertelstunden; dazu r."""
    a, b = a.align(b, join="inner")
    x0, y0 = a.to_numpy(), b.to_numpy()
    lags = range(LAG_STEPS[0] - 1, LAG_STEPS[1] + 2)
    rs = []
    for L in lags:
        x, y = (x0[:len(x0) - L], y0[L:]) if L > 0 else (x0, y0)
        m = np.isfinite(x) & np.isfinite(y)
        rs.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 24 and np.std(x[m]) > 0 and np.std(y[m]) > 0 else np.nan)
    rs = np.array(rs)
    inner = rs[1:-1]
    if not np.isfinite(inner).any():
        return np.nan, np.nan
    i = int(np.nanargmax(inner)) + 1
    f = 0.0
    if np.isfinite(rs[i - 1]) and np.isfinite(rs[i + 1]):
        den = rs[i - 1] - 2 * rs[i] + rs[i + 1]
        f = 0.5 * (rs[i - 1] - rs[i + 1]) / den if den != 0 else 0.0
    return (list(lags)[i] + max(min(f, 0.5), -0.5)) * DT_H, float(rs[i])


def _w_series(raw: Path, hv: dict) -> dict[str, pd.Series]:
    from live import _mez
    out = {"WI": hv["WI"]["W"]}
    j = json.loads((raw / "hvz.json").read_text(encoding="utf-8")).get("kommunal") or {}
    for key, _, _ in REACH[1:]:
        m = ((j.get(key) or {}).get("W") or {}).get("measurements") or []
        if m:
            s = pd.Series([x["y"] for x in m], index=_mez([x["x"] for x in m]), dtype=float).sort_index().dropna()
            out[key] = s[(s > -1000) & (s < 100000)]
    return out


def _model_lag(models: dict, hv: dict, t0: pd.Timestamp) -> tuple[float, float]:
    """Laufzeit des Modells (gleiches Maß) und mittlerer Abfluss Wiersdorf im Fenster."""
    lk = next(lk_ for lk_ in models["PZ"]["rain_full"].links if lk_.up == C.DN["WI"])
    q = hv["WI"]["Q"].sort_index()
    q = q[~q.index.duplicated()].resample("15min").mean().interpolate(limit_direction="both")
    q = q[q.index <= t0]
    y = pd.Series(np.asarray(cascade(q.to_numpy(), lk.n, lk.length_km, lk.lq, lk.lc)).ravel(), index=q.index)
    lag, _ = _xcorr(_deriv(q, t0), _deriv(y, t0))
    win = q[q.index > t0 - pd.Timedelta(hours=WINDOW_H)]
    return lag, float(win.mean())


def estimate(raw: Path, hv: dict, models: dict, t0: pd.Timestamp, state_file: Path | None = None) -> dict:
    """Faktor für chain_fc.CELER["PZ"]["WI"] mit Begründung. Schlüssel f (angewendeter Faktor), f_roh, angewendet, grund, strecken, ..."""
    out = dict(f=1.0, angewendet=False, grund="", strecken=[])
    try:
        W = _w_series(raw, hv)
        valid_km, valid_h, rs = 0.0, 0.0, []
        for (ka, na, da), (kb, nb, db) in zip(REACH[:-1], REACH[1:]):
            rec = dict(von=na, bis=nb, km=round(da - db, 1))
            if ka not in W or kb not in W:
                rec["grund"] = "keine Daten"
                out["strecken"].append(rec)
                continue
            a, b = _deriv(W[ka], t0), _deriv(W[kb], t0)
            rise_a = float(W[ka][W[ka].index > t0 - pd.Timedelta(hours=WINDOW_H)].pipe(lambda s: s.max() - s.min()) if len(W[ka]) else 0)
            rise_b = float(W[kb][W[kb].index > t0 - pd.Timedelta(hours=WINDOW_H)].pipe(lambda s: s.max() - s.min()) if len(W[kb]) else 0)
            lag, r = _xcorr(a, b)
            c = (da - db) / 3.6 / lag if np.isfinite(lag) and lag > 0 else np.nan
            ok = (np.isfinite(r) and r >= MIN_R and min(rise_a, rise_b) >= MIN_RISE_CM and np.isfinite(c) and 0.4 <= c <= 4.0)
            rec.update(laufzeit_h=None if not np.isfinite(lag) else round(lag, 2), r=None if not np.isfinite(r) else round(r, 2),
                       c=None if not np.isfinite(c) else round(c, 2), anstieg_cm=round(min(rise_a, rise_b), 0), gilt=bool(ok))
            if not ok:
                rec["grund"] = ("Korrelation zu schwach" if not (np.isfinite(r) and r >= MIN_R) else
                                "kein deutlicher Anstieg" if min(rise_a, rise_b) < MIN_RISE_CM else "Laufzeit unplausibel")
            else:
                valid_km += da - db
                valid_h += lag
                rs.append(r)
            out["strecken"].append(rec)
        if valid_km <= 0:
            out["grund"] = "keine Teilstrecke mit deutlichem, gut korreliertem Anstieg"
            return out
        lag_m, q_wi = _model_lag(models, hv, t0)
        if not np.isfinite(lag_m) or lag_m <= 0:
            out["grund"] = "Laufzeit des Modells nicht bestimmbar"
            return out
        c_obs = valid_km / 3.6 / valid_h
        c_mod = L_TOTAL / 3.6 / lag_m
        f_raw = c_obs / c_mod
        # Dämpfung zur 1 hin: je mehr Strecke und je besser die Korrelation, desto mehr Gewicht
        w = min(1.0, valid_km / 20.6) * float(np.clip((np.mean(rs) - 0.5) / 0.3, 0.0, 1.0))
        prev = None
        if state_file is not None:
            try:
                prev = json.loads(state_file.read_text(encoding="utf-8")).get("f_roh")
            except (OSError, ValueError):
                prev = None
            if prev and abs(f_raw / prev - 1) > 0.25:               # springt der Wert, halbes Gewicht
                w *= 0.5
        f = float(np.clip(1 + w * (f_raw - 1), F_MIN, F_MAX))
        out.update(f=round(f, 3), f_roh=round(f_raw, 3), gewicht=round(w, 2), angewendet=True, c_gemessen=round(c_obs, 2), c_modell=round(c_mod, 2),
                   laufzeit_gemessen_h=round(L_TOTAL / 3.6 / c_obs, 2), laufzeit_modell_h=round(lag_m, 2), strecke_km=round(valid_km, 1),
                   q_wiersdorf=round(q_wi, 1), vorher_f_roh=prev, grund="")
        if state_file is not None:
            try:
                state_file.write_text(json.dumps(dict(f_roh=f_raw, t0=str(t0))), encoding="utf-8")
            except OSError:
                pass
    except Exception as e:                                         # noqa: BLE001  Zusatz: nie die Vorhersage stören
        out.update(f=1.0, angewendet=False, grund=f"Fehler: {e!r}"[:200])
    return out
