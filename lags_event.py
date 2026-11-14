"""Ereignisbasierte Lag-Analyse für Pegelpaare: ganze Welle mit Scheitel und Rückgang, Lag in Abhängigkeit vom Abfluss.

  python lags_event.py [data/real_5d.csv] [--from 2026-10-24T20:00] [--to 2026-10-27T00:00]

Je Paar (Oberlieger -> Unterlieger):
  Scheitel      Zeitversatz der Scheitel (parabolisch verfeinert), nur wenn beide Scheitel sicher erreicht sind;
                zusätzlich der Versatz der Plateau-Mitten (Mitte des Bereichs >= 99 % des Maximums), weil die echten Scheitel flach sind
  Schwerpunkt   Versatz der zeitlichen Schwerpunkte der Welle über dem Basisabfluss
  Anstieg f     Versatz der Zeitpunkte, an denen jeder Pegel den Anteil f seines eigenen Anstiegs erreicht (steigender Ast)
  Rückgang f    dasselbe auf dem fallenden Ast
  Q_Ober        Abfluss des Oberliegers bei diesen Durchgängen: so lässt sich der Lag gegen den Abfluss auftragen
  Korrelation   bester Versatz der Korrelation über das Fenster (Niveau und Änderungen)
  Fit           y(t) = a * x(t-L) + c nach Kleinste Quadrate, gesamt sowie getrennt für steigenden und fallenden Ast (bis/nach Scheitel des Unterliegers)

Gilt wie lags.py: Das ist die Reaktionsverzögerung zwischen zwei Ganglinien, bei flächigem Regen mit Zwischenzufluss, keine
Wellenlaufzeit. Ein Ereignis mit Q bis etwa 12 m³/s ersetzt keine Hochwasserhistorie.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

NAMES = {"26280050": "Prüm 2", "26280300": "Echtershausen", "26280402": "Wiersdorf", "26280504": "Prümzurlay",
         "26280650": "Sinspelt", "26280708": "Giesdorf", "26280800": "Seffern", "26280854": "Alsdorf-Oberecken"}
PAIRS = [("Prüm 2", "Echtershausen"), ("Echtershausen", "Wiersdorf"), ("Wiersdorf", "Prümzurlay"),
         ("Sinspelt", "Prümzurlay"), ("Giesdorf", "Seffern"), ("Seffern", "Alsdorf-Oberecken")]
FRACS = (0.1, 0.25, 0.5, 0.75, 0.9)


def load_q(path: str) -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["time_utc"], dtype={"number": str})
    w = s[s["var"] == "Q"].pivot(index="time_utc", columns="number", values="value")
    return w.rename(columns=NAMES).sort_index()


def peak_info(s: pd.Series, min_drop: float = 0.9) -> dict:
    """Scheitelzeit (parabolisch verfeinert) und ob der Scheitel sicher erreicht ist (Wert am Ende unter min_drop * Scheitel)."""
    v = s.to_numpy()
    i = int(np.nanargmax(v))
    dt = (s.index[1] - s.index[0]).total_seconds() / 3600
    off = 0.0
    if 0 < i < len(v) - 1:
        a, b, c = v[i - 1], v[i], v[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    reached = i < len(v) - 1 and v[-1] <= min_drop * v[i] and i > 0
    # Plateau: zusammenhängender Bereich um den Scheitel mit Werten >= 99 % des Maximums; Mitte und Breite
    lo = hi = i
    while lo > 0 and v[lo - 1] >= 0.99 * v[i]:
        lo -= 1
    while hi < len(v) - 1 and v[hi + 1] >= 0.99 * v[i]:
        hi += 1
    tmid = s.index[lo] + (s.index[hi] - s.index[lo]) / 2
    return dict(t=s.index[i] + pd.Timedelta(hours=off * dt), q=float(v[i]), reached=bool(reached), i=i, t_mid=tmid,
                plateau_h=(hi - lo + 1) * dt)


def crossing(s: pd.Series, level: float, i0: int, i1: int, rising: bool) -> pd.Timestamp | None:
    """Erstes Durchschreiten von level im Index-Bereich [i0, i1], linear interpoliert."""
    v = s.to_numpy()
    for i in range(max(i0, 1), i1 + 1):
        a, b = v[i - 1], v[i]
        if (rising and a < level <= b) or (not rising and a > level >= b):
            f = (level - a) / (b - a) if b != a else 0.0
            return s.index[i - 1] + (s.index[i] - s.index[i - 1]) * f
    return None


def base_level(s: pd.Series, n: int = 8) -> float:
    return float(s.iloc[:n].median())


def centroid(s: pd.Series, base: float) -> pd.Timestamp:
    x = np.clip(s.to_numpy() - base, 0, None)
    t = (s.index - s.index[0]).total_seconds().to_numpy()
    return s.index[0] + pd.Timedelta(seconds=float((x * t).sum() / x.sum()))


def best_corr_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[int, float]:
    out = []
    for L in range(maxlag + 1):
        x = a[:len(a) - L] if L else a
        out.append(np.corrcoef(x, b[L:])[0, 1])
    out = np.array(out)
    k = int(np.nanargmax(out))
    return k, float(out[k])


def fit_lag(x: pd.Series, y: pd.Series, sel=None, extra: pd.Series | None = None, maxl_h: float = 12.0) -> dict | None:
    """Beste Verschiebung L (und bei extra auch L2) für y(t) = a * x(t-L) [+ b * extra(t-L2)] + c nach Kleinste Quadrate.

    Gütemaß ist die Nash-Sutcliffe-Effizienz (NSE) über die gewählten Zeitschritte. Der Zwischenzufluss steckt in a, b und c.
    """
    dt = (y.index[1] - y.index[0]).total_seconds() / 3600
    n = int(round(maxl_h / dt))
    sel = pd.Series(True, index=y.index) if sel is None else pd.Series(np.asarray(sel), index=y.index)
    best = None
    for L in range(n + 1):
        for L2 in (range(n + 1) if extra is not None else [0]):
            X = pd.concat([x.shift(L)] + ([extra.shift(L2)] if extra is not None else []), axis=1)
            m = X.notna().all(axis=1) & sel & y.notna()
            if m.sum() < 12:
                continue
            A = np.c_[X[m].to_numpy(), np.ones(int(m.sum()))]
            yy = y[m].to_numpy()
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            nse = 1 - ((yy - A @ coef) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            if best is None or nse > best["nse"]:
                best = dict(nse=float(nse), lag_h=L * dt, lag2_h=L2 * dt if extra is not None else None,
                            a=float(coef[0]), b=float(coef[1]) if extra is not None else None, c=float(coef[-1]), n=int(m.sum()))
    return best


def pair_table(w: pd.DataFrame, up: str, dn: str, fracs=FRACS) -> dict:
    u, d = w[up].dropna(), w[dn].dropna()
    idx = u.index.intersection(d.index)
    u, d = u.loc[idx], d.loc[idx]
    dt = (idx[1] - idx[0]).total_seconds() / 3600
    pu, pd_ = peak_info(u), peak_info(d)
    res = dict(up=up, dn=dn, dt_h=dt, peak_up=pu, peak_dn=pd_, both_peaked=pu["reached"] and pd_["reached"])
    res["lag_peak_h"] = (pd_["t"] - pu["t"]).total_seconds() / 3600 if res["both_peaked"] else None
    res["lag_peak_mid_h"] = (pd_["t_mid"] - pu["t_mid"]).total_seconds() / 3600 if res["both_peaked"] else None
    bu, bd = base_level(u), base_level(d)
    res["base"] = (bu, bd)
    hrs = lambda a, b: (b - a).total_seconds() / 3600
    if res["both_peaked"]:
        res["lag_centroid_h"] = hrs(centroid(u, bu), centroid(d, bd))
    else:
        res["lag_centroid_h"] = None
    rows = []
    for f in fracs:
        lu, ld = bu + f * (pu["q"] - bu), bd + f * (pd_["q"] - bd)
        ru, rd = crossing(u, lu, 1, pu["i"], True), crossing(d, ld, 1, pd_["i"], True)
        fu = crossing(u, lu, pu["i"], len(u) - 1, False) if pu["reached"] else None
        fd = crossing(d, ld, pd_["i"], len(d) - 1, False) if pd_["reached"] else None
        rows.append(dict(f=f, q_up=lu, q_dn=ld, rise_h=hrs(ru, rd) if ru is not None and rd is not None else None,
                         fall_h=hrs(fu, fd) if fu is not None and fd is not None else None))
    res["levels"] = rows
    kl, rl = best_corr_lag(u.to_numpy(), d.to_numpy(), int(24 / dt))
    k4 = max(1, int(round(1 / dt)))
    kd, rd_ = best_corr_lag(u.diff(k4).to_numpy()[k4:], d.diff(k4).to_numpy()[k4:], int(12 / dt))
    tp = pd_["t"]
    res["fit"] = fit_lag(u, d)
    res["fit_rise"] = fit_lag(u, d, d.index <= tp)
    res["fit_fall"] = fit_lag(u, d, d.index > tp)
    res["corr_level_h"], res["corr_level_r"] = kl * dt, rl
    res["corr_diff_h"], res["corr_diff_r"] = kd * dt, rd_
    return res


def show(tabs: list[dict]) -> None:
    f = lambda x, d=2: "  –  " if x is None else f"{x:{5 + d - 2}.{d}f}"
    for t in tabs:
        print(f"\n== {t['up']} -> {t['dn']}")
        pu, pd_ = t["peak_up"], t["peak_dn"]
        print(f"   Scheitel Oberlieger {pu['t']:%d.%m. %H:%M} UTC, {pu['q']:.2f} m³/s, erreicht: {'ja' if pu['reached'] else 'NEIN'}"
              f" | Unterlieger {pd_['t']:%d.%m. %H:%M} UTC, {pd_['q']:.2f} m³/s, erreicht: {'ja' if pd_['reached'] else 'NEIN'}")
        print(f"   Plateau (>= 99 % des Maximums): Oberlieger {pu['plateau_h']:.2f} h, Unterlieger {pd_['plateau_h']:.2f} h; "
              f"Lag der Plateau-Mitten {f(t['lag_peak_mid_h'])} h")
        print(f"   Lag Scheitel {f(t['lag_peak_h'])} h | Schwerpunkt {f(t['lag_centroid_h'])} h | "
              f"Korrelation Niveau {t['corr_level_h']:.2f} h (r={t['corr_level_r']:.3f}), Änderungen {t['corr_diff_h']:.2f} h (r={t['corr_diff_r']:.3f})")
        if t.get("fit"):
            fz = lambda r: "–" if not r else f"L={r['lag_h']:.2f} h, a={r['a']:.2f}, NSE={r['nse']:.3f}"
            print(f"   Fit y=a*x(t-L)+c: gesamt {fz(t['fit'])} | steigend {fz(t['fit_rise'])} | fallend {fz(t['fit_fall'])}")
        print("   Anteil f   Q_Ober   Anstieg-Lag h   Rückgang-Lag h")
        for r in t["levels"]:
            print(f"     {r['f']:.2f}   {r['q_up']:7.2f}   {f(r['rise_h']):>12}   {f(r['fall_h']):>12}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/real_5d.csv")
    ap.add_argument("--from", dest="t0")
    ap.add_argument("--to", dest="t1")
    a = ap.parse_args()
    w = load_q(a.path)
    if a.t0:
        w = w[w.index >= pd.Timestamp(a.t0, tz="UTC")]
    if a.t1:
        w = w[w.index <= pd.Timestamp(a.t1, tz="UTC")]
    print(f"{len(w)} Zeitschritte, {w.index[0]:%d.%m. %H:%M} bis {w.index[-1]:%d.%m. %H:%M} UTC")
    show([pair_table(w, u, d) for u, d in PAIRS])
