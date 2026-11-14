"""Zerlegung der Strecke zum Pegel Prümzurlay: Enz (Sinspelt → Enzen → Holsthum Schloss) und Prüm (Wiersdorf → Brecht → Bettingen →
Holsthum Prüm), dann der Zusammenfluss. Grundlage: Wasserstände der kommunalen Pegel (JSON-Schnittstelle, nur W) und Abflüsse der
LfU-Pegel (Wasserportal-CSV) der Welle vom 24.–26.10.2026.

  python confluence.py

Wichtig für die Auswertung:
* Holsthum Schloss liegt an der Enz-Mündung (120 m von Holsthum Prüm). Ab 25.10. etwa 12:30 MEZ steigt die Prüm dort stark an und hebt
  den Stand am Schloss mit (Rückstau): Der Stand am Schloss springt von 54 auf 69 cm, genau gleichzeitig mit Holsthum Prüm. Für die
  Enz-Welle ist deshalb nur der Anstieg bis 25.10. 12:15 MEZ brauchbar (ENZ_END).
* Der Enz-Scheitel (3,5 m³/s) muss nicht mit dem Scheitel der Prüm zusammenfallen und geht in Prümzurlay im Prüm-Scheitel (17,4 m³/s) auf.
  Prümzurlay wird deshalb als Summe geschrieben: Q_PZ(t) = a * Q_Wiersdorf(t - L1) + b * Q_Sinspelt(t - L2) + c, wobei L1 und L2 aus
  den Zwischenpegeln kommen (L = Lag Oberlieger bis Mündungsbereich + Lag Mündungsbereich bis Prümzurlay) und nicht frei angepasst werden.
* Am 25.10. um 08:30 MEZ wurde die Abgabe des Stausees erhöht (Sprung in Wiersdorf: +21 cm in 30 min). Dieser Sprung ist eine Front, deren
  Durchgang an jedem Pegel auf etwa 15 min genau abzulesen ist: die sauberste Laufzeitmessung dieser Welle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lags_event as le

NAMES_W = {"26280402": "Wiersdorf", "1034232090": "Brecht", "1034232100": "Bettingen", "1034232110": "Holsthum Prüm",
           "26280504": "Prümzurlay", "1034232130": "Neuerburg", "1034232140": "Sinspelt Gemeindehaus", "26280650": "Sinspelt",
           "1034232150": "Enzen", "1034232160": "Holsthum Schloss"}
ENZ_END = pd.Timestamp("2026-10-25 12:15") - pd.Timedelta(hours=1)  # in UTC (12:15 MEZ)
ENZ_END = ENZ_END.tz_localize("UTC")
FRONT_WIN = (pd.Timestamp("2026-10-25 07:00", tz="UTC") - pd.Timedelta(hours=1), pd.Timestamp("2026-10-25 15:30", tz="UTC") - pd.Timedelta(hours=1))
PRUEM_CHAIN = ["Wiersdorf", "Brecht", "Bettingen", "Holsthum Prüm", "Prümzurlay"]
ENZ_CHAIN = ["Sinspelt", "Enzen", "Holsthum Schloss"]
# Flusskilometer: Wiersdorf 32,14 und Prümzurlay 7,86 (Portal), Sinspelt 15,16 bis zur Enz-Mündung (Portal). Die Zwischenpegel haben
# keine Kilometer, ihre Abstände sind die Luftlinie, gleichmäßig auf die Flusslänge hochgerechnet (Prüm 1,45, Enz 1,39).
KM_TOTAL = {"Prüm": 32.14 - 7.86, "Enz": 15.16}


def load_w(path: str = "data/json_w_event_2026-10.csv") -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["time_utc"], dtype={"number": str})
    return s.pivot(index="time_utc", columns="number", values="value").rename(columns=NAMES_W).interpolate(limit=2).sort_index()


def river_km(net: pd.DataFrame, chain: list[str], total: float) -> list[float]:
    """Flusskilometer je Teilstrecke einer Kette (Luftlinie auf bekannte Gesamtlänge skaliert)."""
    g = net.set_index("name")
    d = [float(np.hypot(g.loc[a, "easting"] - g.loc[b, "easting"], g.loc[a, "northing"] - g.loc[b, "northing"])) / 1000 for a, b in zip(chain[:-1], chain[1:])]
    k = total / sum(d)
    return [x * k for x in d]


def nse_curve(x: pd.Series, y: pd.Series, sel, maxl_h: float = 12.0) -> pd.DataFrame:
    """NSE von y(t) = a*x(t-L) + c gegen L (alle L von 0 bis maxl_h)."""
    rows = []
    sel = pd.Series(np.asarray(sel), index=y.index)
    for L in range(int(maxl_h / 0.25) + 1):
        X = x.shift(L)
        m = X.notna() & sel & y.notna()
        A = np.c_[X[m].to_numpy(), np.ones(int(m.sum()))]
        yy = y[m].to_numpy()
        c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        rows.append((L * 0.25, 1 - ((yy - A @ c) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(), c[0]))
    return pd.DataFrame(rows, columns=["lag_h", "nse", "a"])


def best_lag(x: pd.Series, y: pd.Series, sel) -> dict:
    cv = nse_curve(x, y, sel)
    k = int(cv["nse"].idxmax())
    near = cv[cv["nse"] >= cv.loc[k, "nse"] - 0.005]["lag_h"]
    return dict(lag=float(cv.loc[k, "lag_h"]), nse=float(cv.loc[k, "nse"]), a=float(cv.loc[k, "a"]), lo=float(near.min()), hi=float(near.max()))


def front_times(W: pd.DataFrame, chain: list[str]) -> list[dict]:
    """Zeitpunkt (Mitte des steilsten 30-min-Anstiegs) des Abgabe-Sprungs an jedem Pegel der Kette."""
    D = W.diff(2)
    win = (W.index >= FRONT_WIN[0]) & (W.index <= FRONT_WIN[1])
    out = []
    for c in chain:
        dd = D[c][win]
        out.append(dict(name=c, t=dd.idxmax() - pd.Timedelta(minutes=15), rise_cm=float(dd.max())))  # Mitte des 30-min-Intervalls
    return out


def front_lags_corr(W: pd.DataFrame, chain: list[str]) -> list[float]:
    D = W.diff(2)
    win = (W.index >= FRONT_WIN[0]) & (W.index <= FRONT_WIN[1])
    res = []
    for a, b in zip(chain[:-1], chain[1:]):
        best = None
        for L in range(0, 17):
            x, y = D[a].shift(L), D[b]
            m = win & x.notna() & y.notna()
            r = np.corrcoef(x[m], y[m])[0, 1]
            if best is None or r > best[1]:
                best = (L * 0.25, r)
        res.append(best[0])
    return res


def joint_fit(Q: pd.DataFrame, l1: float, l2: float) -> dict:
    """Q_PZ = a * Wiersdorf(t - l1) + b * Sinspelt(t - l2) + c mit festen Lags (Stunden)."""
    X = pd.concat([Q["Wiersdorf"].shift(int(round(l1 * 4))), Q["Sinspelt"].shift(int(round(l2 * 4)))], axis=1)
    m = X.notna().all(axis=1)
    A = np.c_[X[m].to_numpy(), np.ones(int(m.sum()))]
    y = Q["Prümzurlay"][m].to_numpy()
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    nse = 1 - ((y - A @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    parts = pd.DataFrame({"prum": c[0] * X.iloc[:, 0], "enz": c[1] * X.iloc[:, 1], "c": c[2], "meas": Q["Prümzurlay"]}, index=Q.index)
    return dict(a=float(c[0]), b=float(c[1]), c=float(c[2]), nse=float(nse), n=int(m.sum()), parts=parts)


def analyse(w_path="data/json_w_event_2026-10.csv", q_path="data/wasserportal_event_2026-10.csv", net_path="pegel_netz.csv") -> dict:
    W = load_w(w_path)
    Q = le.load_q(q_path)
    net = pd.read_csv(net_path)
    res: dict = {"W": W, "Q": Q}
    sel_enz = (W.index <= ENZ_END)
    # --- Enz (nur steigender Ast bis ENZ_END)
    km_enz = river_km(net, ENZ_CHAIN, KM_TOTAL["Enz"])
    res["enz_km"] = km_enz
    enz = {}
    for a, b in (("Sinspelt", "Enzen"), ("Enzen", "Holsthum Schloss"), ("Sinspelt", "Holsthum Schloss")):
        r_all = best_lag(W[a], W[b], sel_enz)
        r_day = best_lag(W[a], W[b], sel_enz & (W.index >= pd.Timestamp("2026-10-25 00:00", tz="UTC") - pd.Timedelta(hours=1)))
        enz[(a, b)] = dict(window_from_24=r_all, window_from_25=r_day)
    res["enz"] = enz
    # --- Prüm-Kette: Front
    km_pr = river_km(net, PRUEM_CHAIN, KM_TOTAL["Prüm"])
    res["prum_km"] = km_pr
    res["front"] = front_times(W, PRUEM_CHAIN)
    res["front_corr"] = front_lags_corr(W, PRUEM_CHAIN)
    # Wiersdorf -> Holsthum Prüm und Holsthum Prüm / Schloss -> Prümzurlay über das ganze Fenster
    allsel = np.ones(len(W), bool)
    res["w_hp"] = best_lag(W["Wiersdorf"], W["Holsthum Prüm"], allsel)
    res["hp_pz"] = best_lag(W["Holsthum Prüm"], W["Prümzurlay"], allsel)
    res["hs_pz"] = best_lag(W["Holsthum Schloss"], W["Prümzurlay"], allsel)
    # --- Prümzurlay als Summe (Q), Lags aus den Zwischenpegeln
    t_front = res["front"][-1]["t"] - res["front"][0]["t"]
    L1 = t_front.total_seconds() / 3600
    L_sin_hs = enz[("Sinspelt", "Holsthum Schloss")]["window_from_25"]["lag"]
    L2 = L_sin_hs + res["hs_pz"]["lag"]
    res["L1"], res["L2"] = L1, L2
    res["fixed"] = joint_fit(Q, L1, L2)
    res["free"] = le.fit_lag(Q["Wiersdorf"], Q["Prümzurlay"], extra=Q["Sinspelt"])
    res["sens"] = {l2: joint_fit(Q, L1, l2) for l2 in (0.5, 1.0, 1.75, 2.75, 3.5, 4.5, 6.0)}
    return res


def main():
    r = analyse()
    W = r["W"]
    mez = lambda t: (t + pd.Timedelta(hours=1)).strftime("%d.%m. %H:%M")
    print("== 1) Enz, nur steigender Ast bis 25.10. 12:15 MEZ (danach Rückstau der Prüm am Schloss)")
    print(f"Flusskilometer der Teilstrecken (Luftlinie auf 15,16 km hochgerechnet): Sinspelt→Enzen {r['enz_km'][0]:.1f} km, Enzen→Schloss {r['enz_km'][1]:.1f} km")
    for (a, b), v in r["enz"].items():
        for k, name in (("window_from_24", "Fenster ab 24.10. 12:00"), ("window_from_25", "Fenster ab 25.10. 00:00")):
            x = v[k]
            print(f"  {a:>16} → {b:<17} {name}: Lag {x['lag']:.2f} h (NSE {x['nse']:.3f}), NSE-Plateau {x['lo']:.2f} bis {x['hi']:.2f} h")
    print("\n== 2) Prüm: Durchgang der Abgabe-Front (Stausee-Sprung am 25.10.), steilster 30-min-Anstieg")
    f = r["front"]
    print("  Pegel            Zeit MEZ   Anstieg cm/30 min")
    for x in f:
        print(f"  {x['name']:<16} {mez(x['t'])}   {x['rise_cm']:.0f}")
    t0 = f[0]["t"]
    for i, (a, b) in enumerate(zip(PRUEM_CHAIN[:-1], PRUEM_CHAIN[1:])):
        dt = (f[i + 1]["t"] - f[i]["t"]).total_seconds() / 3600
        print(f"  {a:>14} → {b:<14} {dt:4.2f} h über {r['prum_km'][i]:.1f} km = {r['prum_km'][i] * 1000 / (dt * 3600):.2f} m/s (Korrelation der Änderungen: {r['front_corr'][i]:.2f} h)")
    tot = (f[-1]["t"] - t0).total_seconds() / 3600
    print(f"  Wiersdorf → Prümzurlay gesamt: {tot:.2f} h über {sum(r['prum_km']):.1f} km = {sum(r['prum_km']) * 1000 / (tot * 3600):.2f} m/s")
    print(f"\n  Fit über ganzes Fenster (Wasserstand): Wiersdorf→Holsthum Prüm {r['w_hp']['lag']:.2f} h (NSE {r['w_hp']['nse']:.3f}), "
          f"Holsthum Prüm→Prümzurlay {r['hp_pz']['lag']:.2f} h, Holsthum Schloss→Prümzurlay {r['hs_pz']['lag']:.2f} h")
    print("\n== 3) Prümzurlay als Summe  Q_PZ = a*Wiersdorf(t-L1) + b*Sinspelt(t-L2) + c")
    print(f"  L1 = Front Wiersdorf→Prümzurlay = {r['L1']:.2f} h;  L2 = Sinspelt→Schloss ({r['enz'][('Sinspelt','Holsthum Schloss')]['window_from_25']['lag']:.2f} h) + Schloss→Prümzurlay ({r['hs_pz']['lag']:.2f} h) = {r['L2']:.2f} h")
    p = r["fixed"]
    print(f"  mit diesen festen Lags: a={p['a']:.2f}, b={p['b']:.2f}, c={p['c']:.2f}, NSE={p['nse']:.4f}")
    fr = r["free"]
    print(f"  freie Anpassung zum Vergleich: L1={fr['lag_h']:.2f} h, L2={fr['lag2_h']:.2f} h, a={fr['a']:.2f}, b={fr['b']:.2f}, c={fr['c']:.2f}, NSE={fr['nse']:.4f}")
    print("  Empfindlichkeit gegen L2 (L1 fest):")
    for l2, s in r["sens"].items():
        print(f"     L2 {l2:4.2f} h: NSE {s['nse']:.4f}, b={s['b']:.2f}")
    parts = p["parts"].dropna()
    pk = parts["meas"].idxmax()
    print(f"\n  Zur Zeit des Scheitels in Prümzurlay ({mez(pk)} MEZ, {parts.loc[pk, 'meas']:.1f} m³/s): Prüm-Anteil {parts.loc[pk, 'prum']:.1f}, Enz-Anteil {parts.loc[pk, 'enz']:.1f}, Rest {parts.loc[pk, 'c']:.1f} m³/s")
    ep = (parts["enz"]).idxmax()
    print(f"  Zur Zeit des größten Enz-Anteils ({mez(ep)} MEZ, {parts.loc[ep, 'enz']:.1f} m³/s): Prümzurlay {parts.loc[ep, 'meas']:.1f}, Prüm-Anteil {parts.loc[ep, 'prum']:.1f}")


if __name__ == "__main__":
    main()
