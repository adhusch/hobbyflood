"""Nachrechnung der Welle vom 24. bis 28.10.2026 mit den ICON-D2-Läufen, die damals wirklich vorlagen (python hindcast.py).

Echte Daten, keine Synthetik:
  Abfluss     Wasserportal, 90 Tage 15-min-Rohdaten der acht Pegel (Export 12.11.2026, MEZ): data/hindcast/wasserportal_q90_20261112.json
  Regen       DWD-Stundenwerte der Stationen (recent), wie im Live-Betrieb (live.station_obs)
  Vorhersage  ICON-D2-Einzelläufe über die Single Runs API von Open-Meteo (Archiv ab April 2026), alle 3 h vom 21.10. bis 28.10.:
              data/hindcast/icon_d2_runs_2026-10.json (15 min an den Stationsorten)
Ausgabe je Lauf 3 h nach dessen Start (so lange dauert es, bis ein Lauf verfügbar ist), jeweils für alle Pegel:
  ohne        Kette wie bisher, Regen nach der Ausgabe 0 (kette_regen)
  ICON-D2     Kette mit dem Regen des Laufs (kette_vorhersage), an den Stationen mit dem Faktor Station / ICON angeglichen
  perfekt     Kette mit dem gemessenen Regen nach der Ausgabe (kette_vorhersage), Obergrenze
Modelle aus allen Ereignissen 2004-2022, also unabhängig von 2026. Ausgaben data/hindcast/hindcast_2026-10.csv und gui/hindcast_2026-10.png.
Grenze: eine kleine Welle (Prümzurlay 17 m³/s, etwa 3 % von HQ), deterministisch (die Einzelläufe des Ensembles sind nicht archiviert).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import live as LV
import rain as RN

HERE = Path(__file__).parent
HC = HERE / "data" / "hindcast"


def load_q() -> pd.DataFrame:
    d = json.loads((HC / "wasserportal_q90_20261112.json").read_text(encoding="utf-8"))
    out = {}
    for n, t in d["dateien"].items():
        df = pd.read_csv(io.StringIO(t), sep=";", decimal=",", na_values=["-"], header=0)
        s = pd.Series(pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 2], format="%d.%m.%Y %H:%M"))
        key = next(k for k, v in LV.NUM.items() if v == n)
        out[C.DN[key]] = s[~s.index.duplicated()]
    return pd.DataFrame(out)


def run_rain(run: list, ids: list[str], fac: dict, hours: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Ein ICON-D2-Lauf (Liste über die Stationsorte) als Stundensummen je Station, MEZ, angeglichen."""
    out = {}
    for sid, loc in zip(ids, run):
        m = loc["minutely_15"]
        s = pd.Series(pd.to_numeric(pd.Series(m["precipitation"]), errors="coerce").to_numpy(), index=LV._mez(m["time"]))
        h = s.resample("1h", label="right", closed="right").sum(min_count=4) * fac.get(sid, {"faktor": 1.0})["faktor"]
        out[sid] = h.reindex(hours)
    return out


def main() -> None:
    Q = load_q()
    raw = LV.latest_raw()
    obs_st, _ = LV.station_obs(raw)
    fac = json.loads((HERE / "data" / "verif" / "rain_factors.json").read_text(encoding="utf-8"))["stationen"]
    runs = json.loads((HC / "icon_d2_runs_2026-10.json").read_text(encoding="utf-8"))
    ids = [s[0] for s in runs["stationen"]]
    grid = pd.date_range(Q.index[0].ceil("1h"), Q.index[-1].floor("1h"), freq="15min")
    hours = pd.date_range(grid[0].ceil("1h"), grid[-1].ceil("1h"), freq="1h")
    Qn = {c: Q[c].reindex(grid).interpolate(limit=4, limit_area="inside").to_numpy() for c in Q.columns}
    info = LV.area_info()
    st = {s: v.reindex(hours) for s, v in obs_st.items() if s in ids}
    p_obs = {k: np.nan_to_num(LV.hourly_to_grid(RN.areal_series(st, {s: w for s, w in info[k]["weights"].items() if s in st}), grid)) for k in C.RAINK}
    C._S.update(pm=C.storm_index({k: dict(p=p_obs[k]) for k in C.RAINK}, info), month=grid.month.to_numpy())
    models = C.from_json(json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))["sets"]["alle"])
    obs_frames = LV.build_frames(p_obs, Qn)
    rows, curves = [], {}
    for run_id, run in sorted(runs["runs"].items()):
        t_init = pd.Timestamp(run_id) + LV.MEZ
        t0 = t_init + pd.Timedelta(hours=2)                        # verfügbar etwa 1,5 h nach Start; Ausgabe zur nächsten vollen Stunde danach
        if t0 not in grid:
            continue
        i0 = int(grid.get_loc(t0))
        if i0 - C.A - C.LEAD < 0 or i0 + C.H >= len(grid):
            continue
        fc_st = run_rain(run, ids, fac, hours)
        # Stationen wie bei den Messungen (nur die, die zur Ausgabezeit messen), Gebietsmittel; vor der Ausgabe gemessen
        live_ids = [s for s in st if st[s].loc[:t0].dropna().index.max() >= t0 - pd.Timedelta(hours=24)]
        p_fc = {}
        for k in C.RAINK:
            w = {s: v for s, v in info[k]["weights"].items() if s in live_ids}
            a = np.nan_to_num(LV.hourly_to_grid(RN.areal_series({s: fc_st[s] for s in w}, w), grid))
            p_fc[k] = np.where(grid > t0, a, p_obs[k])
        p_none = {k: np.where(grid > t0, 0.0, p_obs[k]) for k in C.RAINK}
        res = {}
        for name, frames, var in (("ohne", LV.build_frames(p_none, Qn), "kette_regen"), ("ICON-D2", LV.build_frames(p_fc, Qn), "kette_vorhersage"),
                                  ("perfekt", obs_frames, "kette_vorhersage")):
            S = {}
            for target in C.ORDER:
                S[target] = C.forecast(models, target, Qn, frames, np.array([i0]), var)[target][0]
            res[name] = S
        for target in C.ORDER:
            dn = C.DN[target]
            o = Qn[dn][i0 + 1:i0 + C.H + 1]
            if not np.isfinite(o).all():
                continue
            for name, S in res.items():
                f = S[target][C.A + 1:]
                pr = (p_fc if name == "ICON-D2" else p_obs if name == "perfekt" else p_none)["PZ"]
                rain48 = float(np.sum(pr[i0 + 1:i0 + C.H + 1]) * C.DT)            # Zwischengebiet Prümzurlay, mm in 48 h
                rows.append(dict(lauf=run_id, ausgabe=str(t0), pegel=dn, variante=name, q0=float(Qn[dn][i0]), max_obs=float(o.max()), max_fc=float(f.max()),
                                 t_obs_h=(int(o.argmax()) + 1) * C.DT, t_fc_h=(int(f.argmax()) + 1) * C.DT,
                                 mae_6h=float(np.mean(np.abs(f[:24] - o[:24]))), mae_24h=float(np.mean(np.abs(f[:96] - o[:96]))),
                                 mae_48h=float(np.mean(np.abs(f - o))), regen48_pz_mm=rain48))
            curves.setdefault(dn, {})[str(t0)] = {name: [round(float(x), 3) for x in S[target][C.A + 1:]] for name, S in res.items()}
    df = pd.DataFrame(rows)
    df.to_csv(HC / "hindcast_2026-10.csv", index=False, float_format="%.4g")
    (HC / "hindcast_2026-10_curves.json").write_text(json.dumps(dict(grid0=str(grid[0]), curves=curves)), encoding="utf-8")
    summary(df)
    figure(Q, curves)


def summary(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    ev = df[(df.ausgabe >= "2026-10-23 12:00") & (df.ausgabe <= "2026-10-26 00:00")]
    g = ev.groupby(["pegel", "variante"]).agg(mae6=("mae_6h", "mean"), mae24=("mae_24h", "mean"), mae48=("mae_48h", "mean")).round(2).unstack("variante")
    print("Mittlerer absoluter Fehler (m³/s), Ausgaben 23.10. 12:00 bis 26.10. 00:00:")
    print(g.to_string())
    pz = df[df.pegel == "Prümzurlay"].pivot(index="ausgabe", columns="variante", values="max_fc").join(df[(df.pegel == "Prümzurlay") & (df.variante == "ohne")].set_index("ausgabe")[["q0", "max_obs"]])
    print("Prümzurlay: höchster Wert in den nächsten 48 h je Ausgabe (m³/s)")
    print(pz.round(1).to_string())


def figure(Q: pd.DataFrame, curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    issues = ["2026-10-24 12:00:00", "2026-10-24 21:00:00", "2026-10-25 06:00:00"]
    gauges = ["Prüm 2", "Echtershausen", "Prümzurlay"]
    fig, axs = plt.subplots(len(gauges), len(issues), figsize=(13, 8.5), sharex=True)
    t = pd.date_range("2026-10-23 12:00", "2026-10-28 12:00", freq="15min")
    cols = {"ohne": "#898781", "ICON-D2": "#2a78d6", "perfekt": "#1baf7a"}
    for r, g in enumerate(gauges):
        for c, iss in enumerate(issues):
            ax = axs[r, c]
            ax.plot(t, Q[g].reindex(t), color="#0b0b0b", lw=1.6, label="gemessen")
            cv = curves.get(g, {}).get(iss)
            if cv:
                tt = pd.date_range(pd.Timestamp(iss) + pd.Timedelta(minutes=15), periods=C.H, freq="15min")
                for name, v in cv.items():
                    ax.plot(tt, v, color=cols[name], lw=1.6, ls="--" if name == "perfekt" else "-", label={"ohne": "ohne Regenvorhersage", "ICON-D2": "mit ICON-D2-Lauf", "perfekt": "mit gemessenem Regen (Obergrenze)"}[name])
            ax.axvline(pd.Timestamp(iss), color="#52514e", lw=1)
            ax.set_title(f"{g}, Ausgabe {pd.Timestamp(iss):%d.%m. %H:%M} MEZ", fontsize=10)
            ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel("Abfluss (m³/s)")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    import matplotlib.dates as mdates
    for ax in axs[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Welle Oktober 2026: Kette ohne Regenvorhersage, mit dem damaligen ICON-D2-Lauf und mit dem gemessenen Regen", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "gui" / "hindcast_2026-10.png", dpi=110)


if __name__ == "__main__":
    main()
