"""Baut gui/eval.html: Vorhersagen an historischen Hochwassern nachrechnen und bewerten (eine HTML-Datei, rechnet im Browser mit forecast_core.js).

  python eval_view.py [--out gui/eval.html]        (nach chain_fc.py run und uncert.py tables)

Ereignisse: jedes Hochwasser 1989-2022, bei dem ein Pegel 0,7 HQ2 erreicht (Mai bis September ab 2004: 0,4 HQ2, damit auch kleinere
Starkregenhochwasser dabei sind); Scheitel im Abstand von weniger als 72 h bilden ein Ereignis. Fenster wie chain_fc.flood_windows: Abfluss aller
acht Pegel ab Bezugsscheitel - 96 h bis + 60 h, Regen ab - 168 h. Modellsatz: ohne den Zeitblock des Ereignisses (kreuzvalidiert), vor 2004 der
Satz aus 2004-2022 (unabhängig, dort ohne Regen).

Braucht die Pegeldaten der LfU-Pakete; die fertige Seite enthält die gemessenen Ganglinien der Ereignisse (Quelle: Landesamt für Umwelt
Rheinland-Pfalz) und gehört deshalb nicht ins öffentliche Repository, solange das nicht geklärt ist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain_fc as C
import routing as R
import routing_map as RM
import uncert as UN

HERE = Path(__file__).parent
THR_HQ2, THR_HQ2_STORM = 0.7, 0.4
MERGE_H = 72


def find_events(Q: pd.DataFrame, K: dict) -> list[list[tuple]]:
    """Je Ereignis die Scheitel der Pegel über der Schwelle: (Zeit, Pegelschlüssel, Abfluss / HQ2)."""
    pk = []
    for k in C.ORDER:
        dn = C.DN[k]
        s, hq2 = Q[dn], K[dn]["Q"]["HQ2"]
        summer = (s.index.month >= C.STORM_MONTHS[0]) & (s.index.month <= C.STORM_MONTHS[1]) & (s.index.year >= 2004)
        m = s > np.where(summer, THR_HQ2_STORM, THR_HQ2) * hq2
        grp = (m != m.shift()).cumsum()
        for _, x in s[m].groupby(grp[m]):
            pk.append((x.idxmax(), k, float(x.max() / hq2)))
    pk.sort()
    out: list[list[tuple]] = []
    for p in pk:
        if out and (p[0] - out[-1][-1][0]) < pd.Timedelta(hours=MERGE_H):
            out[-1].append(p)
        else:
            out.append([p])
    return out


def hq_class(q: float, kq: dict) -> str:
    lab = "unter HQ2"
    for x in ("2", "10", "20", "50", "100"):
        if q >= kq["HQ" + x]:
            lab = "HQ" + x
    return lab


def windows() -> list[dict]:
    Q, W, frames = UN._S["Q"], UN._S["W"], UN._S["frames"]
    K = UN.kennwerte()
    a0, b0 = int(96 / C.DT), int(60 / C.DT)
    r0 = a0 + C.LEAD
    out, ids = [], set()
    for ev in find_events(Q, K):
        tref, kref, _ = max(ev, key=lambda p: p[2])
        ip = int(Q.index.get_loc(tref))
        if ip - r0 - 1 < 0 or ip + b0 + 1 >= len(Q):
            continue
        yr = tref.year
        blk = [i for i, (y0, y1) in enumerate(C.BLOCKS) if y0 <= yr <= y1]
        rain_ok = all(np.isfinite(frames[k][c].to_numpy()[ip - r0:ip + b0 + 1]).all() for k in C.RAINK for c in ("p", "api", "api_s"))
        fid = f"{tref:%Y-%m-%d}"
        while fid in ids:
            fid += "b"
        ids.add(fid)
        sl = slice(ip - a0, ip + b0 + 1)
        rec = dict(id=fid, t_peak=str(tref), t_start=str(Q.index[ip - a0]), i_peak=a0, ref=kref, models=str(blk[0]) if blk else "alle", rain=bool(rain_ok),
                   q={C.DN[k]: [None if not np.isfinite(v) else round(float(v), 3) for v in Q[C.DN[k]].to_numpy()[sl]] for k in C.ORDER})
        peaks = {}
        for k in C.ORDER:
            q = Q[C.DN[k]].to_numpy()[sl]
            if not np.isfinite(q).any():
                continue
            j = int(np.nanargmax(q))
            w = W[C.DN[k]].to_numpy()[sl]
            peaks[k] = dict(i=j, q=round(float(q[j]), 2), klasse=hq_class(float(q[j]), K[C.DN[k]]["Q"]), rel=round(float(q[j] / K[C.DN[k]]["Q"]["HQ10"]), 3),
                            w_damals=None if not np.isfinite(w[j]) else int(round(float(w[j]))))
        rec["peaks"] = peaks
        rec["wi_abgeschnitten"] = bool(R.clipped_window(W["Wiersdorf"].to_numpy()[sl][None, :], Q["Wiersdorf"].to_numpy()[sl][None, :])[0])
        if rain_ok:
            rec["storm"] = {k: "".join("1" if x else "0" for x in C.is_storm(k, np.arange(ip - a0, ip + b0 + 1))) for k in C.ORDER}
            rec["rain_start"] = str(Q.index[ip - r0])
            rec["p"] = {k: [round(float(v), 3) for v in frames[k]["p"].to_numpy()[ip - r0:ip + b0 + 1]] for k in C.RAINK}
            rec["api0"] = {k: [float(frames[k]["api"].to_numpy()[ip - r0 - 1]), float(frames[k]["api_s"].to_numpy()[ip - r0 - 1])] for k in C.RAINK}
            rec["art"] = "Starkregen" if any(rec["storm"][k][peaks[k]["i"]] == "1" for k in peaks) else "Winter / Dauerregen"
        else:
            rec["art"] = "ohne Regendaten"
        # Zustandsklassen je Pegel und Index (wie uncert.flood_classes)
        t0 = np.arange(ip - a0, ip + b0 + 1)
        codes = {}
        for k in C.ORDER:
            ft = UN.features_at(k, t0)
            st = UN.state_class(ft, k, use_rain=bool(rain_ok))
            code = st + (4 * ft["hp"] + 8 * ft["hf"] if rain_ok else 0)
            codes[k] = "".join(format(int(c), "x") for c in code)
        rec["codes"] = codes
        out.append(rec)
    return out


def build(out: Path) -> None:
    UN.setup()
    evs = windows()
    floods = [dict(id=e["id"], fcw=e) for e in evs]
    unc = RM.load_uncert(floods)
    unc["floods"] = {e["id"]: e.pop("codes") for e in evs}
    U = json.loads((HERE / "data" / "uncert_tables.json").read_text(encoding="utf-8"))
    unc["level_edges"], unc["wet_edge"] = U.get("level_edges"), U.get("wet_edge")
    cm = json.loads((HERE / "data" / "chain_models.json").read_text(encoding="utf-8"))
    used = {e["models"] for e in evs}
    pk = pd.read_csv(HERE / "data" / "chain_peak.csv")
    pk = pk[pk.before_h.isin([24, 12, 6, 3]) & pk.variant.isin(["kette", "kette_regen", "kette_ideal", "pers"])]
    info = UN._S.get("info") or C._S["info"]
    data = dict(events=evs, sets={k: v for k, v in cm["sets"].items() if k in used}, consts=cm["consts"], unc=unc,
                area={k: info[k]["area_km2"] for k in C.RAINK},
                peak=pk[["set", "target", "variant", "before_h", "ratio_med", "within20", "events"]].round(3).to_dict("records"),
                stand=pd.Timestamp.now().strftime("%d.%m.%Y"))
    tpl = (HERE / "eval_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("/*__FC_CORE__*/", (HERE / "forecast_core.js").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_rain = sum(e["rain"] for e in evs)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB, {len(evs)} Ereignisse, davon {n_rain} mit Regen")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "gui" / "eval.html"))
    build(Path(ap.parse_args().out))
