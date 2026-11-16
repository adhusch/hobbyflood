"""Stausee Bitburg (zwischen Echtershausen und Wiersdorf): Regeleingriffe in den Hochwassern erkennen.

Je Hochwasser (Scheitel in Echtershausen ab 40 m³/s) der Vergleich Abgabe in Wiersdorf gegen Zufluss in Echtershausen 2 h vorher:
  scheitel_verh   Scheitel Wiersdorf / Scheitel Echtershausen (ohne Regelung knapp über 1, weil unterwegs noch Wasser zufließt)
  vorabgabe       größter Überschuss Wiersdorf - Echtershausen in den 24 h vor dem Scheitel (m³/s und relativ zum Zufluss)
  rueckhalt       größter Rückstand Echtershausen - Wiersdorf um den Scheitel (m³/s)
  rueckhalt_mio   zurückgehaltenes Volumen bis zum Scheitel in Wiersdorf (Mio m³)
Ein Eingriff gilt als deutlich, wenn der Scheitel um mehr als 15 % gekappt ist oder die Vorabgabe 10 m³/s und 30 % des Zuflusses übersteigt.
Ausgabe: data/stausee_eingriffe.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pkgdata

HERE = Path(__file__).parent
LAG = 8          # 2 h in Viertelstunden


def main() -> None:
    Q, W = pkgdata.load_all("Q"), pkgdata.load_all("W")
    ec, wi, ww = Q["Echtershausen"], Q["Wiersdorf"], W["Wiersdorf"]
    x = ec.to_numpy()
    rows, last = [], -10 ** 9
    order = np.argsort(-np.nan_to_num(x))
    peaks = []
    for i in order:
        if not np.isfinite(x[i]) or x[i] < 40:
            break
        if all(abs(i - j) > 5 * 96 for j in peaks):
            peaks.append(i)
    for i in sorted(peaks):
        a, b = i - 36 * 4, i + 24 * 4
        if a < LAG or b >= len(x):
            continue
        e = ec.to_numpy()[a - LAG:b - LAG]            # Zufluss, 2 h verschoben
        w = wi.to_numpy()[a:b]
        if not (np.isfinite(e).all() and np.isfinite(w).all()):
            continue
        d = w - e
        ip = i - a + LAG                              # Scheitel des verschobenen Zuflusses
        pre = d[max(0, ip - 96):ip]
        j = int(np.argmax(pre))
        iw = int(np.argmax(w))
        clip_h = float((ww.to_numpy()[a:b] >= 249.5).sum() / 4)
        ratio = float(w.max() / e.max())
        vor, vor_rel = float(pre[j]), float(pre[j] / max(e[max(0, ip - 96) + j], 1.0))
        rows.append(dict(scheitel_ec=str(ec.index[i]), q_ec=round(float(x[i]), 1), q_wi=round(float(w.max()), 1), scheitel_verh=round(ratio, 2),
                         vorabgabe=round(vor, 1), vorabgabe_rel=round(vor_rel, 2), rueckhalt=round(float(-d[max(0, ip - 12):ip + 24].min()), 1),
                         rueckhalt_mio=round(float(-np.minimum(d[:max(iw, ip)], 0).sum() * 900 / 1e6), 2), wi_am_anschlag_h=clip_h,
                         deutlich=bool(clip_h < 2 and (ratio < 0.85 or (vor > 10 and vor_rel > 0.3)))))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "data" / "stausee_eingriffe.csv", index=False)
    pd.set_option("display.width", 220)
    print(len(df), "Hochwasser; Scheitelverhältnis Median", round(df.scheitel_verh.median(), 2), "10-90 %", df.scheitel_verh.quantile([.1, .9]).round(2).tolist())
    print("deutliche Eingriffe:", int(df.deutlich.sum()))
    print(df[df.deutlich | (df.q_ec > 120)].to_string(index=False))
    print("Scheitelverhältnis nach Größe:", df.groupby(pd.cut(df.q_ec, [40, 60, 90, 130, 500])).scheitel_verh.agg(["median", "min", "size"]).round(2).to_dict("index"))


if __name__ == "__main__":
    main()
