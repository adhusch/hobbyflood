"""Liest aus einem Pegelverlauf-Screenshot des Hochwasserportals (zwei Kurven, rot und blau, Zeitachse mit Tagesstrichen) die
Scheitelzeiten und den Zeitversatz je Anteil des eigenen Anstiegs ab.

  python screenshot_lag.py bild.png --first-tick 2018-05-27 [--up blau --down rot] [--axis-y 463:468] [--plot-top 110]

Vorgehen: Tagesstriche unter der Achse (dunkle Pixel in den Zeilen --axis-y) geben Nullpunkt und Pixel je Stunde. Je Kurve wird pro
Bildspalte die Oberkante der gefärbten Pixel genommen, Lücken (verdeckt) werden linear gefüllt. Der Anteil f bezieht sich auf den eigenen
Anstieg der Kurve über ihrer Ausgangslage, also unabhängig vom Maßstab der y-Achse (bei zwei überlagerten Achsen wichtig). Die Kurven sind
Wasserstände, keine Abflüsse: Schlüsselkurven verschieben die f-Anteile, die Scheitelzeiten nicht. Ablesegenauigkeit etwa 0,25 h.
Die Uhrzeit-Basis (MEZ/MESZ) ist für Zeitversätze zwischen den Kurven egal, solange beide dieselbe Achse nutzen.
"""
import argparse

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--first-tick", required=True, help="Datum des ersten Tagesstrichs (Mitternacht), z. B. 2018-05-27")
ap.add_argument("--up", default="blau", help="Kurve des Oberliegers (blau oder rot)")
ap.add_argument("--down", default="rot")
ap.add_argument("--axis-y", default="463:468", help="Bildzeilen der Tagesstriche, von:bis")
ap.add_argument("--plot-top", type=int, default=110, help="Zeilen darüber ignorieren (Titel, Beschriftung)")
ap.add_argument("--plot-bottom", type=int, default=445)
a = ap.parse_args()

im = np.array(Image.open(a.image).convert("RGB")).astype(int)
R, G, B = im[..., 0], im[..., 1], im[..., 2]
y0, y1 = map(int, a.axis_y.split(":"))
dark = (R < 80) & (G < 80) & (B < 80)
cols = np.where(dark[y0:y1].sum(axis=0) >= (y1 - y0) - 1)[0]
ticks = []  # benachbarte Spalten (Strichbreite 2 px) zusammenfassen
for c in cols:
    if ticks and c - ticks[-1][-1] <= 2:
        ticks[-1].append(c)
    else:
        ticks.append([c])
tx = np.array([np.mean(t) for t in ticks])
tx = tx[1:] if len(tx) > 1 and (tx[1] - tx[0]) < 0.5 * np.median(np.diff(tx)) else tx
px_day = float(np.median(np.diff(tx)))
x0, pxh = float(tx[0]), px_day / 24
print(f"{len(tx)} Tagesstriche, {px_day:.1f} px je Tag, erster Strich bei x = {x0:.1f} ({a.first_tick} 00:00)")

masks = {"rot": (R > 170) & (G < 100) & (B < 100), "blau": (B > 160) & (R < 90) & (G < 130)}
env = {}
for k, m in masks.items():
    m = m.copy()
    m[:a.plot_top] = False
    m[a.plot_bottom:] = False
    m[:, :int(x0) + 2] = False
    m[:, int(tx[-1]):] = False
    xs = np.arange(int(x0) + 2, int(tx[-1]))
    u = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        ys = np.where(m[:, x])[0]
        if len(ys):
            u[i] = ys.min()
    ok = ~np.isnan(u)
    env[k] = (xs, np.interp(xs, xs[ok], u[ok]), int(xs[ok].max()))


def clock(x):
    h = (x - x0) / pxh
    import datetime as dt
    return (dt.datetime.fromisoformat(a.first_tick) + dt.timedelta(hours=h)).strftime("%d.%m. %H:%M")


res = {}
for k, (xs, u, last) in env.items():
    mid = len(xs) // 2
    base = float(np.median(u[: int(0.4 * len(xs))]))
    ip = int(np.argmin(u))
    rise, fall = {}, {}
    for fr in (0.1, 0.25, 0.5, 0.75, 0.9):
        L = base - fr * (base - u[ip])
        j = next(i for i in range(1, ip + 1) if u[i] <= L and u[i - 1] > L)
        rise[fr] = xs[j - 1] + (u[j - 1] - L) / (u[j - 1] - u[j]) * (xs[j] - xs[j - 1])
        jf = next((i for i in range(ip + 1, len(xs)) if xs[i] <= last and u[i] >= L), None)
        fall[fr] = None if jf is None else xs[jf - 1] + (L - u[jf - 1]) / (u[jf] - u[jf - 1]) * (xs[jf] - xs[jf - 1])
    res[k] = dict(xp=float(xs[ip]), rise=rise, fall=fall, last=last)
    print(f"{k}: Scheitel {clock(xs[ip])}, letzter Datenpunkt {clock(last)}")
up, dn = res[a.up], res[a.down]
print(f"Scheitel-Lag {a.down} minus {a.up}: {(dn['xp'] - up['xp']) / pxh:.2f} h")
print("Anteil f   steigend h   fallend h")
for fr in (0.1, 0.25, 0.5, 0.75, 0.9):
    f_ = None if dn["fall"][fr] is None or up["fall"][fr] is None else (dn["fall"][fr] - up["fall"][fr]) / pxh
    print(f"  {fr:4.2f}   {(dn['rise'][fr] - up['rise'][fr]) / pxh:9.2f}   {'–' if f_ is None else f'{f_:9.2f}'}")
