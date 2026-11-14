"""Abbildungen zu floods.py: je Hochwasser eine PNG (oben alle Pegel von Prüm und Enz übereinander, darunter jeder der acht Pegel einzeln,
72 h um den Scheitel in Prümzurlay, Abfluss in m³/s) und eine eigenständige HTML-Übersicht mit denselben Daten (SVG, Hover, Tabelle, Dunkelmodus).

Farben: kategorial in fester Reihenfolge, mit dataviz/validate_palette.js geprüft (5 Slots hell und dunkel, benachbarte Paare bestanden; Kontrast
von Aqua, Gelb und Magenta auf hell unter 3:1, deshalb Legende, Scheitelmarken mit Beschriftung und Tabellenansicht).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import floods as F

# Farbe je Pegel (Slot 1-8 der Referenzpalette, hell / dunkel); Prümzurlay als Hauptpegel zuerst
COLORS = {"Prümzurlay": ("#2a78d6", "#3987e5"), "Wiersdorf": ("#eb6834", "#d95926"), "Echtershausen": ("#1baf7a", "#199e70"),
          "Prüm 2": ("#eda100", "#c98500"), "Sinspelt": ("#e87ba4", "#d55181"), "Giesdorf": ("#008300", "#008300"),
          "Seffern": ("#6250d6", "#9085e9"), "Alsdorf-Oberecken": ("#e34948", "#e66767")}
OVERLAY = ["Prüm 2", "Echtershausen", "Wiersdorf", "Prümzurlay", "Sinspelt"]
GRID = [["Prüm 2", "Echtershausen", "Wiersdorf", "Prümzurlay"], ["Sinspelt", "Giesdorf", "Seffern", "Alsdorf-Oberecken"]]
RIVER_LABEL = {"Prüm 2": "Prüm", "Echtershausen": "Prüm", "Wiersdorf": "Prüm", "Prümzurlay": "Prüm", "Sinspelt": "Enz", "Giesdorf": "Nims",
               "Seffern": "Nims", "Alsdorf-Oberecken": "Nims"}
INK, INK2, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def clipped(g: str, w_peak: float) -> bool:
    """Wiersdorf zeigt bei vier älteren Hochwassern denselben Höchstwert (W 249 bis 250 cm, Q 137,1 m³/s): Messbereichsgrenze."""
    return g == "Wiersdorf" and w_peak >= 249 and w_peak <= 250


def flood_title(fl: dict, q: float) -> str:
    t = fl["t_peak"]
    return f"Hochwasser {t:%d.%m.%Y}: Scheitel Prümzurlay {q:.0f} m³/s ({fl['w_peak']:.0f} cm) um {t:%H:%M} MEZ"


def make_png(Q: pd.DataFrame, W: pd.DataFrame, fl: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRIDC, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    win = F.window(Q, fl)
    ww = W.loc[win.index]
    t0 = fl["t_peak"]
    fig = plt.figure(figsize=(14, 11.2), dpi=120, facecolor=SURF)
    gs = fig.add_gridspec(5, 2, height_ratios=[2.1, 1, 1, 1, 1], hspace=0.55, wspace=0.14, left=0.06, right=0.985, top=0.875, bottom=0.055)
    fig.text(0.06, 0.975, flood_title(fl, float(win["Prümzurlay"].max())), fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.947, "Abfluss in m³/s, 15-min-Werte, 72 h (36 h vor bis 36 h nach dem Scheitel in Prümzurlay), Zeit in MEZ (angenommen), "
             "Quelle LfU-Pakete PKG01 bis PKG04", fontsize=9.5, color=INK2, va="top")
    ticks = pd.date_range(win.index[0].ceil("12h"), win.index[-1], freq="12h")
    fmt = mdates.DateFormatter("%d.%m.\n%H:%M")

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.grid(axis="y", color=GRIDC, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlim(win.index[0], win.index[-1])
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(fmt)
        ax.axvline(t0, color=MUTED, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        ax.set_ylim(bottom=0)

    ax = fig.add_subplot(gs[0, :])
    style(ax)
    for g in OVERLAY:
        ax.plot(win.index, win[g], color=COLORS[g][0], linewidth=2, solid_capstyle="round", label=f"{g} ({F.RIVER[g]})", zorder=3)
        tpk = F.peak_time(win[g], t0)
        ax.plot([tpk], [win[g].loc[tpk]], "o", color=COLORS[g][0], markersize=8, markeredgecolor=SURF, markeredgewidth=2, zorder=4)
    ax.set_ylabel("m³/s")
    ax.set_ylim(0, float(win[OVERLAY].to_numpy().max()) * 1.12)
    ax.text(t0, ax.get_ylim()[1] * 0.99, "  Scheitel Prümzurlay", color=MUTED, fontsize=8.5, va="top", ha="left", zorder=5,
            bbox=dict(fc=SURF, ec="none", pad=1.0, alpha=0.9))
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, frameon=False, fontsize=9.5, handlelength=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Prüm- und Enz-Pegel übereinander", loc="left", fontsize=10.5, color=INK, pad=8, fontweight="bold")

    for c, col in enumerate(GRID):
        for r, g in enumerate(col):
            ax = fig.add_subplot(gs[1 + r, c])
            style(ax)
            col_ = COLORS[g][0]
            ax.fill_between(win.index, win[g], color=col_, alpha=0.12, linewidth=0)
            ax.plot(win.index, win[g], color=col_, linewidth=2, solid_capstyle="round", zorder=3)
            tp = F.peak_time(win[g], t0)
            qp = float(win[g].loc[tp])
            ax.plot([tp], [qp], "o", color=col_, markersize=8, markeredgecolor=SURF, markeredgewidth=2, zorder=4)
            wmax = float(ww[g].loc[t0 - pd.Timedelta(hours=30): t0 + pd.Timedelta(hours=6)].max())
            lag = (tp - t0).total_seconds() / 3600
            note = f"{qp:.0f} m³/s, {tp:%d.%m. %H:%M}" if qp >= 10 else f"{qp:.1f} m³/s, {tp:%d.%m. %H:%M}"
            note2 = f"W {wmax:.0f} cm, {lag:+.1f} h zu Prümzurlay" if g != "Prümzurlay" else f"W {wmax:.0f} cm"
            if clipped(g, wmax):
                note2 += ", bei 250 cm abgeschnitten"
            ax.set_ylim(0, qp * 1.32)
            ax.set_title(f"{g} ({RIVER_LABEL[g]})", loc="left", fontsize=10, color=INK, pad=5, fontweight="bold")
            x_frac = (tp - win.index[0]) / (win.index[-1] - win.index[0])
            ha = "left" if x_frac < 0.62 else "right"
            dx = pd.Timedelta(hours=1.2) * (1 if ha == "left" else -1)
            bb = dict(fc=SURF, ec="none", pad=0.6, alpha=0.9)
            ax.annotate(note, (tp + dx, qp * 1.24), ha=ha, va="center", fontsize=8.8, color=INK, bbox=bb, zorder=6)
            ax.annotate(note2, (tp + dx, qp * 1.08), ha=ha, va="center", fontsize=8, color=INK2, bbox=bb, zorder=6)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def make_all(Q: pd.DataFrame, floods: list[dict], outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    W = __import__("pkgdata").load_all("W")
    for fl in floods:
        p = out / f"hochwasser_{fl['t_peak']:%Y-%m-%d}.png"
        make_png(Q, W, fl, p)
        print(p)
