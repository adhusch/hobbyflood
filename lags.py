"""Laufzeitschätzung zwischen Pegelpaaren mit drei Verfahren und Plausibilitätsprüfung über Flusskilometer.

  python lags.py [data/real_5d.csv]

Achtung: Alle drei Verfahren messen die Reaktionsverzögerung zwischen zwei Ganglinien, nicht die Wellenlaufzeit.
Bei flächigem Regen steckt darin der Zwischenzufluss. Ein einzelnes Ereignis sagt nichts über die Abflussabhängigkeit.
"""
import sys
import pandas as pd, numpy as np
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/real_5d.csv"
s = pd.read_csv(PATH, parse_dates=["time_utc"], dtype={"number": str})
w = s[s["var"] == "Q"].pivot(index="time_utc", columns="number", values="value")
nm = {"26280050":"Prüm 2","26280300":"Echtershausen","26280402":"Wiersdorf","26280504":"Prümzurlay","26280650":"Sinspelt","26280708":"Giesdorf","26280800":"Seffern","26280854":"Alsdorf"}
w = w.rename(columns=nm)
pairs = [("Prüm 2","Echtershausen"),("Echtershausen","Wiersdorf"),("Wiersdorf","Prümzurlay"),("Sinspelt","Prümzurlay"),("Giesdorf","Seffern"),("Seffern","Alsdorf")]
dt_h = 0.25
def xcorr(a, b, maxlag):
    """Korrelation von a(t-L) mit b(t) für L=0..maxlag (in Schritten)."""
    out = []
    for L in range(maxlag + 1):
        x = a[:len(a) - L] if L else a
        y = b[L:]
        out.append(np.corrcoef(x, y)[0, 1])
    return np.array(out)

print("== A) Methode der ersten Schätzung: Pearson r der Q-Werte, ganze 5 Tage, Lag 0..24 h")
for u, d in pairs:
    r = xcorr(w[u].to_numpy(), w[d].to_numpy(), 96)
    best = int(np.nanargmax(r))
    plateau = np.where(r >= r[best] - 0.005)[0]
    print(f"{u:>14} -> {d:<14} bester Lag {best*dt_h:5.2f} h (r={r[best]:.3f}), Lags innerhalb 0,005 des Maximums: {plateau.min()*dt_h:.2f} bis {plateau.max()*dt_h:.2f} h")

print("\n== B) Korrelation der 1-h-Änderungen (dQ), ganze 5 Tage, Lag 0..12 h")
for u, d in pairs:
    a = w[u].diff(4).to_numpy()[4:]; b = w[d].diff(4).to_numpy()[4:]
    r = xcorr(a, b, 48)
    best = int(np.nanargmax(r))
    print(f"{u:>14} -> {d:<14} bester Lag {best*dt_h:5.2f} h (r={r[best]:.3f}, r bei Lag 0: {r[0]:.3f})")

print("\n== C) Durchgangszeit: Zeitpunkt, an dem ein Anteil f des bisherigen Anstiegs erreicht wird (Anstieg ab 24.10. 21:00 UTC, Spitze noch nicht erreicht)")
t0 = pd.Timestamp("2026-10-24 21:00", tz="UTC")
def cross(series, frac):
    seg = series[series.index >= t0]
    base = seg.iloc[:4].mean()
    top = seg.max()
    lvl = base + frac * (top - base)
    # erstes Überschreiten, linear interpoliert; vorher darf es nicht schon darüber liegen
    v = seg.to_numpy(); idx = np.where(v >= lvl)[0]
    if len(idx) == 0 or idx[0] == 0: return None
    i = idx[0]
    f = (lvl - v[i-1]) / (v[i] - v[i-1]) if v[i] != v[i-1] else 0
    return seg.index[i-1] + (seg.index[i] - seg.index[i-1]) * f, lvl
fr = [0.1, 0.25, 0.5, 0.75]
print("Anteil f:           " + "  ".join(f"{f:>5.2f}" for f in fr))
for u, d in pairs:
    row = []
    for f in fr:
        cu, cd = cross(w[u], f), cross(w[d], f)
        row.append(f"{(cd[0]-cu[0]).total_seconds()/3600:5.2f}" if cu and cd else "  -  ")
    print(f"{u:>14} -> {d:<14} " + "  ".join(row) + "  h")
print("\nAbfluss am Unterlieger bei diesen Anteilen (m³/s):")
for u, d in pairs:
    print(f"{d:<14} (von {u}): " + "  ".join((f"{cross(w[d], f)[1]:5.2f}" if cross(w[d], f) else "  -  ") for f in fr))

print("\n== Plausibilität: Fließstrecke aus Flusskilometern, implizite Wellengeschwindigkeit")
km = {"Prüm 2": 79.53, "Echtershausen": 38.44, "Wiersdorf": 32.14, "Prümzurlay": 7.86, "Giesdorf": 53.49, "Seffern": 35.04, "Alsdorf": 6.23}
for u, d in [p for p in pairs if p[0] in km and p[1] in km]:
    L = (km[u] - km[d]) * 1000
    ra = xcorr(w[u].to_numpy(), w[d].to_numpy(), 96); la = int(np.nanargmax(ra)) * 900
    rb = xcorr(w[u].diff(4).to_numpy()[4:], w[d].diff(4).to_numpy()[4:], 48); lb = int(np.nanargmax(rb)) * 900
    f = lambda t: f"{L / t:5.1f} m/s" if t else "   inf"
    print(f"{u:>14} -> {d:<14} {L/1000:5.1f} km | Verfahren A: {f(la)} | Verfahren B: {f(lb)}")
print("Typische Hochwasserwellen in Mittelgebirgsbächen: grob 1 bis 3 m/s. Deutlich darüber heißt: Die Reaktion kommt aus dem Zwischengebiet, nicht aus der Welle des Oberliegers.")
