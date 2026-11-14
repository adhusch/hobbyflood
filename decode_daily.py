"""Dekodiert die per Browser übertragenen Tagesmittel (Wasserportal, 3 Jahre) -> data/wasserportal/daily_q.csv.

Einmalig benutzt, weil die Originaldateien auf dem Rechner des Nutzers liegen und nur über einen schmalen Kanal übertragen werden
konnten. Prüfsummen aller 16 Blöcke stimmten mit dem Browser überein; Mittel und Maxima der dekodierten Reihen stimmen mit den
Originalen überein. Die Werte sind durch die Kodierung auf etwa 0,15 % genau (nicht exakt gleich den Originalen).
Mit import_wasserportal.py --daily lassen sich die Originaldateien exakt einlesen und überschreiben diese Näherung.

Kodierung: 2 Zeichen je Wert (12 Bit, Alphabet A-Za-z0-9+_), logarithmisch: Wert = 0.005 * exp((code-1)/K),
K = 4094/ln(1000/0.005); relative Auflösung ca. 0,3 %. 'AA' = 0, '~~' = fehlt.
Reihenfolge der Pegel (Messstellennummer, 10-stellig): 2628050400 Prümzurlay, 2628005000 Prüm 2, 2628030000 Echtershausen,
2628065000 Sinspelt, 2628070800 Giesdorf, 2628080000 Seffern, 2628085400 Alsdorf-Oberecken; je 1096 Tage ab 25.10.2023.
"""
import math
from pathlib import Path

import pandas as pd

AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+_"
K = 4094 / math.log(1000 / 0.005)
ORDER = ["2628050400", "2628005000", "2628030000", "2628065000", "2628070800", "2628080000", "2628085400"]
N = 1096
D = Path(__file__).parent / "data" / "wasserportal"


def dec(pair: str) -> float:
    if pair == "~~":
        return float("nan")
    c = AL.index(pair[0]) * 64 + AL.index(pair[1])
    return 0.0 if c == 0 else 0.005 * math.exp((c - 1) / K)


if __name__ == "__main__":
    s = "".join((D / "_transfer" / f"chunk_{i:02d}.txt").read_text() for i in range(16))
    assert len(s) == 2 * N * len(ORDER), len(s)
    idx = pd.date_range("2023-10-25", periods=N, freq="D")
    cols = {}
    for k, num in enumerate(ORDER):
        part = s[k * 2 * N:(k + 1) * 2 * N]
        cols[num[:8]] = [dec(part[2 * i:2 * i + 2]) for i in range(N)]
    df = pd.DataFrame(cols, index=idx)
    df.index.name = "date"
    df.round(4).to_csv(D / "daily_q.csv")
    print(df.describe().T[["count", "mean", "max"]].round(2))
    print(df.idxmax())
