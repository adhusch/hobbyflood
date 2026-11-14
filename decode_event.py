"""Dekodiert die per Browser übertragenen 15-min-Werte des Ereignisses vom 24.–26.10.2026 (Wasserportal, 90-Tage-Export)
-> data/wasserportal_event_2026-10.csv (langes Format time_utc,number,var,value).

Einmalig benutzt, weil das Netz der Rechenumgebung die Wasserportal-Adresse sperrt und der Rechner des Nutzers sie über den
Proxy ebenfalls nicht erreicht. Alle 8 Blöcke zu 500 Zeichen stimmten in der Prüfsumme (a = (a*31 + c) mod 1000003) mit dem
Browser überein. Mit import_wasserportal.py lassen sich die Originaldateien exakt einlesen und überschreiben diese Näherung.

Kodierung: 2 Zeichen je Wert (12 Bit, Alphabet A-Za-z0-9+_), logarithmisch: Wert = 0.005 * exp((code-1)/K),
K = 4094/ln(1000/0.005), relative Auflösung ca. 0,3 % (Fehler bis ±0,15 %). 0 = 'AA', 4095 = fehlt. Zusätzlich wird zum Code
des j-ten Wertes 1237*j (mod 4096) addiert, damit lange Folgen gleicher Werte nicht gleich aussehen (beim Abtippen gingen sonst
Zeichen verloren). Reihenfolge: 9 Pegel nacheinander, je 206 Werte ab 24.10.2026 12:00 MEZ in 15-min-Schritten (letzter 26.10. 15:15 MEZ).
MEZ = feste UTC+1 (siehe import_wasserportal.py), die Ausgabe ist in UTC.
"""
import math
from pathlib import Path

import pandas as pd

AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+_"
K = 4094 / math.log(1000 / 0.005)
# (10-stellige Nummer, Name, Größe)
ORDER = [("2628050400", "Prümzurlay", "Q"), ("2628005000", "Prüm 2", "Q"), ("2628030000", "Echtershausen", "Q"),
         ("2628040200", "Wiersdorf", "Q"), ("2628065000", "Sinspelt", "Q"), ("2628070800", "Giesdorf", "Q"),
         ("2628080000", "Seffern", "Q"), ("2628085400", "Alsdorf-Oberecken", "Q"), ("2628036600", "Stausee Bitburg", "W")]
N = 206
CHECKSUMS = [581981, 783084, 563361, 723714, 622000, 339872, 404079, 748121]  # im Browser berechnet, je Block
T0_MEZ = pd.Timestamp("2026-10-24 12:00")
D = Path(__file__).parent / "data"


def dec(pair: str, j: int) -> float:
    c = (AL.index(pair[0]) * 64 + AL.index(pair[1]) - 1237 * j) % 4096
    if c == 4095:
        return float("nan")
    return 0.0 if c == 0 else 0.005 * math.exp((c - 1) / K)


if __name__ == "__main__":
    parts = [(D / "wasserportal" / "_transfer90" / f"c{i}.txt").read_text() for i in range(8)]
    for i, part in enumerate(parts):
        a = 0
        for ch in part:
            a = (a * 31 + ord(ch)) % 1000003
        assert a == CHECKSUMS[i], f"Block {i}: Prüfsumme {a} statt {CHECKSUMS[i]}"
    s = "".join(parts)
    assert len(s) == 2 * N * len(ORDER), len(s)
    times = (T0_MEZ + pd.to_timedelta([15 * i for i in range(N)], unit="min") - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for k, (num, name, var) in enumerate(ORDER):
        for i in range(N):
            j = k * N + i
            v = dec(s[2 * j:2 * j + 2], j)
            if v == v:
                rows.append((times[i], num[:8], var, round(v, 4)))
    df = pd.DataFrame(rows, columns=["time_utc", "number", "var", "value"]).sort_values(["time_utc", "number", "var"])
    out = D / "wasserportal_event_2026-10.csv"
    df.to_csv(out, index=False)
    print(len(df), "Zeilen ->", out)
    print(df.groupby("number")["value"].agg(["count", "min", "max"]).round(3))
