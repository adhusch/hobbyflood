"""Dekodiert die per Browser übertragenen Wasserstände (cm) der JSON-Schnittstelle hochwasser.rlp.de für die Welle vom 24.–26.10.2026
-> data/json_w_event_2026-10.csv (langes Format time_utc,number,var,value, UTC, 15 min, 24.10. 11:00 bis 26.10. 14:30 UTC, 207 Schritte).

Einmalig benutzt (Rechenumgebung erreicht die Schnittstelle nicht; kommunale Pegel gibt es nur dort und nur 5 Tage lang).
Kodierung: 2 Zeichen je Wert (Alphabet A-Za-z0-9+_), linear: Code = Wasserstand in cm * 5 (die Werte sind ganzzahlig, daher exakt),
4095 = fehlt; zum Code des j-ten Wertes wird 1237*j (mod 4096) addiert. Prüfsummen je Block zu 500 Zeichen (a = a*31 + c mod 1000003)
stimmten mit dem Browser überein. Neurath/Arzfeld lieferte in dem Fenster keine Werte und ist nicht enthalten.
"""
from pathlib import Path

import pandas as pd

AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+_"
ORDER = [("26280402", "Wiersdorf"), ("1034232090", "Brecht"), ("1034232100", "Bettingen"), ("1034232110", "Holsthum Prüm"),
         ("26280504", "Prümzurlay"), ("1034232120", "Neurath / Arzfeld"), ("1034232130", "Neuerburg"),
         ("1034232140", "Sinspelt Gemeindehaus"), ("26280650", "Sinspelt"), ("1034232150", "Enzen"), ("1034232160", "Holsthum Schloss")]
N = 207
CHECKSUMS = [119594, 391173, 414867, 643356, 83771, 568328, 774215, 485869, 732288, 552921]
D = Path(__file__).parent / "data"

if __name__ == "__main__":
    parts = [(D / "wasserportal" / "_transfer_w" / f"c{i}.txt").read_text() for i in range(10)]
    for i, p in enumerate(parts):
        a = 0
        for ch in p:
            a = (a * 31 + ord(ch)) % 1000003
        assert a == CHECKSUMS[i], f"Block {i}: Prüfsumme {a} statt {CHECKSUMS[i]}"
    s = "".join(parts)
    assert len(s) == 2 * N * len(ORDER), len(s)
    t = pd.date_range("2026-10-24 11:00", periods=N, freq="15min").strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for k, (num, name) in enumerate(ORDER):
        for i in range(N):
            j = k * N + i
            c = (AL.index(s[2 * j]) * 64 + AL.index(s[2 * j + 1]) - 1237 * j) % 4096
            if c != 4095 and not (num == "1034232120"):
                rows.append((t[i], num, "W", c / 5))
    df = pd.DataFrame(rows, columns=["time_utc", "number", "var", "value"])
    df.to_csv(D / "json_w_event_2026-10.csv", index=False)
    print(len(df), "Zeilen")
    print(df.groupby("number")["value"].agg(["count", "min", "max"]))
