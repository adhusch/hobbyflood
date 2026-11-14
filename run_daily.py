"""Baselines auf den 3-Jahres-Tagesmitteln des Wasserportals (data/wasserportal/daily_q.csv).

Tagesmittel können keine Laufzeiten im Stundenbereich auflösen, zeigen aber, wie die Modelle über mehrere Ereignisse
(bis 104 m³/s Tagesmittel in Prümzurlay) abschneiden. Der Test ist das letzte Jahr, das Training die beiden Jahre davor.
"""
import sys

import pandas as pd

import run_baselines as rb

CASES = [
    ("Echtershausen <- Prüm 2", "26280300", ["26280050"], []),
    ("Seffern <- Giesdorf", "26280800", ["26280708"], []),
    ("Alsdorf-Oberecken <- Seffern", "26280854", ["26280800"], []),
    ("Prümzurlay <- Echtershausen + Sinspelt (Wiersdorf hat keine Tagesmittel)", "26280504", ["26280300", "26280650"], []),
]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/wasserportal/daily_q.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).asfreq("D")
    test_from = df.index[-365]
    print(f"{len(df)} Tage {df.index[0]:%d.%m.%Y}–{df.index[-1]:%d.%m.%Y}, Test ab {test_from:%d.%m.%Y}")
    rb.show(rb.run_all(df, CASES, test_from, max_lag=3, own=3))
