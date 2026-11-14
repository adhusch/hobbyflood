"""Liest ein 15-min-Paket aus dem Download-Assistenten (pegel15PKGxx.zip, alle Pegel in RLP) und schreibt die Pegel der Prüm
als breite, komprimierte Tabellen: data/lfu_pkg/<paket>_Q.csv.gz und <paket>_W.csv.gz (Index time_mez, eine Spalte je Pegelname).

  python import_pkg.py /pfad/pegel15PKG04.zip [--out data/lfu_pkg]

Format (geprüft an PKG04, 01.11.2019 bis 01.01.2023):
  Ordner q/ und w/, je Pegel eine Datei `<Nummer10>_<Name><Q|W>.csv`, UTF-8 (BOM), Trenner ';', Dezimalkomma, CRLF
  Kopf `Messst-Nr;Messstelle;Rechtswert;Hochwert;Datum;Wert;Einheit`, Datum `dd.mm.yyyy HH:MM:SS`, Q in m³/s, W vermutlich cm
  Anders als die 90-Tage-Exporte (import_wasserportal.py): anderer Kopf, Sekunden im Datum, keine fehlenden Werte als '-'.
Zeitbasis: lückenlos ohne Sommerzeit-Lücken oder doppelte Stunden, also eine feste Zeitbasis. Sie wird wie bei den Wasserportal-Exporten
als MEZ (UTC+1) angenommen, nicht unabhängig geprüft. Für Zeitversätze zwischen Pegeln desselben Pakets ist das egal.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
# 10-stellige Nummer (8 Stellen LHP + '00') -> Name wie in lags_event.NAMES
GAUGES = {"2628005000": "Prüm 2", "2628030000": "Echtershausen", "2628040200": "Wiersdorf", "2628050400": "Prümzurlay",
          "2628065000": "Sinspelt", "2628070800": "Giesdorf", "2628080000": "Seffern", "2628085400": "Alsdorf-Oberecken"}


def read_series(raw: bytes) -> tuple[str, str, pd.Series]:
    df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig")), sep=";", dtype=str, keep_default_na=False)
    need = ["Messst-Nr", "Messstelle", "Rechtswert", "Hochwert", "Datum", "Wert", "Einheit"]
    if list(df.columns) != need:
        raise ValueError(f"unerwartete Spalten {list(df.columns)}")
    unit = df["Einheit"].iloc[0]
    v = pd.to_numeric(df["Wert"].str.replace(",", ".", regex=False), errors="coerce")
    t = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M:%S")
    return df["Messst-Nr"].iloc[0], unit, pd.Series(v.to_numpy(), index=t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out", default=str(HERE / "data" / "lfu_pkg"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(a.zip).stem.replace("pegel15", "")
    cols: dict[str, dict[str, pd.Series]] = {"Q": {}, "W": {}}
    with zipfile.ZipFile(a.zip) as z:
        for i in z.infolist():
            parts = i.filename.split("/")
            if i.is_dir() or len(parts) < 3 or parts[1] not in ("q", "w"):
                continue
            num = parts[2][:10]
            if num not in GAUGES:
                continue
            nr, unit, s = read_series(z.read(i))
            var = parts[1].upper()
            assert nr == num and unit == ("m³/s" if var == "Q" else unit), (nr, num, unit)
            cols[var][GAUGES[num]] = s
    for var, d in cols.items():
        df = pd.DataFrame(d)[list(GAUGES.values())]
        df.index.name = "time_mez"
        f = out / f"{tag}_{var}.csv.gz"
        df.to_csv(f, float_format="%.5g" if var == "Q" else "%.0f")
        print(f"{f}: {len(df)} Zeilen, {df.index.min()} bis {df.index.max()}, {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
