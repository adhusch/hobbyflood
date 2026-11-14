"""Importer für CSV-Dateien aus dem Wasserportal RLP (geodaten-wasser.rlp-umwelt.de / Download-Assistent).

Format (geprüft an den Exporten vom 25.10.2026):
  Trenner ';', Dezimalkomma, fehlende Werte '-', Kopf `Messstellennummer;Messstellenbezeichnung;Datum;<Größe>`
  Größe: 'Abfluss in m3/s' oder 'Wasserstand in cm'
  Datum: 'dd.mm.yyyy HH:MM' (90-Tage-Dateien, 15 min) oder 'd.m.yyyy' ohne führende Nullen (Tagesmittel, 3 Jahre)
  Zeichensatz: ISO-8859-1 (ü ist sonst kaputt)
  Messstellennummer 10-stellig = LHP-Nummer (8 Stellen) + '00', z. B. 2628050400 -> 26280504 (Prümzurlay)

Zeitzone: Die Zeitstempel sind feste MEZ (UTC+1) das ganze Jahr, also ohne Sommerzeit. Geprüft am 25.10.2026 durch
Vergleich mit der JSON-Schnittstelle von hochwasser.rlp.de (UTC): Korrelation der 15-min-Änderungen 0,998 bzw. 1,000
nur bei einem Versatz von genau +1 h, vor und nach der Zeitumstellung am 25.10.2026. Für die Tagesmittel wird dieselbe
Zeitbasis angenommen, ungeprüft.

Niveau: Der Abfluss der CSV weicht bei Niedrigwasser von der JSON-Schnittstelle ab (Prümzurlay am 20.10.2026 0,65 gegen
0,82 m³/s, bei 6,7 m³/s gleich), vermutlich verschiedene Schlüsselkurven oder ungeprüfte Rohdaten. Deshalb werden
CSV-Reihen nicht mit data/snapshots.csv gemischt, sondern in eigene Dateien geschrieben.

Aufruf:
  python import_wasserportal.py datei1.csv datei2.csv ... [--out data/wasserportal_15min.csv]
      15-min-Dateien -> langes Format (time_utc,number,var,value), wird mit vorhandener Ausgabedatei zusammengeführt
  python import_wasserportal.py tagesmittel1.csv ... --daily [--out data/wasserportal/daily_q.csv]
      Tagesmittel -> breite Tabelle (date, eine Spalte je Pegelnummer)
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

MEZ = pd.Timedelta(hours=1)


def read_text(path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "iso-8859-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: Zeichensatz nicht erkannt")


def parse_text(text: str, source: str = "") -> pd.DataFrame:
    """Gibt Spalten number (8-stellig), name, var (Q/W), time_mez (naive Zeit in MEZ), value zurück."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, na_values=["-", ""], keep_default_na=False)
    need = ["Messstellennummer", "Messstellenbezeichnung", "Datum"]
    if list(df.columns[:3]) != need or len(df.columns) < 4:
        raise ValueError(f"{source}: unerwartete Spalten {list(df.columns)}")
    vcol = df.columns[3]
    var = "Q" if vcol.lower().startswith("abfluss") else "W" if vcol.lower().startswith("wasserstand") else None
    if var is None:
        raise ValueError(f"{source}: unbekannte Größe '{vcol}'")
    d = df["Datum"].str.strip()
    has_time = d.str.contains(":").any()
    fmt = "%d.%m.%Y %H:%M" if has_time else "%d.%m.%Y"
    out = pd.DataFrame({
        "number": df["Messstellennummer"].str.strip().str[:8],
        "name": df["Messstellenbezeichnung"],
        "var": var,
        "time_mez": pd.to_datetime(d, format=fmt),
        "value": pd.to_numeric(df[vcol].str.replace(",", ".", regex=False), errors="coerce"),
    })
    return out


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["value"]).copy()
    d["time_utc"] = (d["time_mez"] - MEZ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return d[["time_utc", "number", "var", "value"]]


def merge_long(frames, out_path: Path) -> pd.DataFrame:
    parts = list(frames)
    if out_path.exists():
        parts.insert(0, pd.read_csv(out_path, dtype={"number": str}))
    allp = pd.concat(parts, ignore_index=True)
    allp = allp.drop_duplicates(["time_utc", "number", "var"], keep="last").sort_values(["time_utc", "number", "var"])
    return allp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--daily", action="store_true", help="Tagesmittel-Dateien (breite Tabelle statt langem Format)")
    ap.add_argument("--out")
    a = ap.parse_args()
    here = Path(__file__).parent
    frames = [parse_text(read_text(f), f) for f in a.files]
    if a.daily:
        out = Path(a.out or here / "data" / "wasserportal" / "daily_q.csv")
        d = pd.concat(frames)
        d = d[d["var"] == "Q"].pivot_table(index="time_mez", columns="number", values="value", aggfunc="last")
        d.index.name = "date"
        if out.exists():
            d = d.combine_first(pd.read_csv(out, index_col=0, parse_dates=True))
        out.parent.mkdir(parents=True, exist_ok=True)
        d.sort_index().round(4).to_csv(out)
        print(f"{out}: {d.shape[0]} Tage, Pegel {', '.join(d.columns)}")
    else:
        out = Path(a.out or here / "data" / "wasserportal_15min.csv")
        allp = merge_long([to_long(f) for f in frames], out)
        out.parent.mkdir(parents=True, exist_ok=True)
        allp.to_csv(out, index=False)
        print(f"{out}: {len(allp)} Werte, {allp['number'].nunique()} Pegel, {allp['time_utc'].min()} bis {allp['time_utc'].max()}")


if __name__ == "__main__":
    main()
