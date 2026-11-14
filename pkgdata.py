"""Lädt die mit import_pkg.py erzeugten 15-min-Reihen aller Pakete (1989-11 bis 2023-01) als eine durchgehende Tabelle je Größe."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PKGS = ("PKG01", "PKG02", "PKG03", "PKG04")


def load_all(var: str = "Q", folder: Path | None = None) -> pd.DataFrame:
    """Index time_mez (naiv, feste Zeitbasis wie im Download-Assistenten), Spalten = Pegelnamen. Q in m³/s, W in cm."""
    folder = folder or HERE / "data" / "lfu_pkg"
    parts = [pd.read_csv(folder / f"{p}_{var}.csv.gz", index_col=0, parse_dates=True) for p in PKGS]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


if __name__ == "__main__":
    for v in ("Q", "W"):
        d = load_all(v)
        full = pd.date_range(d.index.min(), d.index.max(), freq="15min")
        print(v, d.index.min(), d.index.max(), len(d), "fehlende Zeitschritte:", len(full.difference(d.index)),
              "fehlende Werte:", int(d.isna().sum().sum()))
