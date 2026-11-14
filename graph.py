"""Pegelgraph der Prüm aus pegel_netz.csv.

Knoten mit Abfluss: LfU-Pegel (art == "LfU") und der Stausee (art == "Speicher").
Kommunale Pegel (nur Wasserstand) werden beim Aufwärtslaufen übersprungen.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).with_name("pegel_netz.csv")
Q_ART = {"LfU", "Speicher"}


def load(path: Path = CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8").fillna("").set_index("lhp_id")


def parents(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        i: [p for p in (r.oberlieger_lhp_id, r.zufluss_lhp_id) if p]
        for i, r in df.iterrows()
    }


def number(lhp_id: str) -> str:
    """LHP-ID -> Messstellennummer der hochwasser.rlp.de-API (wie in lhpapi: ident[3:])."""
    return lhp_id[3:]


def q_reaches(df: pd.DataFrame) -> dict[str, dict]:
    """Für jeden Abflussknoten die nächsten Abfluss-Oberlieger.

    Rückgabe: {ziel_id: {"sources": [ids], "reservoir": bool}}
    reservoir=True, wenn das Ziel oder eine Quelle der Stausee ist (Routing gilt dort nicht).
    """
    par = parents(df)
    nodes = {i for i in df.index if df.at[i, "art"] in Q_ART}

    def upstream(i: str, seen: set[str]) -> list[str]:
        out: list[str] = []
        for p in par.get(i, []):
            if p in seen or p not in df.index:
                continue
            seen.add(p)
            out += [p] if p in nodes else upstream(p, seen)
        return out

    res = {}
    for t in nodes:
        src = upstream(t, {t})
        special = df.at[t, "art"] == "Speicher" or any(df.at[s, "art"] == "Speicher" for s in src)
        res[t] = {"sources": src, "reservoir": special}
    return res


if __name__ == "__main__":
    d = load()
    for t, v in q_reaches(d).items():
        names = [d.at[s, "name"] for s in v["sources"]] or ["(Quellgebiet)"]
        flag = "  [Speicher]" if v["reservoir"] else ""
        print(f"{d.at[t, 'name']:<20} <- {', '.join(names)}{flag}")
