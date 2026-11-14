"""Abruf der Pegeldaten über die JSON-Schnittstelle von hochwasser.rlp.de.

Die Endpunkte stammen aus dem Quellcode der Bibliothek lhpapi (rp_api.py):
  GET https://hochwasser.rlp.de/api/v1/config
  GET https://hochwasser.rlp.de/api/v1/measurement-site/<nummer>
Geprüft (25.10.2026, im Browser von hochwasser.rlp.de aus): measurement-site liefert pro Größe W und Q
ein 5-Tage-Fenster `measurements` [{x: UTC-Zeit, y: Wert}] in 15-min-Schritten, dazu xLast/yLast und
`extremeevents` (historische Spitzen). `config` enthält Koordinaten (UTM32), Einzugsgebietsfläche, Pegelnull.
Der Server sendet keinen CORS-Header, deshalb funktioniert der Abruf nur serverseitig (dieses Skript) oder
von der Seite hochwasser.rlp.de selbst, nicht aus einer lokalen HTML-Datei.
Nutzungsbedingungen sind nicht geklärt. Abrufe sparsam halten.

Befehle:
  python fetch_rlp.py probe      rohe JSON-Antworten nach data/raw/ speichern und Schlüssel anzeigen
  python fetch_rlp.py snapshot   5-Tage-Fenster aller Pegel mit data/snapshots.csv zusammenführen
  python fetch_rlp.py loop       snapshot alle 6 Stunden wiederholen (Strg+C beendet)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import graph

BASE = "https://hochwasser.rlp.de/api/v1"
HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
SNAP = HERE / "data" / "snapshots.csv"
HEADERS = {"User-Agent": "pruem-ro/0.1 (Forschung, Hobby-Projekt)"}


def get_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def numbers() -> dict[str, str]:
    d = graph.load()
    return {graph.number(i): d.at[i, "name"] for i in d.index if d.at[i, "art"] != "Luxemburg"}


def probe() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = get_json(f"{BASE}/config")
    (RAW / f"config_{stamp}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print("config-Schlüssel:", list(cfg))
    sites = cfg.get("measurementsite", {})
    for num, name in numbers().items():
        site = sites.get(num) or next((s for s in sites.values() if s.get("number") == num), None)
        print(f"{num:>12} {name:<22} in config: {'ja' if site else 'NEIN'}")
        try:
            data = get_json(f"{BASE}/measurement-site/{num}")
        except requests.RequestException as e:
            print("   Fehler:", e)
            continue
        (RAW / f"site_{num}_{stamp}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
        for var in ("W", "Q"):
            if isinstance(data.get(var), dict):
                print(f"   {var}: {sorted(data[var])}")
        time.sleep(0.5)


def _parse_utc(s: str) -> str:
    return datetime.fromisoformat(s.rstrip("Z") + "+00:00").astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot() -> None:
    """Holt für jeden Pegel das ganze 5-Tage-Fenster (15-min-Werte) und führt es mit data/snapshots.csv zusammen.

    Das Fenster ist 5 Tage lang. Wer mindestens alle 4 Tage läuft, bekommt eine lückenlose Historie.
    """
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str, str], str] = {}
    if SNAP.exists():
        with SNAP.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["time_utc"], r["number"], r["var"])] = r["value"]
    before = len(rows)
    for num in numbers():
        try:
            data = get_json(f"{BASE}/measurement-site/{num}")
        except requests.RequestException as e:
            print(num, "Fehler:", e, file=sys.stderr)
            continue
        for var in ("W", "Q"):
            d = data.get(var)
            if isinstance(d, dict):
                for m in d.get("measurements") or []:
                    if m.get("y") is not None and m.get("x"):
                        rows[(_parse_utc(m["x"]), num, var)] = str(m["y"])
        time.sleep(0.5)
    with SNAP.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "number", "var", "value"])
        for (t, n, v), val in sorted(rows.items()):
            w.writerow([t, n, v, val])
    print(f"{datetime.now().isoformat(timespec='seconds')}: {len(rows) - before} neue Werte, {len(rows)} insgesamt")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "probe":
        probe()
    elif cmd == "snapshot":
        snapshot()
    elif cmd == "loop":
        while True:
            snapshot()
            time.sleep(6 * 3600)
    else:
        print(__doc__)
