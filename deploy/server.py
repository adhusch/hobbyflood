"""Webserver des Containers: liefert gui/ aus wie `python -m http.server` und packt auf Anfrage Rohdaten und Seiten als Zip.

    python deploy/server.py [PORT] [--root /app]

    /download?stunden=24&was=alles     Zip mit allem aus den letzten 24 Stunden (stunden=0: alles)
              was=roh                  nur Rohdaten (data/live/raw_*: Pegel, Stationsregen, ICON; dazu data/live/*.json)
              was=seiten               nur Seiten (gui/archiv/live_*.html, dazu live.html und auswertung.html)
    /download/info                     JSON: Zahl der Läufe und Größe je Zeitraum (für die Live-Seite)

Das Zip wird beim Senden erzeugt (nichts auf Vorrat). Zeitraum nach Änderungszeit der Ordner bzw. Dateien. Nur feste Verzeichnisse,
keine Pfade aus der Anfrage.
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
MAX_H = 24 * 365


def _collect(root: Path, hours: float, was: str) -> list[tuple[Path, str]]:
    cutoff = time.time() - hours * 3600 if hours > 0 else 0.0
    live, gui = root / "data" / "live", root / "gui"
    files: list[tuple[Path, str]] = []
    if was in ("alles", "roh"):
        for d in sorted(live.glob("raw_*")):
            if d.is_dir() and d.stat().st_mtime >= cutoff:
                files += [(f, f"roh/{d.name}/{f.relative_to(d).as_posix()}") for f in sorted(d.rglob("*")) if f.is_file()]
        files += [(f, f"roh/{f.name}") for f in sorted(live.glob("*.json")) if f.is_file()]
    if was in ("alles", "seiten"):
        files += [(f, f"seiten/archiv/{f.name}") for f in sorted((gui / "archiv").glob("live_*.html")) if f.stat().st_mtime >= cutoff]
        files += [(gui / n, f"seiten/{n}") for n in ("live.html", "auswertung.html") if (gui / n).is_file()]
    return files


def _info(root: Path) -> dict:
    out = {}
    for h in (24, 72, 168, 0):
        roh = _collect(root, h, "roh")
        seiten = _collect(root, h, "seiten")
        out[str(h)] = dict(laeufe=len({a.split("/")[1] for _, a in roh if a.count("/") >= 2}),
                           seiten=sum(1 for _, a in seiten if a.startswith("seiten/archiv/")),
                           mb_roh=round(sum(f.stat().st_size for f, _ in roh) / 1e6, 1),
                           mb_seiten=round(sum(f.stat().st_size for f, _ in seiten) / 1e6, 1))
    return out


class Handler(SimpleHTTPRequestHandler):
    root = ROOT

    def do_GET(self):                                          # noqa: N802
        u = urlparse(self.path)
        p = u.path.rstrip("/")
        try:
            if p == "/download/info":
                body = json.dumps(_info(self.root)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if p == "/download":
                self._zip(parse_qs(u.query))
                return
        except (BrokenPipeError, ConnectionResetError):
            return
        super().do_GET()

    def _zip(self, q: dict) -> None:
        try:
            hours = min(max(float(q.get("stunden", ["24"])[0]), 0.0), MAX_H)
        except ValueError:
            hours = 24.0
        was = q.get("was", ["alles"])[0]
        if was not in ("alles", "roh", "seiten"):
            was = "alles"
        files = _collect(self.root, hours, was)
        stamp = time.strftime("%Y%m%d_%H%M")
        name = f"hobbyflood_{was}_{'alles' if hours == 0 else f'{hours:g}h'}_{stamp}.zip"
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # Zip direkt in die Verbindung (nicht durchsuchbarer Strom: zipfile schreibt dann Datenbeschreiber); Verbindung endet danach (HTTP/1.0)
        with zipfile.ZipFile(self.wfile, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f, arc in files:
                try:
                    z.write(f, arc)
                except FileNotFoundError:                      # während des Packens weggeräumt
                    pass


def main(argv: list[str]) -> None:
    port, root = 8080, ROOT
    args = list(argv)
    if "--root" in args:
        i = args.index("--root")
        root = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    if args:
        port = int(args[0])
    Handler.root = root
    srv = ThreadingHTTPServer(("", port), partial(Handler, directory=str(root / "gui")))
    print(f"Server auf Port {port}, Wurzel {root}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
