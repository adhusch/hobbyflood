#!/bin/sh
# Live-Schleife im Container: Abruf + Rechnung alle INTERVAL_MIN Minuten, jede Ausgabe wird archiviert (gui/archiv/),
# die Seiten werden über einen einfachen Webserver auf PORT ausgeliefert.
# Die Quellen (hochwasser.rlp.de, DWD Open Data, Open-Meteo) sind frei zugänglich; Abrufe sparsam halten (Standard: stündlich).
set -u
INTERVAL_MIN="${INTERVAL_MIN:-60}"
PORT="${PORT:-8080}"
mkdir -p gui/archiv data/live
[ -f gui/index.html ] || printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=live.html"><a href="live.html">Live-Vorhersage</a> · <a href="archiv/">Archiv</a>\n' > gui/index.html
[ -f gui/live.html ] || printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><p>Der erste Lauf rechnet noch (etwa eine Minute). Die Seite lädt von selbst neu.</p><p>Eigene Modellrechnung, keine amtliche Warnung.</p>\n' > gui/live.html
python -m http.server "$PORT" --directory gui &
while true; do
  ts="$(date -u +%Y%m%dT%H%MZ)"
  if python live.py --out gui/live.tmp.html; then
    mv gui/live.tmp.html gui/live.html
    cp gui/live.html "gui/archiv/live_${ts}.html"
    echo "[$ts] Lauf fertig"
  else
    echo "[$ts] Lauf fehlgeschlagen, alte Seite bleibt stehen" >&2
  fi
  sleep "$((INTERVAL_MIN * 60))"
done
