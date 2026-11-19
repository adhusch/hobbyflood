#!/bin/sh
# Live-Schleife im Container: Abruf + Rechnung alle INTERVAL_MIN Minuten, jede Ausgabe wird archiviert (gui/archiv/),
# die Seiten werden über einen einfachen Webserver auf PORT ausgeliefert.
# Die Quellen (hochwasser.rlp.de, DWD Open Data, Open-Meteo) sind frei zugänglich; Abrufe sparsam halten (Standard: stündlich).
set -u
INTERVAL_MIN="${INTERVAL_MIN:-60}"
# Feste Uhrzeiten statt Pause nach jedem Lauf (sonst wandert der Takt um die Rechenzeit): Läufe um OFFSET_MIN nach jedem Vielfachen von
# INTERVAL_MIN, Standard stündlich um hh:40: ICON-D2-RUC ist bei Open-Meteo rund 1 h 27 min nach dem Start abrufbar (Rechnung beim DWD,
# Bereitstellung, Umwandlung), um hh:40 MEZ also sicher der Lauf von hh-1:00 (um hh:07 war es der von hh-2:00); der Viertelstundenwert
# hh:30 der Pegel (kommt mit einigen Minuten Verzug) ist dann auch da. Ein Lauf dauert auf dem Raspberry Pi rund 7 min, die Seite steht
# also gegen hh:47. Start und Verfügbarkeit jedes ICON-Laufs stehen in data/live/laeufe.csv; danach den Takt nachjustieren.
OFFSET_MIN="${OFFSET_MIN:-40}"
PORT="${PORT:-8080}"
mkdir -p gui/archiv data/live
[ -f gui/index.html ] || printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=live.html"><a href="live.html">Live-Vorhersage</a> · <a href="archiv/">Archiv</a>\n' > gui/index.html
[ -f gui/live.html ] || printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><p>Der erste Lauf rechnet noch (etwa eine Minute). Die Seite lädt von selbst neu.</p><p>Eigene Modellrechnung, keine amtliche Warnung.</p>\n' > gui/live.html
# Fertige Zusatzseiten (Vorhersage-Prüfung eval.html, Karte routing_map.html): werden außerhalb des Containers gebaut, weil sie die Pegeldaten
# der LfU-Pakete brauchen (python eval_view.py, python routing_map.py). Liegen sie in /app/seiten (im Image oder als Volume), stellt der
# Container sie neben der Live-Seite bereit.
cp seiten/*.html gui/ 2>/dev/null || true
# Webserver mit Download (/download, siehe deploy/server.py); scheitert er, der einfache Server ohne Download
(python deploy/server.py "$PORT" || python -m http.server "$PORT" --directory gui) &
while true; do
  ts="$(date -u +%Y%m%dT%H%MZ)"
  if python live.py --out gui/live.tmp.html; then
    mv gui/live.tmp.html gui/live.html
    cp gui/live.html "gui/archiv/live_${ts}.html"
    python live_eval.py --archiv gui/archiv --out gui/auswertung.html || echo "[$ts] Auswertung fehlgeschlagen" >&2   # archivierte Ausgaben gegen die Messung
    echo "[$ts] Lauf fertig"
  else
    echo "[$ts] Lauf fehlgeschlagen, alte Seite bleibt stehen" >&2
  fi
  now=$(date +%s); per=$((INTERVAL_MIN * 60)); off=$((OFFSET_MIN * 60))
  next=$(( ((now - off) / per + 1) * per + off ))
  sleep "$((next - now))"
done
