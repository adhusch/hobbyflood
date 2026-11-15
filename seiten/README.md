# Zusatzseiten für den Container

Hier liegen fertig gebaute Seiten, die der Container neben der Live-Vorhersage ausliefert:

- `eval.html`: Vorhersage-Prüfung an historischen Hochwassern (`python eval_view.py --out seiten/eval.html`)
- `routing_map.html`: Karte und Routing-Modell (`python routing_map.py --out seiten/routing_map.html`)

Beide brauchen zum Bauen die Pegeldaten der LfU-Pakete und enthalten die gemessenen Ganglinien der Hochwasser
(Quelle: Landesamt für Umwelt Rheinland-Pfalz). Sie sind deshalb nicht Teil des Repositorys (`.gitignore`).
Im Betrieb den Ordner als Volume einhängen (`./seiten:/app/seiten:ro`, siehe `docker-compose.yml`) und den Container neu starten.
