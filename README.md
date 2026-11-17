# Hochwasser-Vorhersage Prüm, Enz und Nims (hobbyflood)

**Eigene Modellrechnung, keine amtliche Warnung.** Amtliche Hochwasserinformationen: Hochwasservorhersagezentrale Rheinland-Pfalz, https://www.hochwasser.rlp.de

Kettenvorhersage für acht Pegel (Prüm 2, Echtershausen, Wiersdorf, Prümzurlay, Sinspelt, Giesdorf, Seffern, Alsdorf-Oberecken) mit Stationsregen,
ICON-D2 und Unsicherheitsband; Warnung ab HW 10. Die Abschnitte unten sind das Arbeitsprotokoll in zeitlicher Reihenfolge.

## Repository und Live-Betrieb im Container

- **Im Repository:** der gesamte Code, die Tests, die angepassten Modelle (`data/chain_models.json`), die Unsicherheitstabellen
  (`data/uncert_tables.json`), Pegelkennwerte und Abflusskurven, die Stationsliste und die DWD-Stundenwerte der neun Stationen (`data/dwd_rr/`,
  Quelle: Deutscher Wetterdienst, Open Data) sowie die Einzugsgebiete (`data/radar/catchments.*`, aus dem Copernicus-Höhenmodell GLO-30).
  Das reicht für den Live-Betrieb.
- **Nicht im Repository:** die Abflussreihen des LfU Rheinland-Pfalz (`data/lfu_pkg/`, Wasserportal-Exporte, Ereignisdateien mit gemessenen
  Ganglinien), die Radar-Ausschnitte (`data/radar/crop/`, 55 MB, mit `radar_crop_browser.js` oder `radar_crop.py` aus RADKLIM neu zu erzeugen),
  die Höhenmodell-Kacheln, ICON-Archive und erzeugte Seiten. Für eine Neuanpassung müssen diese Daten wieder nach `data/` gelegt werden.
- **Live im Container:** `docker compose up -d` (`docker-compose.yml`, Image aus ghcr.io) oder selbst bauen: `docker build -t hobbyflood .` und `docker run -p 8080:8080 -v hobbyflood_live:/app/data/live -v hobbyflood_archiv:/app/gui/archiv hobbyflood`.
  Der Container holt stündlich (`INTERVAL_MIN`) Pegel (hochwasser.rlp.de), Stationsregen (DWD) und ICON-D2 (Open-Meteo), rechnet die Kette
  (`python live.py`) und liefert `live.html` auf Port 8080; jede Ausgabe wird unter `archiv/` aufgehoben (`deploy/run_live.sh`).
  Einmalig von Hand: `python live.py` (braucht Internet und `requirements.txt`).
- **Anpassung und Radar:** zusätzlich `requirements-train.txt`.
- **Immer eine Seite:** Bis zum ersten Lauf steht eine Platzhalterseite da; scheitert ein Lauf, bleibt die letzte Seite stehen und zeigt
  ab drei Stunden Alter den Hinweis „Stand veraltet“. Im Container lädt die Seite alle 15 Minuten neu.
- **Szenario** (auf der Live-Seite unter der Ganglinie): Regenmenge als Gebietsmittel, Beginn und Dauer vorgeben, statt oder zusätzlich zur
  Regenvorhersage; die Kette rechnet im Browser (`forecast_core.js`, gleiche Rechnung wie in Python, Abweichung unter 0,001 m³/s) für 48 h.
  Der Regen fällt gleichmäßig über Gebiet und Zeitraum; eine Linie ohne Band, keine Warnung.
- **Ausblick 7 Tage** (`live.outlook`): eine Linie je Pegel ohne Band. Regen je Station aus ICON-D2, nach dessen Ende aus ICON-EU (bis 120 h),
  danach aus ICON global (Open-Meteo); ICON-EU und global ohne Angleichfaktor. Für Vorläufe über 48 h gibt es keine Rückrechnung, die
  Warnung gilt weiter nur für 48 h.

# Arbeitsprotokoll: Abfluss-zu-Abfluss-Modelle und Pegel-GUI

Stand 12.11.2026 (zuletzt: Vorhersage in cm mit Unsicherheit, siehe letzter Abschnitt). Enthalten sind echte Pegeldaten in drei Formen:
- **5 Tage in 15 min** (20.–25.10.2026, JSON-Schnittstelle von hochwasser.rlp.de): ein einzelnes, noch ansteigendes
  Ereignis. Damit lassen sich Pipeline und GUI prüfen, aber keine Modelle trainieren.
- **Welle vom 24.–26.10.2026 in 15 min** (`data/wasserportal_event_2026-10.csv`, Wasserportal 90-Tage-Export, 24.10. 12:00 bis 26.10. 15:15 MEZ,
  206 Schritte, 8 Q-Reihen plus Stausee-Seestand): die erste vollständige Welle mit Scheitel und Rückgang an allen Pegeln (Echtershausen
  18,3 m³/s, Prümzurlay 17,4 m³/s, etwa 3 % von HQ). Übertragen über einen schmalen Kanal, etwa 0,15 % genau (`decode_event.py`).
- **3 Jahre Tagesmittel** (25.10.2023–24.10.2026, Wasserportal, ungeprüfte Rohdaten) für 7 der 8 LfU-Pegel mit Q:
  mehrere Hochwasser (größtes Tagesmittel Prümzurlay 104 m³/s am 03.01.2024), aber zu grob für Laufzeiten im Stundenbereich.
  Wiersdorf hat dort keine Tagesmittel. Das Hochwasser vom Juli 2021 (HQ 596 m³/s) liegt außerhalb.
Für Stundenmodelle über mehrere Ereignisse fehlen noch die geprüften 15-min-Reihen (Download-Assistent, siehe unten). Die Links der
E-Mail liefern seit dem 25.10. durchgehend 502 Bad Gateway (geprüft am 26.10.), sie gelten bis etwa 28.10.2026.

## Datenquelle (geprüft)
`https://hochwasser.rlp.de/api/v1/config` und `.../api/v1/measurement-site/<nummer>` liefern pro Pegel ein
5-Tage-Fenster für W und Q (`measurements`, 15 min, UTC) sowie `extremeevents` (historische Jahreshöchstwerte).
Die Konfiguration enthält Koordinaten (UTM 32), Einzugsgebiet und Pegelnull. Der Server setzt keinen CORS-Header:
Abruf geht nur serverseitig (Skript) oder von der Seite selbst, nicht aus einer lokalen HTML-Datei.
Für Kommunalpegel gibt es nur W. Der Stausee liefert nur W (Seestand, ca. 255,5 m) und keine Abgabe.

## Dateien
| Datei | Zweck |
|---|---|
| `pegel_netz.csv` | 32 Messstellen: Oberlieger/Zufluss, Koordinaten, EZG. Reihenfolge entlang Prüm, Enz, Nims per Nordwert geprüft |
| `data/real_5d.csv` | echte Q-Reihen der 8 LfU-Pegel plus Stausee-Seestand (15 min, 20.–25.10.2026, auf 0,01 gerundet) |
| `graph.py` | liest die CSV, nächste Abfluss-Oberlieger je LfU-Pegel (Stausee = Sonderfall) |
| `fetch_rlp.py` | `probe`, `snapshot`, `loop`: sammelt die 5-Tage-Fenster in `data/snapshots.csv` |
| `lags.py` | vorläufige Laufzeitschätzung vom 25.10. mit drei Verfahren auf dem 5-Tage-Fenster (noch steigender Ast), plus Plausibilität über Flusskilometer. Ersetzt durch `lags_event.py` |
| `ro.py` | Lag-Datensatz, Modelle (Persistenz, Skaliert+Lag, NNLS, Ridge, GBM, NNLS+GBM), Metriken |
| `synth.py` | synthetische Kaskade mit Laufzeit, Zwischenzufluss, Speicher, Starkregen im Testjahr |
| `run_baselines.py` | Vergleichstabellen (`--snapshots data/real_5d.csv` oder synthetisch) |
| `import_wasserportal.py` | liest Wasserportal-CSV (15 min oder Tagesmittel), rechnet MEZ nach UTC, schreibt langes bzw. breites Format |
| `inspect_package.py` | zeigt Aufbau und Kopfzeilen einer Datei aus dem Download-Assistenten (zip oder Text), bevor sie importiert wird |
| `lags_event.py` | Lag-Analyse für eine ganze Welle mit Scheitel und Rückgang: Scheitel- (auch Plateau-Mitte), Schwerpunkt-, Anstiegs- und Rückgangs-Lag je Anteil f mit Q des Oberliegers, Korrelation und Lag-Fit y = a·x(t−L)+c (gesamt, steigend, fallend). An synthetischen Fällen geprüft (Verschiebung, abflussabhängige Laufzeit, fehlender Scheitel). `python lags_event.py data/wasserportal_event_2026-10.csv` |
| `screenshot_lag.py` | liest aus einem Portal-Screenshot mit zwei Kurven (Wasserstand) Scheitelzeiten und Zeitversatz je Anteil des Anstiegs ab (Tagesstriche als Zeitachse, Genauigkeit etwa 0,25 h) |
| `event_checks.py` | Zusatzprüfungen zur Welle: Wasserportal-CSV gegen JSON-Schnittstelle, Wellengeschwindigkeit aus Flusskilometern, Zwei-Eingangs-Fit für Prümzurlay (Wiersdorf + Enz), Stausee-Bilanz, Lag-Fit getrennt für steigenden und fallenden Ast |
| `decode_event.py`, `data/wasserportal/_transfer90/` | dekodiert die übertragene Welle (8 Blöcke zu 500 Zeichen, Prüfsummen stimmten) nach `data/wasserportal_event_2026-10.csv` (UTC). Mit `import_wasserportal.py` lassen sich die exakten Originale einlesen |
| `confluence.py` | Enz-Zerlegung für Prümzurlay: Lag Sinspelt → Holsthum Schloss und Schloss → Prümzurlay aus Wasserständen (Fit y = a·x(t−L)+c über normierte Pegel, Fenster bis zum Rückstau am Schloss), Front der Stausee-Abgabe entlang Wiersdorf – Brecht – Bettingen – Holsthum Prüm – Prümzurlay, Summenmodell Q_PZ = a·Q_Wiersdorf(t−L1) + b·Q_Sinspelt(t−L2) + c mit festen und freien Lags. `python confluence.py` |
| `decode_w.py`, `data/wasserportal/_transfer_w/` | dekodiert die per Browser übertragenen Wasserstände (cm, 15 min, 10 Pegel inkl. kommunale Pegel aus der JSON-Schnittstelle von hochwasser.rlp.de, 10 Blöcke mit Prüfsumme) nach `data/json_w_event_2026-10.csv` (UTC) |
| `build_lag_map.py`, `lag_map_template.html` | rechnen die Lags der Welle (`lags_event.py`) und zum Vergleich die vorläufigen Verfahren A/B/C aus `lags.py` (Stand 25.10.), setzen sie mit der Tagesmittel-Gegenprobe und den Ganglinien jeder Strecke auf eine Karte: `gui/lag_map.html` (einzelne Datei, Hintergrund optional über OpenStreetMap-Kacheln, ungetestet gegen das echte Kachelnetz) |
| `run_daily.py` | Baselines auf `data/wasserportal/daily_q.csv` (Test = letztes Jahr) |
| `data/wasserportal/daily_q.csv` | 3 Jahre Tagesmittel Q, 7 Pegel. Über einen schmalen Kanal übertragen und auf etwa 0,15 % genau (`decode_daily.py`, `_transfer/`); exakte Originale mit `import_wasserportal.py --daily` einlesen |
| `import_pkg.py` | liest ein LfU-Paket `pegel15PKGxx.zip` (alle Pegel RLP, Ordner `q/` und `w/`, 15 min, MEZ) und schreibt die acht Prüm-Pegel nach `data/lfu_pkg/<PKG>_Q.csv.gz` und `_W.csv.gz`. `python -I import_pkg.py <zip>` |
| `pkgdata.py` | `load_all("Q"\|"W")` setzt PKG01 bis PKG04 zusammen (01.11.1989 bis 01.01.2023, 1.162.945 Zeilen, lückenlos). `python pkgdata.py` prüft Raster und fehlende Werte |
| `lags_history.py` | Laufzeitmaße je Ereignis (Scheitel, Plateau-Mitte, Anstiegsanteile, Fit, Korrelation der 1-h-Änderungen) über 98-%-Quantil-Ereignisse, Standard-Eingabe PKG04 |
| `lags_fir.py` | nichtnegative Übertragungsfunktion (FIR) je Abflussklasse klein/groß/extrem, mittlere Verzögerung mit Bootstrap; Ausgabe `data/lags_fir_classes.csv` |
| `floods.py`, `floods_plot.py`, `floods_html.py`, `floods_template.html` | große Hochwasser 1989 bis 2022 (Prümzurlay über 400 cm, Ereignisse unter 48 h Abstand zusammengefasst): Scheitel aller Pegel, Laufzeiten der Prüm-/Enz-Strecken, 72-h-Grafiken (36 h vor bis 36 h nach dem Scheitel) als `figures/hochwasser_<datum>.png` und die eigenständige Übersicht `gui/hochwasser_400cm.html`. `python floods.py` erzeugt alles; Ausgaben `data/floods_400cm_events.csv`, `_gauges.csv`, `_pairs.csv`, `_joint.csv` |
| `routing_core.c`, `routing.py` | Routing-Modell: Kaskade nichtlinearer Linearspeicher (Kalinin-Miljukov) mit abflussabhängiger Wellengeschwindigkeit c(Q), Verstärkung je Zufluss und Zwischenzufluss-Konstante; C-Kern (ctypes, wird beim ersten Import mit `cc -O2 -shared -fPIC` gebaut) mit numpy-Rückfall, Ereignissuche, Anpassung (Least Squares, grob nach fein), Vergleichsmodell mit festem Lag |
| `routing_fit.py` | Anpassung je Unterlieger (Echtershausen, Wiersdorf mit Stausee, Prümzurlay mit Wiersdorf + Sinspelt, Seffern, Alsdorf-Oberecken), Leave-one-flood-out über die neun Hochwasser, Vorhersagebetrieb mit Fehlerkorrektur, Bootstrap für c(Q). `python routing_fit.py --jobs 2 --boot 40` (etwa 7 min), `--quick` zum Testen. Ausgaben `data/routing_params.json`, `routing_skill_floods.csv`, `routing_forecast_skill.csv`, `routing_sim_floods.json` |
| `routing_map.py`, `routing_map_template.html` | bauen `gui/routing_map.html`: Karte mit Abfluss (Fläche der Kreise), Wellengeschwindigkeit je Strecke (Farbe, laufende Strichel), Modellwert als gestrichelter Ring, Zeitregler und Hochwasserauswahl; Tabs Strecke (Auswahl der Strecke; c(Q) und Laufzeit mit Bootstrap-Band), Ganglinie (Auswahl des Pegels; Messung, Modell, Varianten), Vorhersage (Auswahl eines der acht Abflusspegel, auch per Klick in die Karte, die Karte hebt die Kette oberhalb hervor; Kettenvorhersage mit Schieber für die Ausgabezeit im 15-min-Takt, im Browser gerechnet; Güte über 15 min bis 48 h Vorlauf; Scheiteltreffer), Überblick. Einzelne Datei, Hintergrundkarte optional |
| `test_routing.py` | Kern gegen numpy, Laufzeit = L / c, Volumenerhaltung, Wiederfinden bekannter c(Q) an synthetischen Daten |
| `lateral.py`, `rain.py`, `test_rain.py` | Regen-Abfluss-Anteil für die Zwischengebiete (Abflussbeiwert aus Vorregenindex, optional mit Intensität und Ereignisfeuchte, zwei Linearspeicher, exakt gelöst; wird in `routing.Model(lat=[...])` gemeinsam mit dem Routing angepasst), Einlesen von DWD-Stundenwerten (`produkt_rr_stunde_*.txt`), Gebietsniederschlag mit Abstandsgewichten, Vorregenindex. Seit 13.11.2026 mit Sättigungsüberschuss (SCS, schneller Weg; `sat`, mit festen gemeinsamen Parametern `satx`) und nichtlinearen Speichern (`nlr`/`nlrx`, massentreu). Getestet an synthetischen Daten (`python test_rain.py`: Gewichte und Lücken, Massenerhaltung, Wiederfinden der Parameter aus weit entfernten Startwerten, Gebietsfeuchte, Sättigungsüberschuss) |
| `rain_fit.py`, `data/dwd_rr/` | Anpassung und Prüfung des Routings mit Regen, auch mit intensitätsabhängigem Abflussbeiwert (siehe Abschnitt Regen): liest die neun DWD-Stationen (`data/dwd_rr/*.zip`, `stationen.csv`), Gebietsniederschlag und Vorregenindex je Zwischengebiet, blockweise Kreuzvalidierung 2004 bis 2022, Hochwasser 2011/2018/2021, Vorhersagebetrieb. `python rain_fit.py {shift\|info\|run} [--jobs 2]` |
| `chain_fc.py` | Kettenvorhersage über alle acht Pegel: Oberläufe aus Regen (oder Persistenz), Unterlieger aus der Vorhersage ihrer Oberlieger, Fehlerkorrektur je Pegel aus den Trainingsereignissen, Umschalten zwischen Winter- und Starkregenmodellen; Ausgabe alle 15 min, Vorlauf 15 min bis 48 h, blockweise kreuzvalidiert, Sättigungsüberschuss und nichtlineare Speicher gemeinsam für alle Gebiete (ohne den Prüfblock), Wiersdorf mit absoluten Fehlern. `python chain_fc.py run --jobs 2` (etwa 19 min). Ausgaben `data/chain_skill.csv`, `chain_peak.csv`, `chain_models.json`, `chain_floods.json` |
| `forecast_core.js`, `test_forecast_js.py` | dieselbe Kettenvorhersage in JavaScript für die Karte (wird in `gui/routing_map.html` eingebettet, mit Umschalten Winter / Starkregen); `python test_forecast_js.py` vergleicht sie mit `chain_fc.py` (492 Fälle, alle Pegel der Kette, braucht node) |
| `rating.py`, `data/pegelkennwerte.json`, `data/rating_curves.json` | Kennwerte der HVZ Rheinland-Pfalz je Pegel (Meldehöhe, HW/HQ 2 bis 100, MW, MNW; aus `hochwasser.rlp.de/api/v1/config`, 12.11.2026) und die heute gültige Wasserstand-Abfluss-Beziehung je Pegel aus jüngsten Wertepaaren ohne Sprung, der Welle Oktober 2026 und den HVZ-Paaren (HQx, HWx). `python rating.py` |
| `extreme_fit.py` | Prüfgerüst für Extremhochwasser mit perfektem Regen: Modellvarianten je Block anpassen und an den HQ10-Hochwassern, allen Hochwassern und Regenlagen ohne Hochwasser messen (`base`, `try`, `regional`, `refit`, `compare`). Ergebnisse `data/extreme/` |
| `uncert.py` | Unsicherheit der Kettenvorhersage: stündliches Vorhersagearchiv (`archive`, etwa 15 min, Ablage im Zwischenordner), Fehlertabellen je Pegel, Zustand, Abflussstufe und Vorlauf, Starkregenlage (eigene Tabelle aus allen Pegeln), verschachtelte Prüfung von Band, Überschreitungswahrscheinlichkeit, Vorwarnzeit und Scheitelspanne (`tables`, etwa 5 min). Ausgaben `data/uncert_tables.json`, `uncert_band.csv`, `uncert_exceed.csv`, `uncert_warn.csv`, `uncert_peak.csv`, dazu für die Warnschwelle (HQ10) jedes Hochwasser einzeln `uncert_warn_events.csv` und alle Warnphasen `uncert_warn_phases.csv` (`tables` jetzt etwa 15 min) |
| `live.py`, `live_template.html` | Live-Vorhersage mit Wettervorhersage: `python live.py` holt die aktuellen Pegel (HVZ), den Stationsregen (DWD, recent und now) und ICON-D2 / ICON-D2-EPS (Open-Meteo) nach `data/live/raw_<Zeit>/` und baut `gui/live.html` (eine Datei, ohne Internet lesbar); `python live.py build [ordner]` baut nur, aus vorhandenen Rohdaten. Braucht Python 3 mit numpy, pandas, scipy und Internet (auf dem eigenen Rechner, nicht in der Sandbox) |
| `verif_rain.py`, `data/verif/` | ICON-D2 gegen die DWD-Stationen 2022–2026 (Historical Forecast API: Vorlauf 0–3 h; Previous Runs API: 24–27 h), Gebietsregen wie im Modell; Faktoren Station / ICON je Station (`rain_factors.json`), Kennzahlen (`rain_verif.csv`) |
| `hindcast.py`, `data/hindcast/` | Nachrechnung der Welle Oktober 2026 mit den damals verfügbaren ICON-D2-Läufen (Single Runs API), Abfluss aus dem Wasserportal (90 Tage, 15 min); Ausgaben `hindcast_2026-10.csv`, `gui/hindcast_2026-10.png` |
| `test_uncert_js.py` | vergleicht Band, Wahrscheinlichkeiten, Scheitelspanne und W-Q-Umrechnung im Browser (`UQ` in `forecast_core.js`) mit Python (675 Fälle, braucht node) |
| `export_gui_data.py` | baut `gui/data.js`; `--standalone` erzeugt eine einzelne HTML-Datei; `--synthetic` für die Demo |
| `gui/index.html` | GUI in reinem HTML/CSS/JS, ohne Bibliotheken. Schalter oben wechselt zwischen 15-min-Fenster und Tagesmittel |

## Historie aufbauen
1. `pip install -r requirements.txt`
2. `python fetch_rlp.py loop` (oder `snapshot` per Cron/Aufgabenplanung mindestens alle 4 Tage) füllt `data/snapshots.csv`
   lückenlos weiter. Pro Pegel gibt die API immer nur die letzten 5 Tage.
3. Längere Reihen bietet das Wasserportal:
   - `geodaten-wasser.rlp-umwelt.de/wasserstand/<nummer>/download` (Nummer mit zwei angehängten Nullen, z. B. `2628005000`):
     90 Tage 15-min-Werte, 3 Jahre Tagesmittel (Q und W), Hauptwerte, als CSV oder Excel, ungeprüfte Rohdaten. Hinter den Buttons
     steht `GET /api/export/messstellen_wasserstand_<abfluss|messwerte>[_mittel].csv?w=messstellennummer=<nummer>00` auf derselben Seite
     (`abfluss` = Q, `messwerte` = W). Die 90 Tage vom 26.07. bis 25.10.2026 waren ein trockener Sommer (Prümzurlay höchstens
     7,8 m³/s, nur das jetzige Ereignis), enthalten also keine Hochwasserdynamik.
   - **Geprüfte 15-min-Werte** im Download-Assistenten `aksam-web.rlp-umwelt.de/aksam` (Auswahl „15 Minutenwerte der Wasserstände
     und Abflüsse (geprüfte Daten)“): vier Pakete mit allen Landespegeln, 01.11.1989–31.10.1999, 01.11.1999–31.10.2009,
     01.11.2009–31.10.2019 und 01.11.2019–31.12.2022. Paket 4 enthält das Hochwasser vom Juli 2021. Der Assistent verschickt
     je Paket einen Download-Link per E-Mail (E-Mail-Adresse eintragen, Nutzungsbedingung CC BY 4.0 akzeptieren, Quellenangabe
     „LfU (2026): Daten des Landesamtes für Umwelt Rheinland-Pfalz“), deshalb nicht automatisiert abgerufen. Dateiformat und Größe
     sind noch unbekannt: `python inspect_package.py <datei>` zeigt den Aufbau. Neuere Zeiträume (2023 bis heute) gibt es nur als
     Tagesmittel.
   - Zeitbasis der CSV: feste MEZ (UTC+1) ohne Sommerzeit, an Prümzurlay gegen die JSON-Schnittstelle geprüft (siehe
     `import_wasserportal.py`). Das Niveau weicht bei Niedrigwasser von der JSON-Schnittstelle ab (0,65 gegen 0,82 m³/s), deshalb
     nicht in `snapshots.csv` mischen. Im Überlappungszeitraum 24.10. 11:00 bis 25.10. 10:30 UTC stimmen die Scheitel an Prüm 2,
     Echtershausen, Wiersdorf, Prümzurlay, Sinspelt und Giesdorf auf 0,6 % überein; Seffern und Alsdorf-Oberecken liegen in der CSV
     4 bis 5 % niedriger, Sinspelt bei kleinem Abfluss 17 % höher (`python event_checks.py`).
   - Import: `python import_wasserportal.py <dateien> [--daily]`.
4. `python run_baselines.py --snapshots data/snapshots.csv`, danach `python export_gui_data.py --standalone`
   (die GUI liest `data/real_5d.csv`, bei Bedarf `--real data/snapshots.csv` angeben).
5. GUI: `gui/index.html` öffnen. Weitere Snapshot-CSV lassen sich über „Choose File“ laden oder ins Fenster ziehen.

Nutzungsbedingungen der Daten sind nicht geprüft. Abrufe sparsam halten.

## Bekannte Lücken
- Offizielle Oberlieger-Angaben im Wasserportal nennen für Prümzurlay nur Wiersdorf, Echtershausen und Prüm 2, nicht die Enz.
  Die Enz (Sinspelt) ist hier aus der Geografie ergänzt.
- Laufzeiten (Welle vom 24.–26.10.2026, `lags_event.py`): Die Lags sind Reaktionsverzögerungen zwischen zwei Ganglinien einschließlich
  Zwischenzufluss, keine reine Wellenlaufzeit. Die Scheitel sind flach (Plateaus bis 3 h), die Scheitel-Lags deshalb etwa ±1 h unsicher.
  Scheitel-Lag (argmax / Plateau-Mitte): Prüm 2 → Echtershausen 7,6 / 6,9 h (41 km, 1,5 bis 1,7 m/s), Wiersdorf → Prümzurlay
  4,1 / 5,1 h (24 km, 1,3 bis 1,6 m/s; Fit gemeinsam mit der Enz 5,5 h bei NSE 0,995), Giesdorf → Seffern 3,75 / 5,0 h (18 km,
  1,0 bis 1,4 m/s), Seffern → Alsdorf 3,25 / 1,9 h (29 km, 2,5 bis 4,3 m/s, zu schnell, Alsdorf hat fast doppelte Fläche).
  Der Stausee Bitburg dämpft den Scheitel von 18,3 (Echtershausen) auf 12,4 m³/s (Wiersdorf) und verschiebt ihn um 4,5 h. Beim großen
  Hochwasser 2021 dämpfte er laut Portal kaum (445 → 436 m³/s). Die früheren 5-Tage-Werte (`lags.py`, 3 bis 10 m/s) waren
  Zwischengebiets-Reaktion auf dem Anstieg. Die Abhängigkeit vom Abfluss ist nicht belegt: Anstiegs- und Rückgangs-Lags
  unterscheiden sich teils stark (Seffern → Alsdorf Fit 1,5 h steigend, 5,0 h fallend, Prüm 2 → Echtershausen 4,5 und 9 h), das ist ein
  Speicher- und Rückgangseffekt und lässt sich in einer Welle von 3 % des HQ nicht von einem Abflusseffekt trennen. Referenz aus einem Portal-Screenshot (Juni 2018, `screenshot_lag.py`): Wiersdorf → Prümzurlay Scheitel-Lag 7,1 h bei 209 m³/s (Prümzurlay), also länger als die 4,1 bis 5,5 h der kleinen Welle. Das spricht gegen eine kürzere Laufzeit bei großem Abfluss und für Rückhalt in der Aue; ein Datenpunkt, keine Statistik. Dafür braucht es
  Hochwasser bis in den Bereich von 100 m³/s (geprüfte 15-min-Pakete des LfU, MQ Prümzurlay 7,8 m³/s, HQ 596 m³/s am 15.07.2021).
- Enz (`confluence.py`): Die 15 h Scheitel-Lag Sinspelt → Prümzurlay aus der ersten Auswertung waren ein Artefakt, denn der Scheitel in Prümzurlay
  ist der Scheitel der Prüm, in dem der kleinere Enz-Scheitel aufgeht. Zerlegt: Sinspelt → Holsthum Schloss 1,75 h (Fit auf Wasserständen,
  NSE 0,979, Lags mit NSE bis 0,005 unter dem besten: 1,0 bis 2,25 h; mit dem frühen Anstieg vom 24.10. 0 bis 0,75 h), Schloss → Prümzurlay 1,0 h
  (NSE 0,995), zusammen etwa 2,75 h. Prüm Wiersdorf → Prümzurlay: Front der Stausee-Abgabe am 25.10. in 5,25 h über 24,3 km (1,28 m/s).
  Summenmodell mit festen Lags 5,25 h und 2,75 h: NSE 0,994 (frei 5,5 h und 2,0 h: 0,995, der Enz-Lag ist kaum festgelegt). Enz-Anteil an Prümzurlay
  2,7 von 17,4 m³/s zur Zeit des Scheitels, Enz-Scheitel selbst (4,6 m³/s) am 25.10. gegen 14 Uhr, 12 h früher. Grenzen: für Enzen und Holsthum
  Schloss gibt es nur Wasserstände, das Schloss steht ab 25.10. 12:30 MEZ im Rückstau der Prüm, die Flusskilometer der Zwischenpegel sind aus
  den Koordinaten geschätzt.
- Die synthetischen Tests nutzen ein Muskingum mit konstantem K. Lineare Modelle schneiden dort besser ab als in der Realität.
- Erste Ergebnisse auf den echten Tagesmitteln (`python run_daily.py`, Training 2 Jahre, Test 25.10.2025–24.10.2026, nur 1–2
  Spitzen im Test): an Echtershausen←Prüm 2, Seffern←Giesdorf und Prümzurlay←Echtershausen+Sinspelt schlagen die linearen
  Modelle Persistenz um etwa 0,1 bis 0,5 Skill, an Alsdorf←Seffern liegen sie nur gleichauf. GBM ist an den drei kleinen Strecken
  schlechter als Persistenz und nur bei Prümzurlay etwa gleichauf mit den linearen Modellen. Zu wenig Ereignisse, um mehr daraus
  abzuleiten.
- Verhältnis Q(Prümzurlay)/Q(Echtershausen) im Tagesmittel sinkt von median 2,9 bei Niedrigwasser auf 1,6 bei über 50 m³/s
  (Flächenverhältnis 1,77): Zwischenzufluss und Speicherabgabe sind abflussabhängig, ein einfacher Flächenfaktor reicht nicht.
- Die Tagesmittel-CSV im Projekt ist kodiert übertragen (Genauigkeit etwa 0,15 %), siehe Dateitabelle.

## Große Hochwasser 1989 bis 2022 (LfU-Pakete)
Quelle: vier Pakete `pegel15PKG01` bis `PKG04` vom Wasserportal (15 min, Q in m³/s, W in cm, Zeit wie im Paket, angenommen MEZ ohne Sommerzeit).
Rohdaten liegen nicht im Projekt, nur die acht Prüm-Pegel als `data/lfu_pkg/*.csv.gz`.

| Scheitel Prümzurlay | Q (m³/s) | W (cm) | über 400 cm (h) |
|---|---|---|---|
| 22.12.1991 11:15 | 184 | 426 | 8,5 |
| 12.01.1993 18:45 | 192 | 434 | 11,0 |
| 21.12.1993 06:45 | 210 | 453 | 19,8 |
| 23.01.1995 10:15 | 184 | 426 | 12,3 |
| 26.01.1995 07:30 | 170 | 410 | 7,0 |
| 03.01.2003 06:00 | 252 | 492 | 24,8 |
| 08.01.2011 01:15 | 165 | 404 | 9,5 |
| 01.06.2018 19:30 | 209 | 452 | 4,5 |
| 15.07.2021 06:30 | 596 | 698 | 23,0 |

Hinweise:
- **Wiersdorf ist bei vier älteren Hochwassern bei 249 bis 250 cm (137,1 m³/s) abgeschnitten** (01/1993, 12/1993, 01/1995, 01/2003). Scheitel und Lags mit Wiersdorf sind dort nicht belastbar (in Tabellen und Grafiken gekennzeichnet).
- Scheitel je Pegel: Maximum im Fenster 30 h vor bis 6 h nach dem Scheitel in Prümzurlay (`floods.peak_time`), in Grafik, CSV und HTML gleich.
- Laufzeiten sind Reaktionsverzögerungen einschließlich Zwischenzufluss. Zwischen Echtershausen und Wiersdorf liegt der Stausee Bitburg; dort sind Lags Speicherwirkung.
- Befund zur Frage "schnellere Welle bei großem Hochwasser": keine eindeutige Abhängigkeit. Das Hochwasser 2021 hat Echtershausen → Wiersdorf in 0,4 h durchlaufen (Stausee durchlässig, Verhältnis 1,02), Wiersdorf → Prümzurlay in etwa 2 h; 2018 waren es dort rund 6 h, 1991 und 2011 zwischen 2 und 4 h. Die Streuung zwischen den Ereignissen (Regenverteilung, Zwischenzufluss, Stausee) ist größer als ein möglicher Abflusseffekt. Das gemeinsame Fitten mit Wiersdorf und Sinspelt je Ereignis ist nicht identifizierbar.

## Routing-Modell (abflussabhängige Wellengeschwindigkeit)
Aufbau: je Unterlieger `Q_dn(t) = c0 + Σ a_k · Kaskade_k[Q_up,k](t)`. Eine Kaskade ist eine Kette von n Linearspeichern mit `S = K · O`, `K = dx / (3,6 · c(Q))`, c nach Abfluss als Linienzug in log-log über drei Stützstellen (95-%-Quantil, 99,9-%-Quantil, 0,7 · Maximum des Oberliegers; Grenzen 0,15 bis 4 m/s, am Stausee 12 m/s). Die Laufzeit ist L / c(Q), die Dämpfung steuert n. Der Zufluss wird linear zwischen den 15-min-Werten angenommen, die Speichergleichung ist exakt gelöst. `a_k` fängt Zwischengebiet und Flächenverhältnis ab, `c0` den Basisabfluss.

Anpassung: Ereignisse sind Scheitel des Unterliegers über dem 95-%-Quantil (Fenster 60 h vor bis 24 h nach, 12 h Einschwingen ausgenommen), Residuen je Ereignis durch den Scheitel geteilt, Glättung der c(Q)-Stützstellen (zweite und erste Differenz). Wiersdorf ist bei vier älteren Hochwassern abgeschnitten (W 249 bis 250 cm); diese Ereignisse bleiben für Strecken mit Wiersdorf draußen, Prümzurlay rechnet dort Wiersdorf aus Echtershausen (Kette).

Prüfung: Leave-one-flood-out über die neun Hochwasser (Training ohne Ereignisse im Abstand von ±3,5 Tagen), Test 48 h vor bis 36 h nach dem Scheitel Prümzurlay, Kennzahlen im Fenster ±36 h. Vergleich mit zwei Varianten: konstantes c (gleiche Struktur, c fest) und fester Lag mit Verstärkung (Regression). Median NSE im Test (jeweils: c(Q) / konstantes c / fester Lag; Anzahl Hochwasser mit c(Q) besser als konstantes c):

| Unterlieger | c(Q) | konst. c | fester Lag | c(Q) vorn |
|---|---|---|---|---|
| Echtershausen (Prüm 2) | 0,70 | 0,75 | 0,71 | 1 von 9 |
| Wiersdorf (Stausee) | 0,80 | 0,70 | 0,63 | 6 von 6 |
| Prümzurlay (Wiersdorf + Sinspelt) | 0,89 | 0,87 | 0,88 | 4 von 9 |
| Seffern (Giesdorf) | 0,66 | 0,65 | 0,54 | 8 von 9 |
| Alsdorf-Oberecken (Seffern) | 0,76 | 0,80 | 0,74 | 2 von 9 |

(Wiersdorf: nur sechs Hochwasser; bei drei ist Wiersdorf abgeschnitten und wird nicht bewertet.) Werte der Tabelle: Median über die Hochwasser, `data/routing_skill_floods.csv`.

Befunde:
- **Stausee Bitburg** ist die einzige Strecke mit klarer Abflussabhängigkeit: Laufzeit Echtershausen → Wiersdorf etwa 6 h bei 20 m³/s, 2,8 h bei 72 m³/s, 2,3 h bei 311 m³/s (Bootstrap-Bänder getrennt). c(Q) schlägt dort konstantes c und festen Lag in allen sechs prüfbaren Hochwassern. Das ist Speicherwirkung (bei hohem Seestand läuft der Zufluss fast durch), keine Wellengeschwindigkeit im Fluss.
- **Flussstrecken**: kein Gewinn durch c(Q) in der Simulation. Wiersdorf → Prümzurlay und Seffern → Alsdorf werden bei großem Abfluss eher langsamer (3,4 → 4,4 h bzw. 3,4 → 6,0 h), vereinbar mit Vorlandspeicherung; die Vermutung "schnellere Welle bei Hochwasser" wird für die Flussstrecken nicht gestützt. Hochwasser 2021 (596 m³/s, Training bis 252): c(Q) extrapoliert schlechter als konstantes c (NSE Prümzurlay 0,94 gegen 0,98, Scheitel -13 % gegen -9 %).
- **Vorhersage** (Ausgabezeiten 30 h vor bis zum Scheitel, Vorlauf bis 12 h, Oberlieger gehalten, Fehlerkorrektur 6 h): gegen Persistenz positiv, bei Prümzurlay Güte 0,31 (1 h), 0,49 (3 h), 0,48 (6 h) mit c(Q); fester Lag ist bei 1 h negativ (-0,49). Bei 12 h ist Prümzurlay Persistenz-gleich, weil Wiersdorf und Sinspelt dann nicht mehr ausreichen.
- **Lateral dominierte Strecken** (Prüm 2 → Echtershausen: Fläche 16 %, a = 4,9) lassen sich nicht aus dem Oberlieger vorhersagen (Hochwasser 2018: NSE ≈ 0). Dafür braucht es Niederschlag.
- **Sinspelt → Prümzurlay**: c liegt an der Obergrenze (4 m/s) und ist nicht bestimmt; die Strecke wirkt als Zwischenzufluss-Anzeiger.

Grenzen: Flusslinien der Karte sind schematisch (Verbindungen durch die Pegelpositionen, keine Gewässergeometrie); Zeitbasis MEZ ohne Sommerzeit angenommen; Testwerte sind kleine Stichproben (neun Hochwasser). Nächster Schritt: Niederschlag (RADOLAN) als Eingang, Nims als Regenanzeiger, Zustandsnachführung.

## Regen im Routing-Modell (echte DWD-Stundenwerte, 9 Stationen)
Die Zwischengebiete, die das reine Routing nicht erklärt, bekommen einen Regen-Abfluss-Anteil. Fläche je Zwischengebiet aus `pegel_netz.csv`: Echtershausen 272,9 km² (326,34 − 53,41), Prümzurlay 134,7 km² (576,38 − 340,2 − 101,52), Seffern 119,2 km², Alsdorf-Oberecken 128,1 km². Wiersdorf (13,9 km², nur Seeumfeld) bekommt keinen Regenanteil.

Aufbau je Zwischengebiet (`lateral.py`): Abflussbeiwert `phi(t) = phi_max · (1 − exp(−API / S))` mit Vorregenindex API (Abklingzeit fest 7 Tage), wirksamer Regen `phi · P`, dahinter zwei Linearspeicher in Reihe (K1 schnell, K2 langsam), `Q_lat = A / 3,6 · O2`. Vier Parameter je Gebiet (phi_max 0 bis 1, S 5 bis 300 mm, K1 0,5 bis 24 h, K2 2 bis 96 h), gemeinsam mit den Routing-Parametern angepasst. Gebietsniederschlag: Abstandsgewichte (Potenz 2) der Stationen zum Mittelpunkt der Pegel, die das Gebiet umschließen, Lücken je Zeitpunkt renormiert.

### Regendaten (heruntergeladen, `data/dwd_rr/`)
Neun DWD-Stationen, Stundenwerte RR historisch (`stundenwerte_RR_<ID>_<von>_<bis>_hist.zip`, zusammen 4,2 MB, Koordinaten und Zeiträume in `data/dwd_rr/stationen.csv`): Prüm-Watzerath (04017), Lauperath-Scheidchen (00200), Schneifelforsthaus (04508), Winterspelt (05616), Olsdorf (01964), Bitburg (00523, bis 2018), Wiersdorf/Stausee Bitburg (14037, ab 2016), Olmscheid (15155, ab 2012), Weißenseifen (02170).

Abdeckung: Die DWD-Stundenreihen beginnen nominell am 01.09.1995, die Stationen der Region aber erst 2002 bis 2006. Deshalb gibt es Regen nur für **2004 bis 2022**; von den neun großen Hochwassern haben **2011, 2018 und 2021** Regen, die fünf Hochwasser 1991 bis 1995 und das vom Januar 2003 (Lücken) nicht. Für diese Hochwasser zeigt die Karte nur das Routing und sagt das. Zeitbasis: Pegel in fester MEZ (UTC+1), DWD in UTC, Stundenwert = Intervallende, also Versatz +1 h. Ein Test mit 0, +1 und +2 h Versatz ergab fast gleiche Kosten, der Versatz ist aus den Daten nicht bestimmbar; +1 h ist die Umrechnung UTC nach MEZ.

### Prüfung
Blockweise Kreuzvalidierung nach Jahren (2004 bis 2007, 2008 bis 2011, 2012 bis 2015, 2016 bis 2018, 2019 bis 2022). Das Training lässt den Testblock und Ereignisse im Abstand von ±3,5 Tagen aus. Gepoolter NSE (Fehler je Ereignis durch den Scheitel geteilt), gleiche Ereignisse mit und ohne Regenanteil (`data/rain_cv_summary.csv`):

| Unterlieger | Ereignisse | NSE ohne Regen | NSE mit Regen | Güte 3 h voraus ohne / mit | phi_max |
|---|---|---|---|---|---|
| Echtershausen | 126 | 0,698 | 0,770 | 0,20 / 0,39 | 0,27 |
| Prümzurlay | 113 | 0,910 | 0,921 | 0,58 / 0,65 | 0,13 |
| Seffern | 163 | 0,647 | 0,644 | 0,44 / 0,43 | 0,31 (S an der Obergrenze) |
| Alsdorf-Oberecken | 144 | 0,586 | 0,605 | 0,02 / 0,09 | 0,15 |

Vorhersagebetrieb (`data/rain_forecast_skill.csv`): Ausgabezeiten 24 h vor bis zum Scheitel in 2-h-Schritten, Vorlauf bis 12 h, Oberlieger gehalten, Regen bis zur Ausgabezeit beobachtet und danach 0 (keine Regenvorhersage), Vorregenindex klingt ab, Fehlerkorrektur 6 h. Güte gegen Persistenz: Echtershausen 0,39 statt 0,20 bei 3 h und 0,47 statt 0,33 bei 6 h; Prümzurlay 0,65 statt 0,58 bei 3 h; Seffern unverändert; Alsdorf-Oberecken bleibt schwach (0,09 bei 3 h).

### Befunde und Grenzen
- Regen hilft deutlich bei **Echtershausen**, der Strecke, die vom Zwischengebiet (273 km² gegen 53 km² Oberlieger) lebt, und etwas bei Prümzurlay und Alsdorf-Oberecken. Bei **Seffern** bringt er nichts; die Parameter laufen an die Grenzen (S 300 mm, K1 0,5 h), das Zwischengebiet ist dort zu klein und der Regen der nächsten Station zu wenig repräsentativ.
- Die **ganz großen Hochwasser** bleiben unterschätzt. Scheitel mit Regen gegen Messung bei Echtershausen: 2018 −74 % (ohne Regen −85 %), 2021 −55 %; bei Seffern 2018 −50 %, 2021 −44 %; Prümzurlay liegt mit +13 % (2018) und −2 % (2021) nah dran. Ein Abflussbeiwert, der mit der Regenintensität oder der Regenmenge im Ereignis wächst, behebt das nicht (nächster Abschnitt). Das Hochwasser 2011 wird bei Echtershausen mit Regen leicht schlechter (Scheitel −7 % gegen −3 %).
- Nur drei der neun großen Hochwasser haben Regen. Die Aussage zu den Extremen stützt sich damit auf zwei Ereignisse (2018, 2021).
- Gebietsniederschlag aus nur neun Punktmessungen: Die Gewichte konzentrieren sich oft auf eine Station (Prümzurlay zu 90 % Olsdorf), bei Alsdorf-Oberecken wechselt die Hauptstation 2016 bis 2018 von Bitburg zu Wiersdorf. Flächige Daten (RADOLAN) wären der nächste Schritt.
- Es gibt keine Regenvorhersage im Modell: Die Vorhersagegüte mit Regen ist die mit beobachtetem Regen bis zur Ausgabezeit, nicht die eines echten Vorhersagebetriebs mit Niederschlagsprognose.

### Versuch: intensitätsabhängiger Abflussbeiwert (kein Gewinn)
Erweiterung in `lateral.py` (`Lateral.terms`, Standard leer = Grundmodell): `"int"`: `phi = phi_a + (1 − phi_a) · rho · P² / (P² + Pc²)` mit der Regenintensität P (mm/h), dazu rho und Pc als Parameter; `"acc"`: der Vorfeuchte-Index bekommt einen Anteil w · API24 (Vorregenindex mit 24 h Abklingzeit, wächst im Ereignis mit der Regenmenge, ähnlich dem SCS-Verfahren). Beide mit c(Q) wie `rain_full` angepasst (Varianten `int_full`, `acc_full`, `both_full` in `rain_fit.py`), gleiche blockweise Kreuzvalidierung. Synthetisch (`test_rain.py`) werden die Parameter wiedergefunden. Gepoolter NSE in der Kreuzvalidierung:

| Unterlieger | rain_full (Grundmodell) | + Intensität | + Ereignisfeuchte | beides |
|---|---|---|---|---|
| Echtershausen | 0,770 | 0,761 | 0,770 | 0,761 |
| Prümzurlay | 0,921 | 0,924 | 0,921 | 0,923 |
| Seffern | 0,644 | 0,642 | 0,642 | 0,642 |
| Alsdorf-Oberecken | 0,605 | 0,605 | 0,605 | 0,605 |

Die Vorhersagegüte (1, 3, 6 h) ändert sich um höchstens 0,02 (Prümzurlay 1 h: 0,48 → 0,50), die Scheitel der großen Hochwasser bei Echtershausen praktisch nicht (2018 −74 % → −74 %, 2021 −55 % → −53 % mit Intensität). In der Anpassung werden die Zusatzparameter abgeschaltet (rho = 0 oder w = 0) oder laufen an die Grenzen (Pc = 60 mm/h bei Echtershausen); im Training steigt der NSE bei Echtershausen nur von 0,856 auf 0,859.

Warum es nichts bringt (Diagnosen, alle im Training gerechnet, nicht kreuzvalidiert):
- Die Verstärkung des Oberliegers liegt bei Echtershausen bei 4,4 und bei Seffern bei 4,2, also nahe dem Flächenverhältnis (6,1 bzw. 7,7). Prüm 2 und Giesdorf dienen als Anzeiger für den Abfluss des ganzen Gebiets, der Regen erklärt nur den Rest. Erzwingt man die Massenbilanz (Verstärkung höchstens 1,3, `Model.amax`), fällt der NSE im Training von 0,86 auf 0,61 (Echtershausen) und von 0,81 auf 0,51 (Seffern), mit zwei verschiedenen Startwerten dasselbe Ergebnis. Der Regen aus neun Punktmessungen ersetzt den Oberlieger nicht.
- Je Ereignis angepasste Zwischengebietsparameter bei festem Routing (Echtershausen): 2021 erreicht den Scheitel nur mit 343 statt 445 m³/s, 2018 mit 122 statt 222 m³/s, beide mit phi_max an der Obergrenze 1. Selbst wenn der gesamte gemessene Regen abfließt, reicht es nicht. Das spricht dafür, dass der Stationsregen den scheitelbildenden Regen in diesen beiden Ereignissen nicht trifft (2018: Gewitterzelle, Prüm 2 sah nur 8 m³/s, der Abfluss in Echtershausen stieg in 4 h von 11 auf 216 m³/s). Das ist ein Indiz, kein Beweis.
- Daraus folgt: Die Form des Abflussbeiwerts ist nicht der Engpass. Nächster Schritt ist der flächige Regen (RADKLIM-RW des DWD, stündlich, 1 km, 2001 bis 2025; deckt 2003, 2011, 2018 und 2021 ab, nicht 1991 bis 1995).

### Aufruf
`python rain_fit.py shift` (Zeitversatz prüfen), `python rain_fit.py info` (Stationen, Gewichte, Abdeckung), `python rain_fit.py run --jobs 2` (Anpassung, Kreuzvalidierung, Hochwasser 2011/2018/2021, Vorhersagebetrieb; sieben Varianten, etwa 16 min). Ausgaben: `data/rain_cv_events.csv`, `rain_cv_summary.csv`, `rain_forecast_skill.csv`, `rain_params.json`, `rain_sim_floods.json`. `python routing_map.py` baut danach `gui/routing_map.html` mit Regenbalken, Stationsdreiecken, Regenlinie je Hochwasser, Vorhersagediagramm mit Regen und Überblickstabelle.

## Kettenvorhersage: Ausgabe alle 15 min, Vorlauf bis 48 h
Bisher wurde jeder Unterlieger einzeln vorhergesagt, mit dem Oberlieger nach der Ausgabezeit konstant. Damit endet die Vorhersage nach der Fließzeit einer einzigen Strecke. `chain_fc.py` rechnet stattdessen das ganze Netz von oben nach unten: Die Oberläufe (Prüm 2, Sinspelt, Giesdorf; kein Oberlieger) kommen aus einem Regen-Abfluss-Modell auf ihr ganzes Einzugsgebiet (53, 102, 18 km², gleiche Form wie die Zwischengebiete) oder bleiben ohne Regen konstant. Jeder Unterlieger wird aus der Vorhersage seines Oberliegers gerechnet, mit Regen für sein Zwischengebiet. Je Pegel gibt es eine Fehlerkorrektur: Vorhersage = Modell + beta(Vorlauf) · (Messung − Modell zur Ausgabezeit), beta aus den Trainingsereignissen geschätzt (Regressionssteigung des Modellfehlers über den Vorlauf). Die korrigierte Vorhersage geht in den Unterlieger.

Varianten: Persistenz; Oberlieger gehalten (bisher); Kette ohne Regen; Kette mit Regen (gemessen bis zur Ausgabezeit, danach 0) mit einem Modellsatz; Kette mit Regen und Umschalten zwischen Winter- und Starkregenmodellen (siehe unten, in der Karte „Kette mit Regen“); Kette mit dem tatsächlich gefallenen Regen nach der Ausgabezeit (perfekte Regenvorhersage, nur Obergrenze, nicht betrieblich). Prüfung: alle Ereignisse jedes Pegels 2004–2022 (Scheitel über dem 95-%-Quantil, 113 bis 212 je Pegel), Ausgabe alle 15 min von 48 h vor bis 12 h nach dem Scheitel (241 je Ereignis), Vorlauf 15 min bis 48 h in 15-min-Schritten; Modelle je Zeitblock ohne diesen Block angepasst. Zusätzlich unabhängig: Ereignisse 1989–2003 mit Modellen aus 2004–2022 (ohne Regen, weil es für diese Jahre keine Stundenwerte gibt).

Wie weit voraus hält die Vorhersage einen NSE von mindestens 0,7 (Fehler je Ereignis durch den Scheitel geteilt)? Stand nach der Umstellung auf Gebietsfeuchte (letzter Abschnitt); die Zahlen im Text darunter stammen teils von vorher.

| Pegel | Persistenz | Oberlieger gehalten | Kette | Kette + Regen, ein Modell | Kette + Regen, umschaltend | perfekter Regen (Obergrenze) |
|---|---|---|---|---|---|---|
| Prüm 2 | 5,25 h | – | – | 6 h | 6,25 h | 11,5 h |
| Sinspelt | 7,5 h | – | – | 8 h | 8,25 h | 10,75 h |
| Giesdorf | 4 h | – | – | 4 h | 4,25 h | 9 h |
| Echtershausen | 7,25 h | 10 h | 10 h | 10,5 h | 10,75 h | 14,5 h |
| Wiersdorf | 7 h | 9,75 h | 12,25 h | 12,5 h | 12,75 h | 15,5 h |
| Prümzurlay | 7,5 h | 10 h | 12 h | 12,5 h | 12,5 h | 15,75 h |
| Seffern | 5,25 h | 7 h | 7 h | 7,25 h | 7,5 h | 11 h |
| Alsdorf-Oberecken | 5,75 h | 7,25 h | 8,5 h | 8,5 h | 8,75 h | 12,75 h |

Unabhängig an 1989–2003 (ohne Regen): Prümzurlay 7,25 h (Persistenz), 9,5 h (gehalten), 11,5 h (Kette); Alsdorf-Oberecken 5,75 / 7,5 / 9 h. Güte gegen Persistenz bei Prümzurlay (2004–2022, umschaltend, mit Scheitel-Regel): 1 h 0,59, 6 h 0,71, 12 h 0,64, 24 h 0,44, 48 h 0,24; bisheriges Vorgehen 12 h 0,33, 24 h 0,14.

### Umschalten: Winterhochwasser oder Starkregenhochwasser
Die Hochwasser im Sommerhalbjahr verhalten sich anders als die Winterhochwasser (Prümzurlay, Median): Anstiegszeit 25 statt 39 h, doppelt so steiler Anstieg, größter stündlicher Gebietsregen 7 statt 2,4 mm/h, Ausgangsabfluss 3,7 statt 11 m³/s. Dazu gehören 2021, 2018 und Mai/Juni 2016. Deshalb gibt es je Pegel zwei Modellsätze:
- **Einteilung zur Ausgabezeit** (nur Daten bis dahin): Starkregenhochwasser, wenn der Monat Mai bis September ist **und** der größte 1-h-Gebietsregen über dem Einzugsgebiet des Zielpegels in den letzten 48 h mindestens 4 mm/h war; sonst Winterhochwasser. Gilt für die ganze Kette des Zielpegels. Die Trainingsereignisse werden mit derselben Regel zur Scheitelzeit eingeteilt (Prümzurlay: 12 von 113 Starkregen, Seffern 23 von 162).
- **Wintermodelle** an den Winterereignissen angepasst, **Starkregenmodelle** an den wenigen Starkregenereignissen mit Strafterm zum Gesamtmodell (`routing._run(..., prior=(Gesamtmodell, 1.0))`, Abweichung je Parameter auf ein Viertel der zulässigen Spanne bezogen), damit sie nur abweichen, wo die Daten es tragen. Ein schwächerer Strafterm (0,3) ergab fast dasselbe.
- Die Einteilung war in der Prüfung fehlerfrei: kein Winterhochwasser als Starkregen eingestuft, jedes Starkregenereignis 17 bis 48 h vor dem Scheitel erkannt; mit bekannter Art statt Erkennung wird es nicht besser.

Scheitel erkannt (Höhe ±15 %, Zeit ±2 h) bei Ausgabe 6 h vor dem Scheitel:

| Pegel | alle Ereignisse: ein Modell → umschaltend | nur Starkregen: ein Modell → umschaltend (Anzahl) |
|---|---|---|
| Echtershausen | 34 % → 40 % | 7 % → 14 % (14) |
| Wiersdorf | 28 % → 31 % | 21 % → 36 % (14) |
| Prümzurlay | 58 % → 63 % | 25 % → 67 % (12) |
| Seffern | 35 % → 43 % | 4 % → 4 % (23) |
| Alsdorf-Oberecken | 39 % → 39 % | 11 % → 28 % (18) |

Ohne Scheitel-Regel (siehe unten) waren es an Prümzurlay bei Starkregen 25 % → 75 % (6 h vorher), 17 % → 50 % (9 h) und 17 % → 33 % (12 h). Beispiele 6,25 h vor dem Scheitel (ohne Scheitel-Regel): Juli 2021 ein Modell 598 m³/s um 11:15, umschaltend 515 m³/s um 08:00 und danach fallend (gemessen 596 m³/s um 06:30); Juni 2018 269 m³/s um 20:00 gegen 256 m³/s um 18:45 (gemessen 209 m³/s um 19:30). Bei 24 h und mehr Vorlauf wird der NSE an einzelnen Pegeln etwas schlechter. Ohne Regen gibt es kein Umschalten (die Einteilung braucht den Regen).

### Scheitel-Regel: kein Scheitel am Unterlieger ohne gemessenen Scheitel oben
Ein Pegel mit Oberliegern zeigt keinen Scheitel, solange nicht an allen seinen direkten Oberliegern ein Scheitel gemessen ist (`chain_fc.peak_passed`: Höchstwert der letzten 48 h mindestens 30 min zurück, der Abfluss jetzt mindestens 1 % darunter und in den letzten 30 min nicht gestiegen, der Höchstwert mindestens 20 % über dem kleinsten Wert davor). Die Erkennung kommt über alle Ereignisse der Oberlieger im Median 45 min nach dem echten Scheitel (Wiersdorf 30 min, 90 % innerhalb von 2,5 h); die erste Fassung (1 h, 3 %) lag bei 1,5 h, in Wiersdorf bei 2,5 h, weil der Abfluss unter dem Stausee nach dem Scheitel nur langsam fällt. Juli 2021: Scheitel Wiersdorf 04:45, erkannt 05:15. Steigt die Vorhersage über den jetzigen Wert, bleibt sie nach ihrem Höchstwert stehen und ist als Mindestwert zu lesen (`hold_after_max`). Das gilt die Kette hinunter und für alle betrieblichen Varianten, nicht für die perfekte Regenvorhersage. Die Karte zeigt dann „Scheitel oben noch nicht gemessen: kein Scheitel vorhergesagt“.

Wirkung (kreuzvalidiert, Kette mit Regen, umschaltend; Vergleich ohne / mit Regel):
- Verfrühte Scheitel (Maximum der Vorhersage mehr als 2 h vor dem echten Scheitel, danach mindestens 10 % fallend) werden selten: Prümzurlay bei Ausgabe 24 h vorher 18 % → 4 %, 18 h vorher 6 % → 2 %; Alsdorf-Oberecken 18 h vorher 71 % → 30 %; Echtershausen 24 h vorher 75 % → 38 %; Seffern 12 h vorher 48 % → 33 %.
- Der vorhergesagte Wert zur tatsächlichen Scheitelzeit wird genauer: Prümzurlay 12 h vorher Fehler im Median 6,7 % → 5,4 %, 9 h vorher 4,7 % → 3,9 %, an den anderen Pegeln ähnlich.
- Kosten: Weil die Vorhersagen der Oberlieger nicht mehr fallen, steigen die Unterlieger etwas länger; der Zeitpunkt des Vorhersage-Maximums wird ungenauer. Prümzurlay: „Scheitel erkannt“ 6 h vorher 64 % → 63 %, bei Starkregen 75 % → 67 %; NSE bei 12 h 0,756 → 0,747, nutzbarer Vorlauf 13 → 12,75 h. Wiersdorf 12 h vorher „erkannt“ 31 % → 18 %.
- Juli 2021, 6,25 h vor dem Scheitel: 540 m³/s ab +7,5 h und dann gleichbleibend (vorher fallend auf 379 m³/s nach 24 h); Wiersdorf hatte da noch keinen Scheitel. 1 h vor dem Scheitel (beide Oberlieger-Scheitel vorbei) unverändert.

### Versucht und nicht übernommen
- **Woher der Fehler an Prümzurlay kommt:** Mit gemessenen statt vorhergesagten Zuflüssen (Wiersdorf und Sinspelt) hielte die Vorhersage einen NSE von 0,93 bei 12 h und 0,86 bei 24 h. Die letzte Strecke ist also gut, fast der ganze Fehler kommt aus der Vorhersage der Zuflüsse, je etwa zur Hälfte von Prüm und Enz.
- **Gelernte Nachkorrektur** (lineare Regression je Vorlauf auf Kettenvorhersage, Trends an Prümzurlay, Wiersdorf, Sinspelt, Regensummen): Prümzurlay 12,8 → 15 h bei NSE ≥ 0,7, ändert aber an der Form großer Scheitel nichts (2021 Maximum weiter zu spät). Gradient Boosting war schlechter als linear (zu wenige Ereignisse).
- **Formerhaltende Fehlerkorrektur:** (a) Modellspeicher auf Messwert und Anstieg setzen und mit den Modellzeitkonstanten leerlaufen lassen: schlechter (2021 über 1000 m³/s, die Speicher der Oberläufe laufen mit 40 bis 50 h viel zu langsam leer). (b) Eigene Fehlerspeicher mit Zeitkonstanten aus dem Training: kleiner Gewinn im Mittel, 2021 zu hoch (747 m³/s).
- **Rückgang fallender Oberlieger** (gemessen fallender Oberlieger fällt in der Vorhersage mit seiner Rückgangskurve weiter): kleiner Gewinn (Prümzurlay erkannt 3 h vorher 73 → 76 %), 2021 unverändert.
- **Schneller Abflussweg bei Starkregen** im Zwischengebiet (Anteil des wirksamen Regens über einen schnellen Speicher, intensitätsabhängig): von der Anpassung abgeschaltet, kein Gewinn.

Befunde:
- Pegel näher an der Mündung lassen sich weiter vorhersagen, aber die Grenze ist die Fließzeit durch das Netz: von Prüm 2 bis Prümzurlay bei Hochwasser etwa 16 bis 18 h (Modelle: Prüm 2 → Echtershausen 9 bis 11 h, Stausee 3 bis 5 h, Wiersdorf → Prümzurlay 4 bis 5 h), auf der Nims Giesdorf → Alsdorf-Oberecken etwa 9 bis 10 h. Die Kette verlängert den nutzbaren Vorlauf bei Prümzurlay von 10 auf 12,75 h und bei Alsdorf-Oberecken von 7,5 auf 9 h.
- Sobald Wiersdorf und Sinspelt ihren Scheitel überschritten haben (im Median 3,5 h vor dem Scheitel in Prümzurlay), trifft die Kette den Scheitel in Prümzurlay in 75 % der Ereignisse, mit gemessenen Zuflüssen wären es 76 % (gemessen vor Umschalten und Scheitel-Regel).
- 2021 war außerhalb jeder Erfahrung: Echtershausen → Wiersdorf 0,5 h statt im Median 4,25 h, Wiersdorf → Prümzurlay etwa 2 h statt 4,75 h. Die Laufzeiten verschiedener Strecken im selben Ereignis hängen nicht zusammen (Rangkorrelation −0,16 über 210 Ereignisse), aus den oberen Strecken lässt sich die Laufzeit unten also nicht ablesen. Auch mit 2021 im Training bleibt die Vorhersage 1 h vor dem Scheitel zu spät (umschaltend 664 m³/s um 10:00 statt 596 um 06:30). Eine bessere Abbildung solcher Extreme braucht mehr Extremereignisse, zum Beispiel aus den anderen 135 Abflusspegeln der LfU-Pakete (Kyll, Ahr, Sauer).
- Was nach der Fließzeit kommt, fällt erst noch als Regen: Mit perfekter Regenvorhersage wären es 17 h bei NSE ≥ 0,7 und eine Güte gegen Persistenz um 0,8 bis 48 h. Längere Vorhersagen sind nur mit einer Niederschlagsvorhersage möglich (zum Beispiel ICON-D2 des DWD, bis 48 h), und eine echte Regenvorhersage liegt deutlich unter dieser Obergrenze.
- Ab etwa 24 h Vorlauf ist der NSE auch mit Kette und Regen negativ: Die Vorhersage ist dann besser als Persistenz, aber nicht besser als ein mittlerer Wert. Die Oberläufe (Prüm 2, Sinspelt, Giesdorf) sind das schwächste Glied (Regen-Abfluss-Modelle mit NSE um 0,3 im Training, Regen aus neun Stationen).
- Die Fehlerkorrektur beta je Vorlauf ist nicht steigend (ein Fehler kann mit dem Vorlauf nur verblassen); vorher führten Ausreißer bei langen Vorläufen zu einem Wiederanstieg der Vorhersage am Ende des Horizonts.
- Die Karte rechnet die Vorhersage für jede Ausgabezeit im Browser (`forecast_core.js`), mit den Modellen aus dem Zeitblock ohne das gezeigte Hochwasser (vor 2004: Modelle 2004–2022) und zeigt die erkannte Hochwasserart. Die Einteilung je Pegel und Zeitpunkt wird in Python vorberechnet. Python und JavaScript stimmen auf 2·10⁻¹³ überein (`test_forecast_js.py`, 492 Fälle).

## Vorhersage in cm mit Unsicherheit: Scheitel, Warnung, Band
Der Vorhersage-Tab der Karte zeigt jetzt für den gewählten Pegel und die gewählte Ausgabezeit ein Lagebild: Wasserstand jetzt mit Tendenz, den erwarteten Scheitel mit Spanne (sobald die Welle oben gemessen ist), den Warnstatus (seit 13.11.2026 erst ab HW 10, siehe letzter Abschnitt), die Wahrscheinlichkeit, dass HW 10 bis HW 100 in 6, 12, 24 oder 48 h erreicht werden (Meldehöhe und HW 2 nur zur Einordnung), und die Ganglinie in cm mit Median und 80-%-Bereich. Die Modellvarianten, Gütekurven und Prüfzahlen sind eingeklappt.

**Schwellen.** Aus der Konfiguration von hochwasser.rlp.de (`data/pegelkennwerte.json`). Eine Meldehöhe gibt es nur für Echtershausen (160 cm) und Prümzurlay (270 cm); HWx ist der x-jährliche Wasserstand, von der HVZ aus HQx mit der heute gültigen W-Q-Beziehung gerechnet. Gerechnet wird auf dem Abfluss (HQx; Meldehöhe über die W-Q-Beziehung: Prümzurlay 270 cm = 105 m³/s, Echtershausen 160 cm = 66 m³/s).

**Wasserstand.** `rating.py`: Mediane des Wasserstands in 30 logarithmischen Abflussklassen aus jüngsten Wertepaaren, darüber die HVZ-Paare (HQ2, HW2) bis (HQ100, HW100), monoton gemacht, linear in log Q. Die Beziehung hat sich verschoben: In Prümzurlay steht das Wasser seit Januar 2022 bei gleichem Abfluss rund 25 cm tiefer (bei 20 m³/s 76 statt 101 cm), in Seffern war es von Juli 2021 bis März 2022 rund 13 cm höher, in Alsdorf-Oberecken seit November 2021 rund 6 cm tiefer. Deshalb nur Wertepaare ab 2022 (Prümzurlay), April 2022 (Seffern), November 2021 (Alsdorf-Oberecken), sonst ab 2016, dazu die Welle Oktober 2026. Abweichung der Kurve von den Wertepaaren im Median 0 cm, 95 % innerhalb weniger cm. Ältere Hochwasser erscheinen in cm mit der heutigen Beziehung: Prümzurlay 2003 damals 492 cm gemessen, heute wären es bei 252 m³/s etwa 457 cm; 2021 698 gegen 684 cm. Die Karte nennt den damals gemessenen Höchststand dazu.

**Unsicherheit.** `uncert.py archive` gibt jede Stunde über alle Jahre eine Vorhersage aus (nicht nur bei Hochwasser), 2004–2022 mit den Modellen ohne den jeweiligen Zeitblock (Kette mit Regen, umschaltend, Scheitel-Regel), 1989–2003 ohne Regen mit den Modellen aus 2004–2022: rund 162.000 Ausgaben je Pegel. Fehler r = ln((Messung + d) / (Vorhersage + d)), d = 10 % von MQ. Die Quantile von r hängen vom Zustand zur Ausgabezeit ab: ruhig / Anstieg (Regen der letzten 24 h ≥ 10 mm, Anstieg über 6 h um mehr als 5 % oder über 3 MQ) / Welle von oben unterwegs (Scheitel an allen Oberliegern gemessen, am Pegel noch nicht) / fallend, und von der Abflussstufe (größerer Wert aus jetzt und Vorhersage-Höchstwert: unter 2, 2 bis 8, ab 8 MQ; nach dem Pinball-Verlust besser als die Stufe aus dem jetzigen Abfluss allein und als gar keine Stufe). Klassen mit weniger als 200 Fällen bekommen die Quantile ihrer Abflussstufe. Band und Median: Vorhersage · exp(Quantil). Überschreitung bis zum Zeitpunkt L: Die Fehler sind über die Vorläufe stark gekoppelt; ein Fall liegt durchgehend im selben Rang, die Wahrscheinlichkeit ist das laufende Maximum der Einzelwahrscheinlichkeiten. Scheitel (nur „Welle unterwegs“, Vorhersage steigt noch um mehr als 2 % auf mindestens 2 MQ): Quantile von Höhenverhältnis und Zeitversatz je Vorlauf bis zum vorhergesagten Scheitel (0–3, 3–6, 6–12, 12–48 h); gemessener Scheitel = höchster Wert bis 12 h nach dem vorhergesagten. Die Karte nimmt für die Hochwasser 2011, 2018 und 2021 die Tabellen ohne deren Zeitblock, für die älteren Hochwasser (ohne Regen) die Tabellen der Kette ohne Regen aus 2004–2022.

**Prüfung** (verschachtelt: Tabellen jeweils ohne den geprüften Zeitblock; 1989–2003 ganz unabhängig; Stand nach der Umstellung auf Gebietsfeuchte, siehe nächster Abschnitt).
Spalten bis „Fehlalarme“: Kette mit Regen bis zur Ausgabezeit (wie in der Karte); die letzten beiden: dieselbe Kette mit dem tatsächlich gefallenen Regen danach (Obergrenze dessen, was eine Regenvorhersage bringen kann). Band: Anteil der Messwerte im 80-%-Bereich bei 6 / 24 h Vorlauf. Warnung: halbes HQ2 (häufig genug für eine Aussage), Vorwarnzeit = wie lange vor dem Überschreiten die Wahrscheinlichkeit für die nächsten 48 h ununterbrochen mindestens 50 % war; Fehlalarme = Warnphasen ohne Überschreitung in 48 h.

| Pegel | Band alle | Band ab 8 MQ | 1989–2003 ohne Regen | nachträglich: Messung ab HQ2/2 | Überschr. HQ2/2 | Vorwarnzeit (Median) | ≥ 3 h | Fehlalarme | mit perfektem Regen: Vorwarnzeit | Fehlalarme |
|---|---|---|---|---|---|---|---|---|---|---|
| Prüm 2 | 79 / 78 | 79 / 80 | 71 / 76 | 64 / 49 | 48 | 0,2 h | 0 % | 2 / 32 | 0,8 h | 27 / 85 |
| Sinspelt | 79 / 80 | 77 / 75 | 82 / 79 | 60 / 43 | 17 | 0,2 h | 0 % | 1 / 13 | 1,0 h | 1 / 18 |
| Giesdorf | 79 / 79 | 78 / 77 | 78 / 81 | 60 / 42 | 41 | 0,0 h | 0 % | 2 / 15 | 0,0 h | 6 / 28 |
| Echtershausen | 79 / 78 | 75 / 77 | 76 / 76 | 68 / 55 | 57 | 7,0 h | 84 % | 63 / 145 | 8,0 h | 39 / 130 |
| Wiersdorf | 79 / 79 | 77 / 76 | 80 / 80 | 68 / 52 | 33 | 7,8 h | 79 % | 66 / 122 | 8,5 h | 21 / 87 |
| Prümzurlay | 78 / 80 | 82 / 78 | 78 / 80 | 69 / 53 | 39 | 10,2 h | 85 % | 42 / 93 | 11,5 h | 12 / 85 |
| Seffern | 79 / 79 | 75 / 79 | 86 / 85 | 59 / 45 | 35 | 1,2 h | 14 % | 6 / 38 | 1,8 h | 9 / 44 |
| Alsdorf-Oberecken | 79 / 79 | 78 / 77 | 84 / 86 | 64 / 49 | 34 | 1,5 h | 29 % | 65 / 118 | 2,5 h | 19 / 86 |

- Das Band hält seine 80 % (Bedingung nur auf Wissen zur Ausgabezeit), auch bei hohem Abfluss und auch in der unabhängigen Prüfung 1989–2003. Wählt man die Fälle nachträglich danach aus, dass die Messung danach HQ2/2 erreichte, liegt sie seltener im Band (6 h: 59 bis 69 %, 24 h: 42 bis 55 %), vor allem nach oben: Regen nach der Ausgabezeit kennt das Modell nicht. Das ist die ehrliche Zahl für „im echten Hochwasser“.
- Die Wahrscheinlichkeiten stimmen (halbes HQ2 in 6 h, alle Pegel): angesagt 5–20 % → eingetreten 14 %, 20–40 % → 38 %, 40–60 % → 61 %, 60–80 % → 81 %, 80–95 % → 90 %, über 95 % → 98 %. Im oberen Bereich sind sie eher vorsichtig (die gekoppelte Rechnung ist eine Untergrenze).
- Warnung: Prümzurlay wird im Median 10 h vorher gewarnt, Wiersdorf knapp 8 h, Echtershausen 7 h; Meldehöhe Prümzurlay: 4 Überschreitungen 2004–2022, 3 davon mindestens 6 h vorher gewarnt, Vorwarnzeit im Median 6,6 h, kein Fehlalarm. Seit der Umstellung auf Gebietsfeuchte gibt es dafür mehr Fehlalarme beim halben HQ2 (Prümzurlay 42 von 93 Warnphasen, vorher 6 von 53; die Vorwarnzeit war vorher 7,2 h): Die Modelle geben bei nassem Gebiet mehr Abfluss aus dem schon gefallenen Regen. Mit bekanntem Regen danach sinken die Fehlalarme wieder (12 von 85). Seffern und Alsdorf-Oberecken (Nims) nur gut 1 h, die Oberläufe praktisch gar nicht: Dort kommt die Welle aus Regen, den das Modell erst kennt, wenn er gefallen ist.
- Scheitel, sobald die Welle oben gemessen ist: Prümzurlay bei Scheitel in 0–3 h Höhe im Median auf 2 %, Zeit auf 0,5 h; in 3–6 h 2,5 % und 1 h; Höhe und Zeit je in 59 bis 80 % der Fälle im 80-%-Bereich (bei 6–12 h nur 59 Ausgaben). Ein vorhergesagter Scheitel in mehr als 12 h ist unsicher (Höhe ± 20 %, Zeit ± 10 h und mehr), die Karte sagt das dazu. Juli 2021, 1 h vor dem Scheitel: erwartet 717 cm (80 %: 706 bis 733) um 09:00, gemessen 684 cm (heutige Beziehung) um 06:30. Das Extrem liegt außerhalb der Erfahrung, auch die Spanne trifft es nicht.
- Die Rechnung im Browser stimmt mit Python überein (`test_uncert_js.py`: Band 7·10⁻¹⁴, Wahrscheinlichkeiten 2·10⁻¹⁴, W-Q 3·10⁻¹³ cm).
- Live-Betrieb mit Regenvorhersage: siehe die beiden nächsten Abschnitte.


## Gebietsfeuchte im Abflussbeiwert (Umstellung 12.11.2026)
Anlass: Nachrechnung der Welle vom Oktober 2026 mit dem tatsächlich gefallenen Regen. 65 bis 70 mm in vier Tagen nach dem trockenen Sommer gaben nur 7 bis 10 mm Abfluss; die Modelle rechneten 3- bis 4-mal so viel (Prümzurlay 65 statt 17 m³/s, Echtershausen 50 statt 18). Im Vorhersagearchiv 2004–2022 war das kein Einzelfall: Lag der Abfluss vorher unter 0,3 MQ, war der gemessene Scheitel im Median nur 20 bis 29 % des mit perfekt bekanntem Regen vorhergesagten, über 1 MQ stimmte es. Ursache: angepasst nur an Hochwassern, Feuchte nur aus dem 7-Tage-Vorregen.

Umstellung (`chain_fc.py`, `lateral.py`):
- Anpassung zusätzlich an **Regenereignissen ohne Hochwasser** (48-h-Gebietsregen ab 25 mm, nicht innerhalb von 3,5 Tagen um einen Hochwasserscheitel; je Gebiet etwa 50 Ereignisse), Fehler je Fenster durch max(Scheitel, 95-%-Quantil) geteilt.
- **Gebietsfeuchte** aus dem Abfluss: x = API7 + w · B im Abflussbeiwert, B = Abfluss am Bezugspegel zu Beginn des Abflussfensters / MQ (bei der Vorhersage 48 h vor der Ausgabezeit). Verglichen wurde auch ein Faktor B^g / (B^g + b50^g): Er dämpft trockene Lagen stärker, unterschätzt aber die Oktoberwelle (Prüm 2: 16 % des Scheitels), weil der trockene Fensterbeginn das ganze Ereignis bremst; die additive Form lässt den Boden im Ereignis nass werden (Prüm 2 Oktoberwelle: Scheitel 58 % statt 165 %, Regenereignisse 2019–2022 ohne Hochwasser 1,07-fach statt 2,3-fach; Sinspelt 2,3- statt 8,9-fach).
- Fehlertabellen zusätzlich nach Feuchte (jetzt unter 0,3 MQ: trocken). Dort lagen die Vorhersagen nach 48 h im Median um den Faktor 2,3 (Prümzurlay) bis 3,5 (Prüm 2) zu hoch; die Seiten zeigen den Median nach dieser Statistik.
- Versucht und nicht übernommen: den Sockel c0 bei Niedrigwasser dauerhaft abziehen (`C0_SHIFT`). Die Vorhersage steigt ohne Regen dann nicht mehr zum Modellsockel an, fällt dafür viel zu schnell (Prümzurlay von 1,4 auf 0,4 m³/s in 48 h), weil dem Modell ein langsamer Grundwasseranteil fehlt.

Wirkung (kreuzvalidiert):
- Hochwasserfenster (wie bisher geprüft): fast unverändert, Kette mit Regen bis zur Ausgabezeit etwas schwächer bei langen Vorläufen (Prümzurlay NSE 12 h 0,747 → 0,736), „Scheitel erkannt 6 h vorher“ gleich oder besser (Prümzurlay 63 → 65 %, Echtershausen 40 → 43 %).
- Alle Stunden mit perfekt bekanntem Regen (zeigt, was eine Regenvorhersage höchstens bringen kann): Güte der Überschreitungswahrscheinlichkeit (halbes HQ2, 24 h, gegen Klimatologie) Prümzurlay 0,47 → 0,55, Echtershausen 0,46 → 0,53, Prüm 2 0,28 → 0,38, Alsdorf-Oberecken 0,30 → 0,38; deterministische Fehlalarme Prüm 2 27 → 8 %, Echtershausen 15 → 6 %, Alsdorf-Oberecken 19 → 11 %; Vorwarnzeit Prümzurlay 6,3 → 11,5 h.
- Oktober 2026 mit den damaligen ICON-D2-Läufen (`hindcast.py`, `gui/hindcast_2026-10.png`): Prümzurlay 1 bis 2 Tage vorher ein Höchstwert von 14 bis 16 m³/s statt gemessen 15 bis 17 (vorher 65 bis 75); der Scheitel kommt im Modell meist 6 bis 8 h zu spät; Echtershausen 12 bis 15 statt 18; Prüm 2 zu niedrig (2,2 statt 5,4). Ohne Regenvorhersage zeigt die Kette bis zum Regen am 25.10. nichts. Mittlerer Fehler über 48 h (Ausgaben 23.10. 12:00 bis 26.10.): Prümzurlay 3,1 m³/s mit ICON-D2, 3,2 mit gemessenem Regen, 5,1 ohne Regenvorhersage. Eine kleine Welle (3 % von HQ) und deterministisch, also ein Beispiel, keine Statistik. Nach der Starkregen-Überarbeitung (13.11.2026, letzter Abschnitt): Prümzurlay 1 bis 2 Tage vorher 16 bis 21 statt gemessen 15 bis 17 m³/s, mittlerer Fehler über 48 h 3,0 mit ICON-D2 und 3,1 mit gemessenem Regen.

## Regenvorhersage ICON-D2 und Live-Betrieb
**Wie gut ist ICON-D2 hier?** (`verif_rain.py`, ICON-D2 an den neun Stationsorten über Open-Meteo gegen die DWD-Stundenwerte, 24.11.2022 bis 11.11.2026; Gebietsregen wie im Modell)
- Vorlauf 0 bis 3 h: Korrelation der 6-h-Summen 0,81 bis 0,90, Tagessummen 0,87 bis 0,93; 6-h-Starkregen (über 10 mm im Gebiet) zu 50 bis 69 % vorhergesagt.
- Vorlauf 24 bis 27 h (Previous Runs API, ab 2024): Korrelation 0,63 bis 0,67 (6 h) und 0,77 bis 0,79 (Tag); 6-h-Starkregen zu 37 bis 45 % vorhergesagt, 48 bis 58 % der vorhergesagten kommen nicht.
- ICON-D2 liegt an den höheren Stationen 12 bis 17 % unter der Messung (Station / ICON: Prüm-Watzerath 1,14, Lauperath 1,14, Schneifelforsthaus 1,19, Winterspelt 1,18, Olmscheid 1,21, Weißenseifen 1,17; Olsdorf 1,00, Wiersdorf 0,94), über alle Vorläufe gleich; `live.py` gleicht das je Station aus. Das Modell ist auf Stationsregen angepasst.

**Live-Betrieb** (`live.py`, Seite `gui/live.html`):
- Daten: Pegel der HVZ (letzte 5 Tage, 15 min), Stundenregen der Stationen (DWD recent bis gestern, dazu die 10-Minuten-Werte von heute), Lücken bis jetzt aus ICON-D2; Vorhersage ICON-D2 (15 min, 48 h) und ICON-D2-EPS (20 Läufe, stündlich, 48 h) über Open-Meteo an den Stationsorten, Gebietsregen mit denselben Stationsgewichten wie bei der Anpassung.
- Rechnung: dieselbe Kette (Modellsatz aus allen Ereignissen 2004–2022, Variante `kette_vorhersage`: Regen nach jetzt aus der Vorhersage, Umschalten Winter / Starkregen, Scheitel-Regel) für den Hauptlauf und jeden Ensemble-Lauf. Unsicherheit: Mischung über die 21 Läufe, je Lauf mit dem Restfehler des Abflussmodells bei bekanntem Regen (Archiv mit dem gemessenen Regen als „perfekter Vorhersage“, kreuzvalidiert). Zum Vergleich die Kette ohne Regenvorhersage und, für Prümzurlay und Alsdorf-Oberecken, die amtliche Vorhersage der HVZ.
- Seite: alle acht Pegel auf einen Blick (jetzt, Warnstatus ab HW 10, Höchststand in 48 h mit Spanne), je Pegel Ganglinie in cm mit 60- und 80-%-Bereich, Regen im Einzugsgebiet (gemessen, ICON-D2, Ensemble), Überschreitungswahrscheinlichkeiten für Meldehöhe und HW 2 bis HW 100. Ausdrücklich keine amtliche Warnung.
- Aufruf auf dem eigenen Rechner (mit Internet): `python live.py`. Mit Python 3, numpy, pandas, scipy; der C-Kern wird beim ersten Lauf übersetzt (sonst numpy). Abrufe sparsam halten (Open-Meteo frei für nicht-kommerzielle Nutzung; DWD Open Data; HVZ-Schnittstelle).
- Grenzen: Die Streuung des Ensembles ließ sich nicht prüfen (Open-Meteo bewahrt die einzelnen Läufe nur drei Tage auf, ein Archiv der Einzelläufe gibt es erst ab April 2026 und nur für den Hauptlauf). Für die großen Hochwasser 2011, 2018, 2021 gibt es keine archivierte ICON-Vorhersage. Bei Niedrigwasser zieht das Modell ohne Regen zu seinem Sockel; die gezeigte Vorhersage ist deshalb der Median nach der Fehlerstatistik, nicht der rohe Modelllauf.

## Warnung erst ab HW 10 (Umstellung 13.11.2026)
Vorgabe: HQ2 richtet keinen Schaden an; erste Schäden ab HQ10. Gewarnt wird deshalb an allen acht Pegeln einheitlich erst ab **HW 10** (der Wasserstand zu HQ10). Meldehöhe und HW 2 stehen nur noch zur Einordnung da. Die Schwelle lässt sich je Pegel umstellen: `uncert.py`, `WARN` (z. B. `{"PZ": "HQ20"}`) und `WARN_DEFAULT`.

**Regel** (Live-Seite und Karte gleich): P = Wahrscheinlichkeit, HW 10 binnen 48 h mindestens einmal zu erreichen (auf 5 % gerundet).
- unter 10 %: keine Warnung
- ab 10 %: **Vorwarnung** (HW 10 möglich)
- ab 50 %: **Warnung** (HW 10 wahrscheinlich)
- jetzt darüber: HW 10 überschritten

**Rückrechnung** (`uncert.py tables`, jede Stunde, kreuzvalidiert). Für jedes Hochwasser über HQ10 wird festgehalten, wie lange vor dem Überschreiten die Vorwarnung bzw. die Warnung ununterbrochen stand (höchstens 48 h gezählt, auf halbe Stunden abgerundet), dazu die erste und letzte Ankündigung in den 48 h davor. Je Pegel gibt es nur ein bis drei solche Hochwasser, deshalb steht hier jedes einzeln und keine Mittelwerte.

Bedeutung der Werte in der Tabelle:
- **Vorwarnung / Warnung:** so viele Stunden vor dem Überschreiten stand sie ununterbrochen.
- **0:** in den 48 h davor nicht erreicht.
- **„zurückg.“:** angekündigt, aber vor dem Überschreiten wieder zurückgenommen.

Spalten:
- **ohne Regenvorhersage:** Kette mit Regen bis zur Ausgabezeit.
- **perfekte Regenvorhersage:** mit dem später gemessenen Regen.

Eine echte Regenvorhersage liegt dazwischen. Für diese Hochwasser gibt es keine archivierte ICON-Vorhersage.

Stand nach der Starkregen-Überarbeitung vom 13.11.2026, Schritt 2 (nächster Abschnitt). Vorwarnung / Warnung, jeweils so lange vor dem Überschreiten ununterbrochen; ≥ 44 h heißt: praktisch so früh, wie die 48-h-Rechnung überhaupt reicht.

| Pegel | Hochwasser | Scheitel / HQ10 | ohne Regenvorhersage | perfekte Regenvorhersage |
|---|---|---|---|---|
| Prüm 2 | 29.01.2021 | 1,0 | 15 min / 0 | 15 min / 0 (Schneeschmelze) |
| Prüm 2 | 14.07.2021 | 3,7 | 15 min / 0 | 47 h / 47 h |
| Sinspelt | 14.07.2021 | 2,2 | 15 min / 15 min | 47 h / 47 h |
| Giesdorf | 14.07.2021 | 2,9 | 0 / 0 (höchstens 5 %) | 40,5 h / 38,5 h |
| Echtershausen | 01.06.2018 | 1,6 | 45 min / 0 | zurückg. (Vorwarnung 29,5–5,5 h vorher) / 0 |
| Echtershausen | 14.07.2021 | 3,3 | 10 h / 2 h | 47 h / 47 h |
| Wiersdorf | 01.06.2018 | 1,5 | 2 h / 1 h | 1 h / 15 min |
| Wiersdorf | 14.07.2021 | 3,0 | 11,5 h / 2,5 h | 47,5 h / 47,5 h |
| Prümzurlay | 01.06.2018 | 1,1 | 14 h / zurückg. (Warnung 8–4 h vorher) | zurückg. (Vorwarnung 38–4 h vorher) / zurückg. (Warnung 7–4 h vorher) |
| Prümzurlay | 14.07.2021 | 3,0 | 11,5 h / 3,5 h | 47,5 h / 47,5 h |
| Seffern | 05.06.2016 | 1,2 | 2 h / 0 | zurückg. (Vorwarnung 26–5 h vorher) / zurückg. (Warnung 18–13 h vorher) |
| Seffern | 14.07.2021 | 4,0 | zurückg. (Vorwarnung 5,5–4,5 h vorher) / 0 | 46,5 h / 45,5 h |
| Alsdorf-Oberecken | 14.07.2021 | 2,8 | 8,5 h / 2,5 h | 47,5 h / 47,5 h |

**Ältere Hochwasser 1989–2003** (unabhängig, nur Kette ohne Regen, Stundenregen erst ab 2004), Vorwarnung / Warnung:
- Echtershausen 1991: 7,5 / 5,5 h; 1993: 15 / 15 min; 1995: 4 h / 15 min; 2003: 2,5 / 1,5 h
- Seffern 2003: 3 / 1 h
- Alsdorf-Oberecken 2003: 9 / 2 h
- Oberläufe: kaum
- Prümzurlay 1993 und 2003: nicht nachrechenbar, weil Wiersdorf damals bei 250 cm abgeschnitten aufgezeichnet ist.

**Fehlalarme**: Warnphasen ohne HW 10 in den 48 h danach. Lücken bis 12 h zählen zur selben Phase. In Klammern: davon knapp, d. h. der Abfluss erreichte danach mindestens 80 % von HQ10.

| Pegel | 2004–2022 ohne Regenvorhersage: Vorwarnung / Warnung | perfekte Regenvorhersage | 1989–2003 ohne Regen |
|---|---|---|---|
| Prüm 2 | 5 (1) / 0 | 7 (1) / 0 | 6 (4) / 0 |
| Sinspelt | 1 / 0 | 3 / 0 | 5 (4) / 0 |
| Giesdorf | 3 (1) / 0 | 4 (1) / 0 | 9 (3) / 0 |
| Echtershausen | 16 / 2 | 5 / 1 | 2 (1) / 0 |
| Wiersdorf | 9 (1) / 0 | 7 (1) / 1 | 1 (1) / 1 (1) |
| Prümzurlay | 7 (1) / 0 | 5 (2) / 0 | 2 (2) / 2 (2) |
| Seffern | 15 / 0 | 4 / 0 | 6 (2) / 1 (1) |
| Alsdorf-Oberecken | 40 / 1 | 11 (1) / 0 | 5 (4) / 1 (1) |

Was daraus folgt:
- **Mit richtigem Regen warnt die Kette im Juli 2021 früh, an allen acht Pegeln.** Die Warnung (≥ 50 %) stand mit perfekter Regenvorhersage 38,5 bis 47,5 h vor dem Überschreiten, also praktisch so früh, wie die 48-h-Rechnung reicht. Vor der Überarbeitung waren es 0 bis 3,5 h, an den Oberläufen gar keine Warnung.
- **Ohne Regenvorhersage bleibt es kurz.** An der Prüm unterhalb des Stausees und in Alsdorf-Oberecken steht die Warnung 2,5 bis 3,5 h vorher, die Vorwarnung 8,5 bis 11,5 h; an den Oberläufen und in Seffern gibt es keine Warnung. Dafür gibt es mehr Vorwarnungen ohne HW 10, vor allem an der Nims (Seffern 15, Alsdorf-Oberecken 40 in 19 Jahren). Im Betrieb rechnet die Live-Seite mit ICON-D2; wie nah das an „perfekt“ kommt, hängt an der Regenvorhersage.
- **Warnungen ohne HW 10 bleiben selten.** Ohne Regenvorhersage sind es 3 (Echtershausen im Januar 2011 und 2021, Alsdorf-Oberecken im Juni 2018; erreicht wurden 61 bis 75 % von HQ10). Mit perfektem Regen sind es 2, beide am 12.07.2021 (Echtershausen, Wiersdorf), zwei Tage vor dem eigentlichen Hochwasser. Nach der 48-h-Regel zählen sie als Fehlalarm, praktisch sind es frühe Warnungen.
- **Die Gewitter bleiben schwach.** Echtershausen 2018 wurde auch mit gemessenem Regen nicht gewarnt. Seffern 2016 und Prümzurlay 2018 wurden nur zeitweise gewarnt; die Warnung wurde zurückgenommen, weil die Welle später kam als gerechnet. Eine Haltezeit (Warnung bleibt z. B. 12 h stehen) wäre eine sinnvolle Ergänzung; für die Gewitter selbst bräuchte es Radarniederschlag.
- **Prüm 2, Januar 2021:** Schneeschmelze; ohne Schneemodul nicht vorhersagbar.
- Ehrlich gesagt: Mit ein bis drei Hochwassern je Pegel ist das eine Fallsammlung, keine Statistik. Die Juli-2021-Werte stammen aus Modellen, die ohne 2019–2022 angepasst wurden. Die Form der neuen Bausteine wurde aber mit Blick auf diese Hochwasser gewählt (nächster Abschnitt).

Anzeige: Die Live-Seite zeigt je Pegel den Warnstatus, die Wahrscheinlichkeit für HW 10 in 48 h und ab wann Vorwarnung bzw. Warnung gilt. Eingeklappt steht „Wie hätte diese Warnung bei früheren Hochwassern abgeschnitten?“ mit dieser Rückrechnung je Pegel. In der Karte steht sie unter „Wie verlässlich?“ im Vorhersage-Tab, mit allen drei Varianten und 1989–2003, und kurz im Überblick. Die Diagramme ziehen HW 10 kräftiger als die übrigen Schwellen; Meldehöhe und HW 2 sind blass.

## Starkregen: Abflussmodell überarbeitet, Band je Lage, alles mit perfektem Regen (13.11.2026)
**Anlass.** Bei den Hochwassern über HQ10 lag die Messung weit außerhalb des 80-%-Bands. Die Prüfung gab das recht: In den 13 Pegel-Ereignissen über HQ10 (2004–2022, Ausgaben 48 h vor dem Überschreiten bis zum Scheitel) lag die Messung nur in 48 % (3 h Vorlauf), 38 % (6 h) bzw. 10 % (24 h) der Fälle im Band, mit perfektem Regen in 46, 45 und 35 %; über alle Stunden waren es 79–80 %. Mit dem gemessenen Regen erreichte die Kette 12–24 h vor dem Überschreiten nur 7–39 % des Scheitels und 5–68 % des Volumens, der Scheitel kam 1,5 bis über 20 h zu spät.

**Regen oder Modell?** Der Stationsregen ist plausibel (Juli 2021: 120–136 mm im Gebiet), die gemessenen Abflussbeiwerte lagen bei 0,23–0,66, im Modell bei 0,03–0,34. Nach Trockenheit (September 2013, 60–72 mm) lagen Messung und Modell gleich niedrig (0,03–0,06). Der Abflussbeiwert hing im Modell fast nur an der Gebietsfeuchte zu Fensterbeginn (zwei Parameter an ihren Grenzen), die Regenmenge im Ereignis spielte kaum eine Rolle. Ausnahme Prüm 2, Januar 2021: mehr Abfluss (60 mm) als Regen (32 mm), Schneeschmelze.

**Was nicht half** (`extreme_fit.py`, Modelle je Block ohne den Block, mit perfektem Regen geprüft; `data/extreme/vergleich.csv`). Ereignisregen im Beiwert (acc), Intensitätsanteil (int), schneller Abflussweg (fast), stärkere Gewichtung großer Hochwasser, alles je Pegel angepasst: der Scheitel bei HQ10 blieb bei 18–24 %. Grund: Je Pegel gibt es nur ein bis zwei Extremhochwasser, an den Oberläufen nur 2021. Ohne den Block angepasst, hat ein Oberlauf nie ein Extrem gesehen, und die Anpassung schaltet jede Erweiterung ab. Der Intensitätsanteil blieb auch gemeinsam angepasst bei 0, und ein freier SCS-Anfangsverlust passte sich zu stark an die Blöcke an (außerhalb 27 statt 39 %).

**Was übernommen ist: Sättigungsüberschuss, gemeinsam für alle Gebiete** (`lateral.py`, Baustein „sat“/„satx“; `chain_fc.py`, `REGIONAL_SAT`). Zusätzlich zum bisherigen Abfluss fließt der Anteil (1 − φ) · g(Pe; Sv) des Regens schnell ab (zwei Speicher mit Kf). g ist die Steigung der SCS-Abflussfunktion (Pe − 0,2 Sv)² / (Pe + 0,8 Sv), also 0 bis zum Anfangsverlust und dann wachsend gegen 1. Pe ist der Ereignisregen (Vorregenindex mit 24 h Abklingzeit). Der Rückhalt Sv = Sev · exp(−g_ev · B) wird mit der Gebietsfeuchte B kleiner. Die vier Parameter (Anteil, Sev, g_ev, Kf) sind für alle sieben Regengebiete gemeinsam und werden an allen Hochwasser- und Regenfenstern aller Gebiete angepasst, ohne den Prüfblock; die Grundmodelle bleiben, wie sie sind. So lernt etwa Prüm 2 für 2021 aus Echtershausen 2018 und Seffern 2016. Ergebnis je Block (Schritt 1): Anteil 0,97–1, Sev 165–181 mm, g_ev 0,16–0,24, Kf 2,4–2,6 h (stabil, also keine Zufallsanpassung).

**Zweiter Schritt: nichtlineare Speicher und die Strecke über den Stausee.** Mit dem Sättigungsüberschuss stimmte an den Oberläufen im Juli 2021 das Volumen schon (Prüm 2 und Giesdorf: Beiwert 0,76 im Modell gegen 0,58–0,65 gemessen), aber die Welle war halb so hoch: Der größte Teil lief durch Speicher mit 17–25 h. Dazu kommt ein nichtlinearer Speicher (`lateral.py`, „nlr“/„nlrx“): Die Speicherzeit sinkt, je voller der Speicher, K(S) = K / (1 + (S/K)/oref)^a, die Wasserbilanz geht exakt auf. Die zwei Parameter sind ebenfalls für alle Gebiete gemeinsam. Sie greifen erst bei sehr großen Abflussspenden (oref 4,5 bis 20 mm/h), gewöhnliche Wellen bleiben, wie sie waren. Außerdem dämpfte die Strecke Echtershausen → Wiersdorf über den Stausee große Wellen auch mit gemessenem Zufluss auf die Hälfte (2018: 116 statt 218 m³/s, 2021: 334 statt 436), weil die vielen kleinen Wellen die Wellengeschwindigkeit bei hohem Abfluss bestimmten. Wiersdorf wird jetzt mit absoluten Fehlern angepasst (`chain_fc.py`, `WTS_EXP_KEY`). Geprüft ohne den jeweiligen Block kommen dann 179 statt 116 (2018) und 424 statt 334 m³/s (2021) heraus, und die Güte der Strecke steigt in jedem Block (NSE 0,86–0,90 → 0,93–0,97). Nebenbei fiel auf, dass die Gewichtungseinstellung für Wiersdorf bisher gar nicht griff; das ist behoben.

Prüfung mit perfektem Regen, alle Blöcke, Modelle jeweils ohne den Block:

| | anfangs | Schritt 1 (Sättigungsüberschuss) | Schritt 2 (+ nichtlineare Speicher, Wiersdorf) |
|---|---|---|---|
| HQ10: Scheitel, Ausgabe 24 / 12 / 6 h vor dem Überschreiten (Median) | 18 / 18 / 25 % | 39 / 40 / 46 % | 57 / 56 / 58 % |
| HQ10: Volumen 48 h (Median) | 26 % | 55 % | 67 % |
| HQ10: Zeitfehler des Scheitels | 7 h zu spät | richtig | 1 h zu früh |
| alle Hochwasser, NSE 6–24 h | | gleich oder etwas besser | gleich (± 0,02) |
| Regenlagen ohne Hochwasser: mehr als doppelt so hoch gerechnet | 7,5 % | 7,5 % | 7,8 % |
| Oktober 2026 mit ICON-D2, Prümzurlay mittlerer Fehler 48 h | 3,1 m³/s | 3,0 | 3,0 |

Einzeln, Ausgabe 12 h vor dem Überschreiten, perfekter Regen (Anteil am gemessenen Scheitel):

| Hochwasser | anfangs | Schritt 1 | Schritt 2 | Schritt 2 / gemessen |
|---|---|---|---|---|
| Alsdorf-Oberecken 07/2021 | 22 % | 79 % | 87 % | 208 / 239 m³/s |
| Seffern 07/2021 | 18 % | 74 % | 89 % | 180 / 201 |
| Giesdorf 07/2021 | 32 % | 63 % | 79 % | 25 / 32 |
| Echtershausen 07/2021 | 20 % | 52 % | 65 % | 289 / 445 |
| Wiersdorf 07/2021 | 17 % | 36 % | 61 % | 267 / 436 |
| Prümzurlay 07/2021 | 17 % | 38 % | 56 % | 337 / 596 |
| Prüm 2 07/2021 | 19 % | 40 % | 53 % | 62 / 118 |
| Sinspelt 07/2021 | 7 % | 43 % | 51 % | 59 / 116 |
| Seffern 06/2016 | 23 % | 81 % | 66 % | 41 / 62 |
| Echtershausen 06/2018 | 11 % | 26 % | 28 % | 63 / 222 |
| Wiersdorf 06/2018 | 10 % | 19 % | 22 % | 47 / 218 |
| Prümzurlay 06/2018 | 17 % | 14 % | 14 % | 29 / 209 |
| Prüm 2 01/2021 (Schnee) | 39 % | 39 % | 41 % | 14 / 33 |

Was bleibt: Das Gewitter vom Juni 2018 (Echtershausen, und damit Wiersdorf und Prümzurlay) gibt das Modell nicht wieder. Mit 75 mm im Gebietsmittel und 50 mm in 3 h war der Regen in den Stationsdaten zwar da, aber ein Zwischengebiet von 273 km² mit gleichmäßigem Regen bildet eine örtliche Gewitterzelle nicht ab. Dafür bräuchte es Radarniederschlag (RADOLAN/RADKLIM des DWD). Sinspelt und Prüm 2 bleiben 2021 bei gut der Hälfte. Der Januar 2021 in Prüm 2 war Schneeschmelze. Die Parameter des nichtlinearen Speichers schwanken zwischen den Blöcken stärker als die des Sättigungsüberschusses (a 0,7 bis 2), weil nur die Extremhochwasser sie bestimmen; die Modellform ist mit Blick auf diese wenigen Lagen gewählt. „Modellierungsmäßig sicher“ ist die Kette damit noch nicht, aber im Juli 2021 mit richtigem Regen in der richtigen Größenordnung.

**Band je Lage: Starkregenlage** (`uncert.py`). Eine Fehlerstatistik nur aus Hochwassern über HQ10 ist nicht möglich: Zur Ausgabezeit weiß niemand, ob es eins wird, und ein so gebautes Band sähe nur in der Rückschau gut aus. Bekannt ist dagegen der Regen. Starkregenlage heißt: mindestens 40 mm in 24 h oder 20 mm in 3 h im Gebiet, ohne Regenvorhersage in den 24 h bis zur Ausgabezeit, mit Regenvorhersage von 24 h davor bis 24 h danach (im Archiv der gemessene Regen, live je ICON-Lauf). 2004–2022 gab es 24 solche Lagen, 3 brachten HQ10. Ihre Fehler bilden eine eigene Tabelle, über alle Pegel zusammengefasst (relative Fehler) und je Block ohne den Block gerechnet: 3.063 Fälle ohne, 5.271 mit Regenvorhersage. Live rechnet jeder Ensemble-Lauf mit seiner eigenen Lage; die Seite zeigt „Starkregenlage in x von 21 Läufen“.

Anteil der Messwerte im 80-%-Band (kreuzvalidiert, perfekter Regen, Stand Schritt 2):

| | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| alle Fälle | 79 % | 78 % | 78 % | 77 % | 78 % |
| Starkregenlagen | 77 % | 76 % | 76 % | 76 % | 75 % |
| Hochwasser über HQ10 (nachträglich ausgewählt), anfangs | 50 % | 46 % | 45 % | 40 % | 35 % |
| Hochwasser über HQ10, jetzt | 65 % | 60 % | 58 % | 52 % | 53 % |
| davon in einer Starkregenlage | 71 % | 70 % | 67 % | 61 % | 61 % |

Die Vorhersage selbst liegt bei den HQ10-Hochwassern jetzt viel näher an der Messung (siehe oben). Das Band trifft sie trotzdem nur wenig öfter, weil es relativ zur Vorhersage gebaut ist und die Fehler bei Extremen groß bleiben. Ein Band, das nur aus HQ10-Fällen gelernt hätte, wäre live nicht anwendbar. Ohne Regenvorhersage bleibt das Band bei HQ10 schwach (43 % bei 6 h, 17 % bei 24 h), weil der kommende Regen unbekannt ist.

**Alles mit perfektem Regen.** Das Archiv rechnet jede Stunde 2004–2022 auch mit dem später gemessenen Regen als Vorhersage (`kette_vorhersage`), mit eigenen Fehlertabellen je Block; Warnungen und Band oben sind damit geprüft. Vor 2004 gibt es keinen Stundenregen. In der Karte schaltet „perfekter Regen“ im Vorhersage-Tab jedes Hochwasser mit Regendaten auf diese Variante um, mit den passenden Tabellen (Beispiel Prümzurlay 14.07.2021 15:30, 15 h vor dem Scheitel: mit perfektem Regen Warnung, HW 10 in 48 h über 95 %, Scheitel mindestens 558 cm, 80 %: 460–673 cm, nach Schritt 1 waren es 446 cm; gemessen 698 cm; mit Regen nur bis zur Ausgabezeit Vorwarnung).

Dateien: `extreme_fit.py` (Prüfgerüst), `data/extreme/*.json` (je Variante Modelle und Prüfwerte), `data/extreme/vergleich.csv`; `data/chain_models.json` enthält unter `sat` die gemeinsamen Parameter je Satz. Die Karte ist durch die Tabellen mit perfektem Regen größer geworden (7,6 MB).

Nächste Schritte: Radarniederschlag (RADOLAN/RADKLIM) für die Gewitter (2016, 2018) und für die Oberläufe; Haltezeit für Warnungen; Schneemodul (Temperatur DWD) für Winterhochwasser.


## Scheitel-Regel nur noch ohne Regenvorhersage (14.11.2026)

Die Regel „solange an den Oberliegern kein Scheitel gemessen ist, fällt die Vorhersage nach ihrem Höchstwert nicht“ gilt nur noch für die Rechnung
ohne Regenvorhersage (`kette`, `kette_regen`). Mit Regen als Vorhersage (`kette_vorhersage`, Live-Betrieb) ist sie entfallen.

Prüfung (`hold_test.py`, `data/hold_test.csv`): alle Hochwasser 2004–2022 an den fünf Pegeln mit Oberliegern (113 bis 162 je Pegel), Modelle je Block
ohne den Block, Abklingzeit der Regel 0 (aus), 6, 12, 24, 48 h und unendlich (streng). NSE mit dem gemessenen Regen als Vorhersage:

| Pegel | 24 h streng | 24 h aus | 48 h streng | 48 h aus |
|---|---|---|---|---|
| Echtershausen | 0,28 | 0,28 | −0,46 | −0,15 |
| Wiersdorf | 0,21 | 0,34 | −0,97 | −0,05 |
| Prümzurlay | 0,08 | 0,25 | −0,99 | 0,02 |
| Seffern | 0,18 | 0,33 | −0,70 | 0,09 |
| Alsdorf-Oberecken | 0,03 | 0,30 | −1,07 | 0,14 |

Die Scheitelhöhe ändert sich nicht (Median gleich, 90-%-Wert mit Regel bis 0,03 höher); der Scheitel kam mit Regel an Wiersdorf und Alsdorf-Oberecken
im Median 2 h zu spät. Abklingzeiten bis 24 h sind gleich gut wie „aus“, eine weiche Form lohnt nicht. Ohne Regenvorhersage hilft die Regel
(24 h: Echtershausen −1,55 statt −1,95), dort bleibt sie.

Unsicherheitstabellen und Warnstatistik für `kette_vorhersage` sind neu gerechnet (`uncert.py archive --variants kette_vorhersage`, dann `tables`).
Die Warnungen bei den 13 HQ10-Hochwassern sind praktisch gleich geblieben; Phasen ohne Hochwasser: Vorwarnung 41 statt 46, Warnung 3 statt 2.

## Neuronales Netz im Starkregen-Modellsatz, ICON-RUC, Scheitel-Kachel (15.11.2026)

**Netz.** `ml_fit.py` passt ein kleines Netz (11 Eingänge, 2 × 16 Knoten, gemeinsam für alle Gebiete) an, das im Regen-Abfluss-Teil den Abflussbeiwert und den schnellen Anteil verschiebt. Eingänge sind Vorfeuchte, Basisabfluss, Regen und Regen über 4, 10 und 20 mm/h, alles aus dem Stationsregen. Radar (RADKLIM, `radar.py`) wurde geprüft und bringt als Eingang keinen Gewinn; der Betrieb braucht kein Radar. Das Netz beginnt beim bisherigen Modell (letzte Schicht null) und wird am besten Stand auf einem zurückgehaltenen Zeitblock gestoppt.

Vier Stände wurden kreuzvalidiert verglichen (Scheitel Vorhersage/Messung im Median, gemessener Regen als Vorhersage):

| Stand | HQ10-Hochwasser (13), 24 / 12 / 6 h vorher | Starkregen-Hochwasser (148), 24 / 12 / 6 h | übrige Hochwasser (1022), 12 h | Regen ohne Hochwasser (319): mehr als 1,5-fach überschätzt |
|---|---|---|---|---|
| ohne Netz | 0,57 / 0,56 / 0,58 | 0,65 / 0,67 / 0,71 | 0,88 | 15 % |
| Netz in allen Modellsätzen | 0,65 / 0,73 / 0,74 | 0,48 / 0,53 / 0,62 | 0,88 | 12 % |
| Netz von null nur an Starkregen-Hochwassern trainiert | wie ohne Netz | 0,52 / 0,55 / 0,67 (mit Netz im Wintersatz) | 0,88 | |
| **eigenes Starkregen-Netz (übernommen)** | 0,60 / 0,66 / 0,70 | 0,65 / 0,66 / 0,70 | 0,88 | 15 % |

Übernommen ist: Gesamtmodell und Wintersatz ohne Netz; der Starkregensatz bekommt ein eigenes Netz, das vom Netz des Gesamtmodells startet und nur an den Starkregen-Hochwassern auf dem Starkregensatz nachtrainiert wird (150 Schritte; der Fehler auf dem zurückgehaltenen Block fiel am Ende noch, länger wurde nicht getestet). Gründe:

- Das Netz im Wintersatz ändert die Winterhochwasser nicht, drückt aber Starkregen-Hochwasser, solange die Kette noch nicht umgeschaltet hat (sie schaltet erst, wenn der starke Regen gefallen ist).
- Früher umschalten (schon bei vorhergesagtem Starkregen) ist schlechter, auch ohne Netz (12 h vorher 0,59 statt 0,67).
- Die Starkregen-Hochwasser allein reichen nicht, um ein Netz von null zu trainieren.
- Was entfällt: die geringere Überschätzung bei Regen ohne Hochwasser. Die gab es nur mit Netz im Wintersatz.

Einzelne Pegel bei den Starkregen-Hochwassern, 12 h vorher: Sinspelt 0,83 (ohne Netz 0,91, 11 Ereignisse), Wiersdorf 0,83 (0,87), alle anderen innerhalb 0,01. Warnstatistik und Band sind neu gerechnet und praktisch unverändert (HQ10-Ereignisse mit perfektem Regen: Vorwarnzeiten gleich bis auf Prümzurlay 06/2018, 0 statt 1 h).

Die Rechnung im Browser (`forecast_core.js`) kennt das Netz; `test_forecast_js.py` vergleicht sie mit Python. Im Betrieb wird kein JAX gebraucht, nur für die Anpassung (`requirements-train.txt`). `PRUEM_BASE=<chain_models.json ohne Netz> python chain_fc.py run` übernimmt die Grundmodelle eines früheren Laufs und rechnet nur das Netz neu.

**ICON-RUC.** `live.py` holt zusätzlich ICON-D2-RUC (stündliche Läufe, kurze Reichweite) und zeichnet eine eigene gestrichelte Linie: RUC-Regen, soweit er reicht, danach ICON-D2. Die Linie geht nicht ins Band und nicht in die Warnwahrscheinlichkeit ein, der Regen bleibt ohne Angleichfaktor, weil es dafür noch keine Prüfung gibt. Der Abrufname bei Open-Meteo ist ungeprüft; `live.py` probiert drei Namen und schreibt ins Protokoll, welcher geklappt hat (`ICON-RUC: …`). Klappt keiner, fehlt nur diese Linie.

**Scheitel-Kachel.** Die Kachel „Scheitel erwartet“ nannte den höchsten Wert der 48 h auch dann Scheitel, wenn er am Ende des Zeitraums lag, die Vorhersage dort also noch stieg. Jetzt heißt sie in diesem Fall „Höchststand in 48 h“ mit dem Hinweis, dass der Scheitel erst danach kommt, und nennt den Scheitel aus dem 7-Tage-Ausblick, wenn dort einer liegt.

## Starkregen: Fehlerzerlegung, Scheitelfehler in der Anpassung, Regenschwellen, Nachbarschaft, Prüf-Seite (15.11.2026)

**Fehlerzerlegung** (`storm_diag.py`, `data/storm_diag.csv`): 148 Starkregen-Hochwasser, kreuzvalidiert, gemessener Regen als Vorhersage, Ausgabe 24 h vor dem Scheitel.

- Das Volumen stimmt ungefähr (Median 0,80 der Messung), der Scheitel nicht (0,65), und er kommt im Median 3 h zu spät: Die Kette reagiert bei Starkregen zu langsam und zu flach.
- Am klarsten an den Oberläufen und bei trockenem Boden: Giesdorf Volumen 0,94, Scheitel 0,48; bei niedrigem Ausgangsabfluss (35 Ereignisse) Volumen 0,92, Scheitel 0,37.
- Im größten Viertel je Pegel (39 Ereignisse) fehlt zusätzlich Volumen: gemessen flossen 33 % des Regens ab, im Modell 18 %.
- Der Regen ist selten das Problem: Radar (RADKLIM) liegt im Median bei 0,83 des Stationsregens, das Verhältnis erklärt die Fehler kaum. Nur bei 6 Ereignissen zeigt das Radar deutlich mehr als die Stationen (dort Scheitel 0,44).

**Scheitelfehler in der Anpassung des Starkregensatzes** (`ml_fit.py`: `peak_err`, `STORM_PEAK_W = 0.2`, `STORM_FAST`). Die Abweichung je Zeitschritt bestraft eine steile Welle doppelt, sobald sie eine Stunde daneben liegt, und zieht die Anpassung zu flachen, späten Wellen. Beim Nachtraining des Starkregen-Netzes zählt deshalb zusätzlich der Fehler des Höchstwerts je Fenster; die Speicherzeit des schnellen Wegs ist je Gebiet frei (bis 15 min).

| Stand | HQ10 (13), 24 / 12 / 6 h vorher | Starkregen-Hochwasser (148) | übrige (1022), 12 h | Regen ohne Hochwasser (319): mehr als 1,5-fach überschätzt |
|---|---|---|---|---|
| eigenes Starkregen-Netz (vorher) | 0,60 / 0,66 / 0,70 | 0,65 / 0,66 / 0,70 | 0,88 | 14,7 % |
| nur freie Speicherzeiten | 0,61 / 0,66 / 0,69 | 0,65 / 0,66 / 0,70 | 0,88 | 14,7 % |
| nur Scheitelfehler (Gewicht 0,2) | 0,77 / 0,76 / 0,80 | 0,66 / 0,68 / 0,74 | 0,88 | 16,0 % |
| **beides (übernommen)** | 0,79 / 0,77 / 0,82 | 0,66 / 0,68 / 0,74 | 0,88 | 16,0 % |
| beides, Gewicht 0,5 | 0,83 / 0,82 / 0,84 | 0,67 / 0,69 / 0,75 | 0,88 | 17,2 % |

Schneller ist der Weg dabei nicht geworden (Speicherzeiten 3,7 → rund 3 h); der Gewinn kommt daraus, dass das Netz bei starkem Regen mehr Wasser auf den schnellen Weg gibt. Die Oberläufe 12 bis 24 h vorher bleiben schwach (Giesdorf 0,53, Seffern 0,55): Dort rechnet die Kette noch mit dem Wintersatz, weil der starke Regen noch nicht gefallen ist. Früher umschalten (schon bei vorhergesagtem Starkregen) ist auch mit dem neuen Starkregensatz schlechter (Starkregen-Hochwasser 12 h vorher 0,65 statt 0,68) und bleibt verworfen.

Im regulären Ablauf (Kette, Archiv, Tabellen neu gerechnet) ändert sich für Band und Warnung wenig, obwohl die Scheitel höher liegen, weil die Fehlerstatistik den alten Rückstand schon ausglich: Mit gemessenem Regen als Vorhersage 35 statt 39 Vorwarnphasen ohne HQ10 danach und 5 statt 3 solche Warnphasen; Messung im 80-%-Band bei den HQ10-Hochwassern 54 % statt 58 % (6 h) und 52 % statt 53 % (24 h); alle Fälle unverändert 78 %. Die Vorwarnzeiten der 13 HQ10-Ereignisse sind gleich, bis auf Wiersdorf im Juli 2021: Dort fällt die Wahrscheinlichkeit eine Stunde vor dem Überschreiten kurz auf 15 % (der Stausee hält die Welle zurück, die Fehlerkorrektur zieht die Vorhersage nach unten), die Auswertung zählt die Warnung deshalb nicht mehr als ununterbrochen. Eine Haltezeit für Warnungen würde das abfangen.

**Regenschwellen** (`rain_thr.py`, Live-Seite „Wie viel Regen bis zur Warnung?“). Für 3, 6, 12 und 24 h: die Regenmenge, gleichmäßig im Einzugsgebiet ab jetzt, bei der die Kette beim jetzigen Zustand HW 10 mit 10 % (Vorwarnung) bzw. 50 % (Warnung) erreicht; daneben die größte vorhergesagte Summe gleicher Dauer. Prüfung (`python rain_thr.py check`, `data/rain_thr_check.csv`): 861 Ausgabezeiten 2004–2022, kreuzvalidiert, Schwelle gegen den in den 48 h danach gemessenen Regen.

| Fälle | Anzahl | Regen über Vorwarnschwelle | über Warnschwelle |
|---|---|---|---|
| HW 10 nicht erreicht (Winter- und Starkregen-Hochwasser, Regenlagen ohne Hochwasser) | 810 | 1 | 0 |
| HW 10 erreicht, alle Ausgabezeiten | 51 | 41 | 33 |

Je HQ10-Ereignis 24 h vorher: Vorwarnschwelle bei 10 von 13 überschritten, Warnschwelle bei 8 von 13 (Juli 2021 an allen acht Pegeln; Juni 2018 in Echtershausen und Wiersdorf knapp über der Vorwarnschwelle, Prümzurlay darunter; Seffern 2016 erst 12 h vorher; Prüm 2 Januar 2021 Schneeschmelze). Die Vergleichsfälle sind eine Auswahl, nicht jede Stunde, und geprüft ist gegen den gemessenen Regen. Die Schwellen hängen kaum von der Dauer ab, weil die Kette auf Intensität zu träge reagiert. Gerechnet mit dem Stand vor dem Scheitelfehler in der Anpassung.

**Nachbarschafts-Ensemble** (`live.py`). ICON-D2 zusätzlich an acht um 30 km versetzten Orten; die Seite zeigt die Spanne von Regen und Höchststand und in wie vielen Richtungen Vorwarnung oder Warnung erreicht würde. Nicht im Band und nicht in der Warnung. Der Abruf ist von der Entwicklungsumgebung aus nicht testbar; im Protokoll steht `Nachbarschaft: n von 8 Richtungen`.

**Prüf-Seite** (`eval_view.py`, `eval_template.html` → `eval.html`). Vorhersagen an 76 Hochwassern 1989–2022 nachrechnen (40 mit Stundenregen): Ereignisliste mit Filter, je Pegel die Vorhersagen im Verlauf (Fächer), Kennzahlen (Scheitelhöhe 24/12/6 h vorher, Vorwarnzeit vor HW 10 oder Fehlalarm, Messung im 80-%-Band), Tabelle der Entwicklung je Ausgabezeit und eine Übersicht aller Ereignisse und Pegel. Gerechnet wird im Browser mit `forecast_core.js`, jeweils mit dem Modellsatz ohne den Zeitblock des Ereignisses. Die Seite braucht zum Bauen die LfU-Pakete und enthält gemessene Ganglinien; sie liegt deshalb nicht im Repository, sondern wird über den Ordner `seiten/` in den Container gegeben (siehe `seiten/README.md`).

## Niveaufaktor für Prümzurlay und Alsdorf-Oberecken (15.11.2026)

Die Abflussreihen beruhen auf der jeweils gültigen Beziehung zwischen Wasserstand und Abfluss. In Prümzurlay hat das Hochwasser vom Juli 2021 das Gerinne unterhalb des Pegels freigespült; seit Januar 2022 gilt eine neue Beziehung (HQ10 = 199 m³/s: bis 2021 bei 442 cm, heute bei 399 cm). Die Kennwerte HQ2 bis HQ100 der HVZ passen zur historischen Abflussreihe (Überschreitungen in 32 Jahren: 12 / 4 / 1 für HQ2 / HQ10 / HQ20, erwartet 16 / 3 / 2); alte Wasserstände mit der heutigen Kurve umzurechnen wäre falsch (dann käme in Prümzurlay das 1,7- bis 1,9-Fache der Oberlieger an). Die Prüf-Seite rechnet deshalb je Ereignis mit der damals gültigen Beziehung.

Unabhängig davon zeigt Prümzurlay seit 2022 bei gleichem Zufluss aus Wiersdorf und Sinspelt 20 bis 30 % mehr Abfluss als in den Jahren davor; an der Nims gab es mehrere solche Wechsel. Die Kette ist fast nur an den Jahren bis 2021 angepasst. `niveau.py` misst das Verhältnis Messung / reine Simulation (gemessene Oberlieger, gemessener Regen, ohne Fehlerkorrektur, ab 1,5 MQ) je Jahr (`data/niveau_jahre.csv`):

| | 2008–2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|
| Prümzurlay | 0,95–1,04 | 0,93 | 0,95 | 0,93 | 1,16 |
| Alsdorf-Oberecken | 0,99–1,24 | 1,38 | 1,17 | 0,79 | 0,72 |
| Seffern | 0,95–1,20 | 1,02 | 1,23 | 1,30 | 1,43 |

Der Betrieb multipliziert die Simulation des Pegels vor der Fehlerkorrektur mit einem Faktor (`data/niveau.json`, `chain_fc.LEVEL`, in `forecast_core.js` `fl.niveau`): Prümzurlay 1,16 (seit Januar 2022, Bereich ab 4 MQ), Alsdorf-Oberecken 0,76 (seit April 2022; Seffern lag von Juli 2021 bis März 2022 vorübergehend höher). Seffern bekommt keinen Faktor: seit April 2022 bei kleinem Abfluss 0,99, ab 4 MQ 1,31 aus einem einzigen Ereignis.

Prüfung (`niveau_check.py`, Median Vorhersage/Messung, Modellsatz ohne 2019–2022):

| Pegel, Variante | Bezugsjahre 2012–2018 | heute ohne Faktor | heute mit Faktor |
|---|---|---|---|
| Prümzurlay, ohne Regen, 6 / 12 / 24 h | 1,02–1,03 / 1,03–1,05 / 1,07–1,10 | 0,97 / 0,96 / 0,97 | 1,01 / 1,03 / 1,07 |
| Prümzurlay, gemessener Regen | 1,00 / 0,98 / 0,92–0,94 | 0,96 / 0,92 / 0,85 | 0,99 / 0,96 / 0,90 |
| Alsdorf-Oberecken, ohne Regen | 1,00–1,01 / 1,02–1,04 / 1,05–1,06 | 1,07 / 1,11 / 1,11 | 0,98 / 1,00 / 0,97 |
| Alsdorf-Oberecken, gemessener Regen | 0,97 / 0,94 / 0,88–0,90 | 1,02 / 1,00 / 0,97 | 0,97 / 0,94 / 0,90 |

Mit Faktor liegt die Vorhersage wieder dort, wo sie in den Bezugsjahren lag, und die Fehlerstatistik für Band und Warnung passt wieder. Grenzen: 2022 war trocken (Prümzurlay 529 Werte, Alsdorf-Oberecken 118, kaum Hochwasser); ob die neue Kurve bei großem Abfluss trägt, ist offen; der Faktor ist fest und wird noch nicht aus den Live-Daten nachgeführt.

## Radarregen 2001–2003 für die Zeit vor den Stationsdaten (15.11.2026)

RADKLIM (RW 2017.002) liegt jetzt von Januar 2001 bis Ende 2025 als Ausschnitt vor (`radar_crop_browser.js`; 2023–2025 nützen erst mit Pegeldaten dieser Jahre). Vor 2004 gibt es keinen Stundenregen der Stationen; `rain_fit.rain_frames` setzt dort das Radar-Gebietsmittel ein, angeglichen mit dem Verhältnis Station / Radar des Gebiets aus den gemeinsamen Stunden ab 2004 (Faktor 1,00 für Prüm 2 bis 1,13 für die Nims), Lücken bis 6 h linear gefüllt (`RADAR_FILL`, abschaltbar mit `PRUEM_RADAR_FILL=0`). Ab 2004 bleibt der Stationsregen unverändert; die Modelle sind nicht neu angepasst.

Damit lassen sich neun Hochwasser 2001–2003 mit Regen nachrechnen, mit dem Modell aus 2004–2022, also unabhängig. Januar 2003, gemessener (Radar-)Regen als Vorhersage, Höchstwert der Kette im Verhältnis zum gemessenen Scheitel 24 / 12 / 6 h vorher (ohne Abgleich mit der Fehlerstatistik):

| Pegel | ohne Regen | mit Regen |
|---|---|---|
| Prüm 2 | 0,24 / 0,29 / 0,58 | 0,86 / 0,79 / 0,76 |
| Sinspelt | 0,22 / 0,45 / 0,61 | 0,82 / 0,86 / 0,82 |
| Echtershausen | 0,26 / 0,35 / 0,54 | 0,99 / 0,73 / 0,55 |
| Prümzurlay | 0,41 / 0,74 / 0,94 | 1,15 / 0,75 / 0,86 |
| Giesdorf | 0,22 / 0,36 / 0,77 | 1,44 / 1,33 / 1,18 |
| Seffern | 0,44 / 0,34 / 0,53 | 1,55 / 1,29 / 0,84 |
| Alsdorf-Oberecken | 0,60 / 0,71 / 0,89 | 1,65 / 1,17 / 0,64 |

An der Prüm und der Enz trifft die Kette das Hochwasser 2003 einen Tag vorher auf 15 bis 20 %; an der Nims liegt sie einen Tag vorher 45 bis 65 % zu hoch. Wiersdorf ist bei diesem Hochwasser bei 250 cm abgeschnitten und nicht bewertbar.

## Versuch: flächiger Regen auf dem Radarraster (15.11.2026, nicht übernommen)

Bisher ist der Gebietsregen der abstandsgewichtete Stationswert am Gebietsmittelpunkt. `grid_rain.py` baut zwei Flächenmittel auf dem 1-km-Raster: (A) Stationen interpoliert mit Höhenabhängigkeit (rund 8 % mehr Regen je 100 m), (B) Menge von den Stationen, Muster vom Radar (je Stunde der Faktor Station / Radar an den Stationsorten, interpoliert, auf das Radarfeld angewendet).

- **A** ist praktisch derselbe Regen wie bisher (Jahressummen +0 bis +6 %, stündliche Korrelation 0,97 bis 1,00, Ereignissummen der Starkregen-Hochwasser im Median +2 %); nicht weiterverfolgt.
- **B** ordnet die 148 Starkregen-Hochwasser besser (Rangkorrelation Regensumme – abgeflossenes Volumen 0,69 statt 0,61, größte 3-h-Summe – Scheitel 0,30 statt 0,20). Mit B neu angepasst und kreuzvalidiert (`PRUEM_RAIN=merge`) wird die Kette aber nicht besser:

| gemessener Regen als Vorhersage | Stationsregen (Betrieb) | Stationen + Radarmuster |
|---|---|---|
| HQ10-Scheitel 24 / 12 / 6 h vorher (Median) | 0,79 / 0,77 / 0,82 | 0,67 / 0,62 / 0,70 |
| Starkregen-Hochwasser, Scheitel 24 / 12 / 6 h (Median über die Pegel) | 0,72 / 0,77 / 0,83 | 0,65 / 0,69 / 0,83 |
| Starkregen-Hochwasser, NSE bei 12 h (Mittel über die Pegel) | 0,62 | 0,66 |
| alle Hochwasser, Scheitel 12 h | 0,86 | 0,87 |

Der Verlauf passt bei Starkregen etwas besser (NSE), die Scheitel liegen aber tiefer, am deutlichsten bei den HQ10-Hochwassern. Vermutlich unterschätzt das Radar gerade die stärksten Regen (Dämpfung), und das Muster trägt diese Unterschätzung in das kombinierte Feld. Der Betrieb bleibt beim Stationsregen.

## Auswertung des Live-Betriebs und Haltezeit (16.11.2026)

`live_eval.py` liest alle archivierten Ausgaben (`gui/archiv/live_*.html`, jede enthält ihre Daten) und stellt sie gegen die inzwischen gemessenen Wasserstände: Fehler nach Vorlauf (3 bis 48 h) für die Kette, den ICON-D2-Hauptlauf, ICON-RUC, die Vorhersage der HVZ und „bleibt wie jetzt“, Anteil der Messungen im 80-%-Band, je Ausgabe vorhergesagter und gemessener Höchststand und Warnstatus. Der Container baut die Seite nach jedem Lauf neu (`auswertung.html`). Geprüft nur mechanisch an vervielfältigten Testausgaben; echte Zahlen gibt es erst aus dem laufenden Betrieb.

Haltezeit: Eine Vorwarnung oder Warnung bleibt 12 h stehen, auch wenn die Wahrscheinlichkeit zwischendurch unter die Schwelle fällt (`live.HOLD_H`, Zustand in `data/live/warnstatus.json`). Die Trefferstatistik der Warnungen ist weiterhin ohne Haltezeit gerechnet.

## Stundenregen aus Tagessummen und ERA5 für die Jahre vor 2004 (16.11.2026)

`day_rain.py` baut für 1988–2022 einen Stundenregen je Teilgebiet: die Menge aus den Tageswerten von 24 DWD-Stationen im und um das Gebiet (abstandsgewichtet am Gebietsmittelpunkt), der Stundenverlauf aus der Reanalyse ERA5 (Open-Meteo-Archiv, 0,25°). Die Zuordnung der Tageswerte (Datum D = 07 Uhr MEZ an D bis 07 Uhr am Folgetag) ist an den Stundenstationen nachgeprüft (Korrelation 1,00). Gegen den gemessenen Stundenregen 2004–2022: Summen +0,5 bis +4 %, Korrelation stündlich 0,51–0,58, über 6 h 0,82–0,85, über 24 h 0,95–0,96; über 3 h im Winter 0,84–0,87, im Sommer 0,63–0,68.

Prüfung an der Kette (`day_rain_check.py`, kreuzvalidierte Modelle unverändert, einmal mit gemessenem Stundenregen, einmal mit dem Ersatzregen als bekannte Vorhersage):

| | gemessener Stundenregen | Tagessummen + ERA5 |
|---|---|---|
| Winterhochwasser (1022), Scheitel 24 / 12 / 6 h vorher | 0,80 / 0,88 / 0,94 | 0,80 / 0,87 / 0,94 |
| … innerhalb ±20 %, 12 h vorher | 64 % | 63 % |
| … Fehler der Scheitelzeit (Median des Betrags), 12 h vorher | 2,25 h | 2,75 h |
| Starkregen-Hochwasser (148), Scheitel 24 / 12 / 6 h vorher | 0,66 / 0,68 / 0,74 | 0,60 / 0,66 / 0,74 |
| … Fehler der Scheitelzeit, 12 h vorher | 2,6 h | 4,5 h |

Für Winterhochwasser ersetzt dieser Regen den Stundenregen fast ohne Verlust, für Starkregen nicht. Damit lassen sich die Winterhochwasser vor 2004 mit Regen rechnen (`PRUEM_RAIN=tag`). Mit dem Modell aus 2004–2022 und dem Ersatzregen als Vorhersage, Höchstwert der Kette im Verhältnis zum gemessenen Scheitel 24 h vorher:

| Pegel | Dez. 1991 | Jan. 1993 | Dez. 1993 | Jan. 1995 | Jan. 2003 |
|---|---|---|---|---|---|
| Prüm 2 | 0,26 | 0,62 | 0,73 | 0,68 | 0,77 |
| Echtershausen | 0,38 | 0,53 | 0,91 | 0,72 | 0,78 |
| Wiersdorf | 0,27 | 0,47 | 0,90 | 0,64 | 1,08 |
| Sinspelt | 0,27 | 0,20 | 0,72 | 0,42 | 0,66 |
| Prümzurlay | 0,39 | 0,48 | 0,96 | 0,75 | 0,98 |
| Giesdorf | 0,36 | 0,69 | 1,07 | 0,64 | 0,91 |
| Seffern | 0,26 | 0,40 | 1,70 | 0,74 | 1,05 |
| Alsdorf-Oberecken | 0,40 | 0,47 | 1,40 | 0,74 | 1,30 |

Dezember 1993 und Januar 2003 trifft die Kette an der Prüm gut; an der Nims liegt sie zu hoch (2003 mit Radarregen noch höher: Giesdorf 1,44, Seffern 1,55, Alsdorf-Oberecken 1,65). Dezember 1991 und Januar 1995 liegen weit zu niedrig; beide hatten Schneeschmelze (Schneehöhen in den Tageswerten: 20 bzw. 18 cm vor dem Scheitel), die das Modell nicht kennt. Januar 1993 liegt ebenfalls zu niedrig, ohne Schnee; die Ursache ist offen. Wiersdorf ist in diesen Hochwassern teils abgeschnitten oder vom Stausee geregelt. Die Daten liegen nicht im Repository (`data/tag/`, mit dem Browser geladen).

## Kommunale Pegel und Seestand Stausee Bitburg (17.11.2026)

Seit einigen Jahren betreiben Gemeinden im Eifelkreis eigene Wasserstandsmessstellen; sie erscheinen auf hochwasser.rlp.de (Typ „municipal“ in `/api/v1/config`) und liefern über dieselbe Schnittstelle wie die Landespegel die Wasserstände der letzten 5 Tage im 15-Minuten-Takt, ohne Abfluss und ohne Archiv. In unseren Einzugsgebieten sind es 20 Messstellen: an der Prüm Hermespand, Niederprüm, Pittenbach, Kinzenburg, Mauel, Merkeshausen, Brecht, Bettingen und Holsthum; Niedermehlen am Mehlenbach, Habscheid am Alfbach; an der Enz Neurath/Arzfeld, Neuerburg, Sinspelt Gemeindehaus, Enzen und Holsthum Schloss; an der Nims Rommersheim, Nimsreuland, Rittersdorf und Dockendorf. Dazu kommt der Seestand des Stausees Bitburg (Landesmessstelle 26280366).

`live.py` ruft sie bei jedem Lauf ab (`fetch_kommunal`), legt sie in `hvz.json` des Rohordners ab (damit entsteht das Archiv, das es sonst nicht gibt) und zeigt auf der Live-Seite einen Abschnitt „Kommunale Pegel und Stausee Bitburg“: Wasserstand, Änderung in 1, 3 und 24 h, Höchstwert der 5 Tage, Verlauf der letzten 72 h; Anstiege ab 5 cm/h oder 10 cm/3 h sind hervorgehoben. In die Vorhersage gehen sie noch nicht ein.

Die Messstellen sind ungeprüfte Rohdaten der Betreiber und weniger ausfallsicher als die Landespegel. Deshalb ist alles daran rein zusätzlich: Die Landespegel werden zuerst gesichert; jede Messstelle wird einmal mit 10 s Zeitlimit abgefragt, nach 90 s insgesamt wird abgebrochen; Netzfehler, kaputtes JSON, fehlende oder unsinnige Werte und veraltete Daten (älter als 3 h) lassen nur die betroffene Messstelle weg; kommt nichts Brauchbares an, fehlt der Abschnitt, und ein Fehler beim Zeichnen blendet ihn aus. Geprüft mit vier Fällen (keine Daten, leere Antwort, kaputte Daten, gültige Daten): Seite und Vorhersage entstehen in allen Fällen gleich, ohne Fehler im Browser.

### Lage der kommunalen Messstellen: Rückstau an Mündungen

`kommunal_lage.py` setzt jede Messstelle ins Gewässernetz aus dem Höhenmodell (wie `catchments.py`) und misst flussab den Abstand zur nächsten Mündung eines Zuflusses mit mindestens 30 % der eigenen Fläche bzw. zur Mündung in einen größeren Fluss, dazu das Gefälle auf 2 km und als grobes Maß der Rückstaulänge 2 m Wassertiefe durch das Gefälle (Ergebnis `data/kommunal_lage.csv`). Rückstau ist nur an zwei Messstellen ein Thema:

- **Holsthum Schloss (Enz):** wenige Meter vor der Mündung in die Prüm (418 km² gegen 148 km² Enz). Bei Hochwasser zeigt der Pegel den Prümstand. In der Oktoberwelle 2026 folgte er bis zum Mittag des 25.10. der Enz und sprang dann mit der Prümwelle von 55 auf 69 cm; sein Scheitel (82 cm, 26.10. 02 Uhr) fiel mit dem der Prüm zusammen, nicht mit dem der Enz (25.10. gegen 13 Uhr). Für die Enz taugt er nicht; dafür Enzen und Sinspelt.
- **Holsthum Prüm:** rund 100 m oberhalb der Enzmündung. Die Enz ist kleiner, Rückstau ist möglich, wenn die Enz viel früher und stärker steigt als die Prüm (Gewitter über der Enz). Im Oktober nicht zu sehen: Während die Enz stieg, blieb Holsthum Prüm bei 33 cm und sprang erst mit der Prümwelle.

Alle anderen liegen mindestens 2,5 km oberhalb der nächsten nennenswerten Mündung, bei Rückstaulängen von grob 200 bis 600 m. Niedermehlen (Mehlenbach) liegt 3,3 km, Habscheid (Alfbach) 6,6 km und Enzen 7,3 km vor der Mündung. Örtlich können Brücken stauen (Bettingen, Neurath/Arzfeld und Nimsreuland sind Pegellatten an Brücken); das verfälscht die Höhe eher als den Zeitpunkt. Die Nims-Messstellen haben nur Ortskoordinaten, ihre Lage ist unsicherer.

### Faktor auf die Wellengeschwindigkeit (vorbereitet, derzeit 1)

Die Laufzeit im Routing ist L / c(Q) je Strecke. Eine physikalische Wellenlaufzeit hat nur Wiersdorf → Prümzurlay (24,3 km, 20 Speicher, c 1,4–1,5 m/s, 4,6–4,8 h); bei Prüm 2 → Echtershausen, Giesdorf → Seffern und Seffern → Alsdorf-Oberecken (1–2 Speicher, Verstärkung bis 5) ist c eher ein Formparameter, Sinspelt → Prümzurlay ist mit 2,9–4 m/s zu schnell für eine Welle, Echtershausen → Wiersdorf ist der Stausee.

`chain_fc.CELER` (`{Unterlieger: {Oberlieger: f}}`, in der Seite `fl.celer`) multipliziert c(Q) einer Strecke mit f (Grenzen 0,5 bis 2). Leer oder 1 rechnet exakt wie bisher (im Test geprüft); die Laufzeit des gerouteten Anteils skaliert mit 1/f (Juli 2021, Wiersdorf-Anteil in Prümzurlay: 4,25 h bei f = 1, 6,25 h bei 0,7, 3,25 h bei 1,3). Gedacht ist f für die im Ereignis gemessene Laufzeit des Scheitels über die kommunalen Pegel (Wiersdorf → Brecht → Bettingen → Holsthum Prüm); noch nirgends gesetzt. Der Scheitel in Prümzurlay verschiebt sich dabei nicht einfach um dieselbe Zeit, weil Enz und Zwischengebiet mitwirken.

`kommunal_lage.py` schreibt außerdem `data/kommunal_strecken.csv`: Fließstrecke jeder kommunalen Messstelle zu den Landespegeln flussab und Flächenanteil (zum Beispiel Kinzenburg → Echtershausen 20 km bei 80 %, Mauel → Echtershausen 11 km bei 92 %, Rittersdorf → Alsdorf-Oberecken 21,5 km bei 70 %).
