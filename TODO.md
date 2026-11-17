# Offene Punkte

Stand 17.11.2026. Reihenfolge innerhalb der Abschnitte nach erwarteter Wirkung.

## Vor dem Sturm am 17./18.11.2026

- [x] **Auswertung des Live-Tests:** `live_eval.py` stellt nach jedem Lauf alle archivierten Ausgaben gegen die Messung (`auswertung.html`): Fehler nach Vorlauf für Kette, ICON-D2-Hauptlauf, ICON-RUC, HVZ und „bleibt wie jetzt“, Messung im 80-%-Band, je Ausgabe Höchststand und Warnstatus. Nach dem Sturm ansehen, dabei den Niveaufaktor in Alsdorf-Oberecken (0,76) und Prümzurlay (1,16) prüfen.
- [x] **Kommunale Pegel und Seestand Stausee Bitburg** werden ab jetzt bei jedem Lauf mit abgerufen, im Rohordner archiviert und auf der Live-Seite gezeigt (20 Messstellen an Prüm, Mehlenbach, Alfbach, Enz und Nims). Rein zusätzlich: Fehler, Zeitüberschreitung oder kaputte Daten lassen nur die betroffene Messstelle oder den Abschnitt weg, Abrufbudget 90 s, Vorhersage und Warnung hängen nicht daran.
- [x] **Haltezeit für Warnungen:** Vorwarnung und Warnung bleiben 12 h stehen (`live.HOLD_H`). Offen: Wirkung auf die Fehlalarme in der historischen Auswertung nachrechnen.

## Modell

- [ ] **Zwischenzuflüsse als Teilgebiete:** Zwischengebiete und Oberläufe an den Mündungen der Nebenbäche teilen (Höhenmodell), je Teil Laufzeit bis zum Pegel und größenabhängige Reaktionszeit, Regen je Teil aus dem Raster (Menge von den Stationen, Verteilung vom Radar). Test zuerst an Echtershausen. Hintergrund: Fehlerzerlegung der Starkregen-Hochwasser (Volumen 0,80, Scheitel 0,65, 3 h zu spät).
- [ ] **Hochwasser Januar 2003 ins Training** (siehe unten).
- [ ] **Kommunale Pegel nutzen** (sobald genug archiviert ist; die Schnittstelle liefert nur die letzten 5 Tage, ein Archiv gibt es nicht): (1) als Prüfstein für die Teilgebiete (Mehlenbach bei Niedermehlen, Alfbach bei Habscheid, Enz-Oberlauf bei Neurath/Arzfeld, Nims bei Rommersheim), (2) als Frühzeichen in der Korrektur der Oberlieger (Anstieg oberhalb Prüm 2 / Giesdorf / Sinspelt), (3) Wasserstand-Abfluss-Beziehung je Messstelle aus Vergleich mit den Landespegeln. Immer nur als Zusatz: die Vorhersage muss ohne sie gleich gut laufen, weil die Messstellen weniger ausfallsicher sind (ungeprüfte Rohdaten der Gemeinden).
- [ ] **Stausee Bitburg (zwischen Echtershausen und Wiersdorf):** Regeleingriffe sind im Modell nicht abgebildet. Befund aus `stausee.py` (63 Hochwasser ab 40 m³/s in Echtershausen): Bei kleinen und mittleren Hochwassern (unter 90 m³/s) lag der Scheitel in Wiersdorf in den 1990ern beim 1,07-Fachen von Echtershausen, 2000–2009 beim 1,02-Fachen, 2010–2015 beim 0,90-Fachen und 2016–2022 beim 0,82-Fachen; zurückgehalten werden heute je Hochwasser rund 0,6 bis 1,2 Mio m³. Der See dämpft also heute deutlich stärker als früher, und das Modell Echtershausen → Wiersdorf mittelt über beide Zustände. 2003: Vorabgabe, dann Rückhalt von rund 90 m³/s im Scheitel. Februar 1991: Wiersdorf 12 m³/s bei 69 m³/s Zufluss (See offenbar leer). Der Seestand (Messstelle 26280366) wird seit 17.11.2026 live archiviert; eine Ganglinie zum Herunterladen gibt es unter geodaten-wasser.rlp-umwelt.de (Laden braucht Freigabe). Schritte: (1) Wiersdorf nur am heutigen Zustand (ab etwa 2016) anpassen oder einen Speicher mit rund 1 Mio m³ Rückhalteraum als eigenes Glied einbauen und prüfen, (2) Betriebsregeln und Seestand beim Betreiber erfragen, (3) Vorabgabe sauber erkennen (der einfache Vergleich mit 2 h Versatz verwechselt sie mit der Laufzeit).
- [ ] **Winterhochwasser der frühen 1990er (1991, 1993, 1995) ins Training:** Der Regen dafür ist gebaut und geprüft (`day_rain.py`: Tagessummen von 24 DWD-Stationen, Stundenverlauf aus ERA5; an 2004–2022 liefert die Kette damit bei Winterhochwassern praktisch dieselben Scheitel wie mit gemessenem Stundenregen, siehe README). Es fehlt die **Schneeschmelze**: Dezember 1991 (20 cm Schnee, am Scheiteltag weg), Dezember 1993 (25 cm), Januar 1995 (18 cm), auch Januar 2011 (54 cm) und Januar 2021 (31 cm) hatten Schnee; nur Januar 1993 und Januar 2003 nicht. Die Kette liegt 1991 und 1995 weit zu niedrig (Prümzurlay 24 h vorher 0,39 und 0,75). Januar 1993 liegt ebenfalls zu niedrig (0,48), obwohl kein Schnee lag: dort die Ursache suchen, bevor es ins Training geht. Reihenfolge: erst das Schneemodul, dann den Zeitraum 1989–2003 mit Ersatzregen in Anpassung und Kreuzvalidierung aufnehmen. Wiersdorf ist 1991 lange abgeschnitten.
- [ ] **Schneemodul:** Schneehöhen liegen als Tageswerte vor (in `data/tag/dwd_tageswerte_rr.json`). Einfachster Ansatz: Schmelze aus dem Rückgang der Schneehöhe mal einer angepassten Dichte, über den Tag verteilt; besser mit Temperatur (Grad-Tag-Verfahren, braucht Stundentemperaturen und im Betrieb die Temperaturvorhersage). Prüfsteine: Prüm 2 Januar 2021, Januar 2011, Dezember 1991.
- [ ] **Große Winterhochwasser einen Tag vorher zu niedrig:** Bei den größten 10 % der Winterhochwasser liegt der Scheitel der Kette mit gemessenem Regen 24 h vorher im Median bei 0,53, 12 h vorher bei 0,72 (alle Winterhochwasser: 0,80 und 0,88). Ursache offen; Schnee ist ein Kandidat.
- [ ] **Niveaufaktor aus den Live-Daten nachführen,** sobald Hochwasserwerte mit der heutigen Kurve vorliegen; Seffern klären (bisher kein Faktor).
- [ ] **Nachbargebiete dazunehmen** (Kyll, Our, Sauer, Ahr, Lieser, Salm): gemeinsames Modell über viele Pegel, um mehr Starkregen-Hochwasser zu haben. Braucht neue Daten.
- [ ] **Längeres Nachtraining / anderes Gewicht des Scheitelfehlers** im Starkregensatz (0,5 statt 0,2 brachte HQ10 0,82 statt 0,77 bei 17,2 statt 16,0 % Überschätzung).
- [ ] **Radar-Nowcasting** für die Oberläufe (zurückgestellt, bis es mit gemessenem Regen gut läuft).

### Januar 2003 ins Training: was fehlt

- [ ] **Zeitblock 2001–2003** in Anpassung und Kreuzvalidierung aufnehmen (`chain_fc.BLOCKS`, Unsicherheitstabellen, Texte); alles neu rechnen (Kette, Archiv, Tabellen: rund 5 h).
- [x] **Regen prüfen:** Tageswerte geladen. Für Januar 2003 den Ersatzregen aus Tagessummen und ERA5 nehmen statt des Radars: Damit liegt die Kette an der Nims einen Tag vorher bei 0,91 (Giesdorf), 1,05 (Seffern) und 1,30 (Alsdorf-Oberecken) des Scheitels statt 1,44, 1,55 und 1,65 mit Radar; das Radar hatte dort zu viel Regen. Kein Schnee im Ereignis.
- [ ] **Wiersdorf / Stausee Bitburg:** 2003 ist der Abfluss in Wiersdorf nicht (nur) am Pegel abgeschnitten, sondern vom Stausee geregelt: am 02.01. 20 Uhr 133 m³/s bei 77 m³/s Zufluss (Vorabgabe), im Scheitel rund 100 m³/s bei 181 m³/s in Echtershausen (rund 8 h zurückgehalten), danach wieder 130 m³/s. Den Abfluss aus Echtershausen zu ergänzen wäre hier falsch. Erst die Stausee-Regelung klären (siehe „Stausee Bitburg“), dann entscheiden, wie Wiersdorf und Prümzurlay 2003 ins Training gehen. Echtes Abschneiden bei 250 cm betrifft vor allem 1991 (18 h) und 2018 (5 h).
- [ ] **Radarlücke** am 03.01.2003, 03 bis 07 Uhr (linear gefüllt) gegen die Tageswerte prüfen.

## Daten und Klärungen

- [ ] **Anfrage ans LfU:** Pegelreihen bis heute; Grundlage von HQ2 bis HQ100 (gelieferte oder neu umgerechnete Abflüsse); Niveausprung in Prümzurlay seit 2022 (20 bis 30 % mehr Abfluss bei gleichem Zufluss); Veröffentlichung der Messdaten (Prüf-Seite und Karte liegen bis dahin nur im Volume `seiten/`).
- [ ] **Anfrage an den Betreiber des Stausees Bitburg:** Betriebsregeln bei Hochwasser (Vorabsenkung, Rückhalteraum, Abgabegrenzen) und Zeitreihen von Seestand und Abgabe.
- [ ] **Regen vor 2001:** beim DWD (oder Kachelmannwetter) nach Radar- und Stundendaten der 1990er fragen.
- [ ] **RADKLIM 2023–2025** liegt vor, nützt erst mit Pegeldaten dieser Jahre.

## Betrieb und Seiten

- [ ] **Umstellung bei Open-Meteo zum 30.11.2026:** vorher prüfen, ob die Abrufe (ICON-D2, Ensemble, EU, global, RUC, Nachbarschaft) weiterlaufen.
- [ ] **Nachbarschafts-Ensemble mit echten Daten prüfen** (Protokollzeile `Nachbarschaft: n von 8 Richtungen`); ICON-RUC läuft (16.11.: 26 h Reichweite).
- [ ] **Regenschwellen** mit dem heutigen Starkregensatz neu prüfen (`rain_thr.py check`).
- [ ] **Routing-Karte:** alte Hochwasser mit der damals gültigen Wasserstand-Abfluss-Beziehung zeigen (wie die Prüf-Seite).
- [ ] **Laufzeit des Seitenaufbaus** auf dem ARM-Rechner messen (im Test rund 50 s).
- [ ] **Download-Ordner:** versteckte Zwischendateien der Browser-Downloads (rund 100 MB) löschen, braucht Freigabe.
