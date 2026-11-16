# Offene Punkte

Stand 16.11.2026. Reihenfolge innerhalb der Abschnitte nach erwarteter Wirkung.

## Vor dem Sturm am 17./18.11.2026

- [x] **Auswertung des Live-Tests:** `live_eval.py` stellt nach jedem Lauf alle archivierten Ausgaben gegen die Messung (`auswertung.html`): Fehler nach Vorlauf für Kette, ICON-D2-Hauptlauf, ICON-RUC, HVZ und „bleibt wie jetzt“, Messung im 80-%-Band, je Ausgabe Höchststand und Warnstatus. Nach dem Sturm ansehen, dabei den Niveaufaktor in Alsdorf-Oberecken (0,76) und Prümzurlay (1,16) prüfen.
- [x] **Haltezeit für Warnungen:** Vorwarnung und Warnung bleiben 12 h stehen (`live.HOLD_H`). Offen: Wirkung auf die Fehlalarme in der historischen Auswertung nachrechnen.

## Modell

- [ ] **Zwischenzuflüsse als Teilgebiete:** Zwischengebiete und Oberläufe an den Mündungen der Nebenbäche teilen (Höhenmodell), je Teil Laufzeit bis zum Pegel und größenabhängige Reaktionszeit, Regen je Teil aus dem Raster (Menge von den Stationen, Verteilung vom Radar). Test zuerst an Echtershausen. Hintergrund: Fehlerzerlegung der Starkregen-Hochwasser (Volumen 0,80, Scheitel 0,65, 3 h zu spät).
- [ ] **Hochwasser Januar 2003 ins Training** (siehe unten).
- [ ] **Stausee Bitburg (zwischen Echtershausen und Wiersdorf):** Regeleingriffe sind im Modell nicht abgebildet. 2003 gut sichtbar (Vorabgabe, dann Rückhalt von rund 70 bis 80 m³/s im Scheitel), 2021 Vorabsenkung am 13.07. (4 bis 6 m³/s mehr Abgabe als Zufluss) und dann wenig Wirkung. Schritte: (1) alle Hochwasser nach dem Muster Abgabe minus Zufluss durchsehen und Eingriffe kennzeichnen, (2) Betriebsregeln und Seestand beim Betreiber erfragen, (3) Speicher mit Rückhalteraum und Vorabgabe als eigenes Glied zwischen Echtershausen und Wiersdorf, (4) bis dahin gekennzeichnete Eingriffe aus der Anpassung von Wiersdorf nehmen und auf der Live-Seite darauf hinweisen, dass Wiersdorf und Prümzurlay bei Regelung abweichen können.
- [ ] **Winterhochwasser der frühen 1990er (1991, 1993, 1995) ins Training:** Stundenregen gibt es dafür nicht (Radar ab 2001, Stationen ab 2002/2004). Weg: Tagessummen der DWD-Stationen für die Menge, Stundenverlauf aus der Reanalyse ERA5 (stündlich ab 1940). Vorher an 2004–2022 prüfen, wie gut ein so gebauter Regen den gemessenen Stundenregen ersetzt (Scheitelzeit und -höhe der Kette). Wiersdorf ist 1991 lange abgeschnitten.
- [ ] **Niveaufaktor aus den Live-Daten nachführen,** sobald Hochwasserwerte mit der heutigen Kurve vorliegen; Seffern klären (bisher kein Faktor).
- [ ] **Nachbargebiete dazunehmen** (Kyll, Our, Sauer, Ahr, Lieser, Salm): gemeinsames Modell über viele Pegel, um mehr Starkregen-Hochwasser zu haben. Braucht neue Daten.
- [ ] **Schneeschmelze:** Prüm 2 im Januar 2021 verfehlt; das Modell kennt keinen Schnee.
- [ ] **Längeres Nachtraining / anderes Gewicht des Scheitelfehlers** im Starkregensatz (0,5 statt 0,2 brachte HQ10 0,82 statt 0,77 bei 17,2 statt 16,0 % Überschätzung).
- [ ] **Radar-Nowcasting** für die Oberläufe (zurückgestellt, bis es mit gemessenem Regen gut läuft).

### Januar 2003 ins Training: was fehlt

- [ ] **Zeitblock 2001–2003** in Anpassung und Kreuzvalidierung aufnehmen (`chain_fc.BLOCKS`, Unsicherheitstabellen, Texte); alles neu rechnen (Kette, Archiv, Tabellen: rund 5 h).
- [ ] **Regen prüfen:** Für 2001–2003 gibt es nur Radar (angeglichen mit dem Verhältnis ab 2004). Die Station Prüm-Watzerath hat in der Ereigniswoche nur 6 Stundenwerte, ein Vergleich ist damit nicht möglich. Tageswerte des DWD für die Stationen der Umgebung laden und die Radarsummen im Dezember 2002 / Januar 2003 daran prüfen und angleichen; dabei auch Schneehöhe ansehen. Die Kette lag an der Nims mit Radarregen einen Tag vorher 45 bis 65 % zu hoch.
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
