# Offene Punkte

Stand 16.11.2026. Reihenfolge innerhalb der Abschnitte nach erwarteter Wirkung.

## Vor dem Sturm am 17./18.11.2026

- [ ] **Auswertung des Live-Tests:** Skript, das nach dem Ereignis alle archivierten Ausgaben gegen die Messung stellt, getrennt nach Kette, ICON-RUC, Nachbarschaft, Regenschwellen und Vorhersage der HVZ; dabei den Niveaufaktor in Alsdorf-Oberecken (0,76) und Prümzurlay (1,16) prüfen.
- [ ] **Haltezeit für Warnungen:** Eine Warnung bleibt 12 h stehen, auch wenn die Wahrscheinlichkeit kurz absackt (Fall Wiersdorf Juli 2021).

## Modell

- [ ] **Zwischenzuflüsse als Teilgebiete:** Zwischengebiete und Oberläufe an den Mündungen der Nebenbäche teilen (Höhenmodell), je Teil Laufzeit bis zum Pegel und größenabhängige Reaktionszeit, Regen je Teil aus dem Raster (Menge von den Stationen, Verteilung vom Radar). Test zuerst an Echtershausen. Hintergrund: Fehlerzerlegung der Starkregen-Hochwasser (Volumen 0,80, Scheitel 0,65, 3 h zu spät).
- [ ] **Hochwasser Januar 2003 ins Training** (siehe unten).
- [ ] **Niveaufaktor aus den Live-Daten nachführen,** sobald Hochwasserwerte mit der heutigen Kurve vorliegen; Seffern klären (bisher kein Faktor).
- [ ] **Nachbargebiete dazunehmen** (Kyll, Our, Sauer, Ahr, Lieser, Salm): gemeinsames Modell über viele Pegel, um mehr Starkregen-Hochwasser zu haben. Braucht neue Daten.
- [ ] **Schneeschmelze:** Prüm 2 im Januar 2021 verfehlt; das Modell kennt keinen Schnee.
- [ ] **Längeres Nachtraining / anderes Gewicht des Scheitelfehlers** im Starkregensatz (0,5 statt 0,2 brachte HQ10 0,82 statt 0,77 bei 17,2 statt 16,0 % Überschätzung).
- [ ] **Radar-Nowcasting** für die Oberläufe (zurückgestellt, bis es mit gemessenem Regen gut läuft).

### Januar 2003 ins Training: was fehlt

- [ ] **Zeitblock 2001–2003** in Anpassung und Kreuzvalidierung aufnehmen (`chain_fc.BLOCKS`, Unsicherheitstabellen, Texte); alles neu rechnen (Kette, Archiv, Tabellen: rund 5 h).
- [ ] **Regen prüfen:** Für 2001–2003 gibt es nur Radar (angeglichen mit dem Verhältnis ab 2004). Die Station Prüm-Watzerath hat in der Ereigniswoche nur 6 Stundenwerte, ein Vergleich ist damit nicht möglich. Tageswerte des DWD für die Stationen der Umgebung laden und die Radarsummen im Dezember 2002 / Januar 2003 daran prüfen und angleichen; dabei auch Schneehöhe ansehen. Die Kette lag an der Nims mit Radarregen einen Tag vorher 45 bis 65 % zu hoch.
- [ ] **Wiersdorf ergänzen:** Der Wasserstand ist bei 250 cm abgeschnitten (2003 rund 14 h am Anschlag, Abfluss 137 m³/s bei 181 m³/s in Echtershausen). Den abgeschnittenen Teil aus Echtershausen ergänzen (Modell Echtershausen → Wiersdorf aus den nicht abgeschnittenen Hochwassern, an den Rändern an die Messung angeschlossen). Betrifft auch 1991, 1993, 1995, 2018 und 2021. Ergänzte Werte nur als Eingang für Prümzurlay verwenden, nicht als Ziel für Wiersdorf.
- [ ] **Radarlücke** am 03.01.2003, 03 bis 07 Uhr (linear gefüllt) gegen die Tageswerte prüfen.

## Daten und Klärungen

- [ ] **Anfrage ans LfU:** Pegelreihen bis heute; Grundlage von HQ2 bis HQ100 (gelieferte oder neu umgerechnete Abflüsse); Niveausprung in Prümzurlay seit 2022 (20 bis 30 % mehr Abfluss bei gleichem Zufluss); Veröffentlichung der Messdaten (Prüf-Seite und Karte liegen bis dahin nur im Volume `seiten/`).
- [ ] **Regen vor 2001:** beim DWD (oder Kachelmannwetter) nach Radar- und Stundendaten der 1990er fragen.
- [ ] **RADKLIM 2023–2025** liegt vor, nützt erst mit Pegeldaten dieser Jahre.

## Betrieb und Seiten

- [ ] **Umstellung bei Open-Meteo zum 30.11.2026:** vorher prüfen, ob die Abrufe (ICON-D2, Ensemble, EU, global, RUC, Nachbarschaft) weiterlaufen.
- [ ] **Nachbarschafts-Ensemble mit echten Daten prüfen** (Protokollzeile `Nachbarschaft: n von 8 Richtungen`); ICON-RUC läuft (16.11.: 26 h Reichweite).
- [ ] **Regenschwellen** mit dem heutigen Starkregensatz neu prüfen (`rain_thr.py check`).
- [ ] **Routing-Karte:** alte Hochwasser mit der damals gültigen Wasserstand-Abfluss-Beziehung zeigen (wie die Prüf-Seite).
- [ ] **Laufzeit des Seitenaufbaus** auf dem ARM-Rechner messen (im Test rund 50 s).
- [ ] **Download-Ordner:** versteckte Zwischendateien der Browser-Downloads (rund 100 MB) löschen, braucht Freigabe.
