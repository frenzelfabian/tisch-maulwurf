# Tisch-Maulwurf · Kalibrier-Labor

Klopf-Erkennung fürs iPhone, inspiriert von [Holo](https://github.com/JustinGamer191/Holo): Das iPhone liegt flach auf dem Tisch, du klopfst **neben** das Handy in eine von vier Ecken (hinten/vorne × links/rechts). Mikrofon (und Bewegungssensor) erkennen, welche Ecke es war.

**Öffnen:** https://frenzelfabian.github.io/tisch-maulwurf/ (Safari, Mikrofon + Bewegung erlauben)

## Ablauf
1. **Kalibrieren:** 10 Klopfer pro Ecke. Danach Übereinstimmung pro Ecke; schwächste Ecke einzeln wiederholbar.
2. **Störgeräusche aufnehmen:** 20 s Musik, Reden, Rascheln. Wird als „kein Klopfen“ gelernt; außerdem werden die Kalibrier-Klopfer mit diesen Geräuschen gemischt nachtrainiert.
3. **Genauigkeitstest** (wie Holo): 15 Klopfer pro Ecke + 15 s Pause zum Zählen von Fehlauslösern, je Hintergrund (ruhig / Reden / Musik). Ergebnis als Verwechslungstabelle, „Bericht kopieren“.
4. **Daten sammeln:** frei klopfen und danach beschriften, geführt (App sagt die Ecke), oder nur Geräusche. Alles wird mit Rohaudio in der Browser-Datenbank gespeichert. **Exportieren** gibt eine JSON-Datei (Teilen-Menü auf dem iPhone).

## Erkennung
`engine.js`: Onset-Erkennung (plötzlicher Anstieg über Rauschpegel) → 85-ms-Fenster → Abzug des laufenden Hintergrundspektrums → Merkmale (20 Frequenzbänder der ersten ~40 ms, Abklingverlauf, Stereo-Unterschiede, Bewegungssensor) → Ridge-Regression wie HoloCore + Nächste-Nachbarn-Prüfung für unbekannte Geräusche und gelernte Störgeräusche. Bewegungs-Filter wird nur aktiv, wenn die Klopfer das Handy messbar erschüttern.

Alles läuft lokal im Browser, nichts wird hochgeladen.

## Export-Format
`events[]`: `label` (0 hinten links, 1 hinten rechts, 2 vorne links, 3 vorne rechts, −1 kein Klopfen), `table`, `background`, `kind`, `predicted`, `motion`, `ch0`/`ch1` = 4096 Samples 16-bit PCM base64, Klopfbeginn bei Sample 256.
