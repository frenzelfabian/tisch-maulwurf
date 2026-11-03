# Tisch-Maulwurf

Ein Klopf-Spiel fürs iPhone, inspiriert von [Holo](https://github.com/JustinGamer191/Holo): Das iPhone liegt flach auf dem Tisch, du klopfst **neben** das Handy, dort wo der Maulwurf auftaucht. Das Mikrofon erkennt, in welche der vier Ecken du geklopft hast.

**Spielen:** https://frenzelfabian.github.io/tisch-maulwurf/ (in Safari öffnen, Mikrofon erlauben)

## So funktioniert's
1. Kalibrieren: 8 Klopfer pro Ecke (hinten links, hinten rechts, vorne links, vorne rechts).
2. Testen: prüfen, ob die Ecken richtig erkannt werden.
3. Spielen: 45 Sekunden, so viele Maulwürfe wie möglich treffen.

Die Erkennung ist eine vereinfachte JavaScript-Version der Holo-Pipeline: Klopf-Erkennung mit adaptivem Rauschpegel, Frequenz- und Zeit-Merkmale, k-nächste-Nachbarn mit Ablehnung unklarer Geräusche. Alles läuft lokal im Browser, es wird kein Audio gespeichert oder hochgeladen.
