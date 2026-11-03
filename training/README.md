# Training (PyTorch)

Kreislauf: **in der App Daten sammeln → exportieren → hier trainieren → `model.onnx` hochladen → App nutzt das Netz automatisch → weiter sammeln.**

```sh
cd training
pip install -r requirements.txt
python train.py daten/*.json          # alle bisherigen Exporte zusammen
```

Ergebnis:
- `../model.onnx` (~300 KB) und `../model-meta.json`. Sobald beide im Repo liegen, lädt die App das Netz (onnxruntime-web) und benutzt es für Genauigkeitstest, Live-Test, Daten sammeln („Modell sagt …“) und Spiel.
- Ein Bericht: Genauigkeit pro Ecke, Verwechslungstabelle, Fehlauslöser bei Störgeräuschen. Bei mehreren Tischen zusätzlich: Wie gut klappt es auf einem Tisch, den das Netz nie gesehen hat?

## Was das Netz sieht
- `tapdata.py`: ~43 ms Wellenform ab kurz vor dem Klopfbeginn (Mikrofon 1), auf Spitzenwert normiert, plus 6 Zusatzwerte (Lautstärke, Bewegungssensor-Stärke und -Richtung). Muss exakt zu `netInput()` in `../app.js` passen.
- `model.py`: kleines 1D-CNN (5 Faltungsschichten, ~60k Parameter), 5 Klassen: 4 Ecken + „kein Klopfen“.
- `train.py`: Training mit zufälliger Verschiebung, Lautstärke, Einmischen echter Störgeräusche aus den Daten und fehlendem Bewegungssensor. Unsichere Vorhersagen (< 60 %) gelten als „abgelehnt“.

`make_synthetic.py` erzeugt künstliche Daten, nur zum Testen der Pipeline.
