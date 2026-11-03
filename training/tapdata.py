"""Lädt Exporte aus dem Kalibrier-Labor und baut die Netz-Eingaben.

Die Vorverarbeitung hier muss exakt zu `netInput()` in ../app.js passen.
"""
import base64, glob, json
import numpy as np

CLASSES = ["hinten links", "hinten rechts", "vorne links", "vorne rechts", "kein Klopfen"]
NOISE = 4               # Klassenindex für "kein Klopfen" (Export-Label -1)
WIN = 4096              # Länge eines Aufnahmefensters
START, LEN = 192, 2048  # Netz sieht ~43 ms ab kurz vor dem Klopfbeginn (Sample 256)
N_EXTRA = 6


def decode(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(np.float32) / 32767.0


def extras(peak, motion):
    """[log Pegel, log Bewegung, Richtung x/y/z, Bewegung vorhanden]"""
    has = 1.0 if motion and motion.get("peak", 0) > 0 else 0.0
    mp = motion["peak"] if has else 0.0
    d = motion.get("dir", [0, 0, 0]) if has else [0, 0, 0]
    return np.array([np.log10(peak + 1e-4), np.log10(mp + 1e-3) * has, d[0], d[1], d[2], has], dtype=np.float32)


def wave_input(x, start=START):
    w = x[start:start + LEN]
    return (w / (np.abs(w).max() + 1e-6)).astype(np.float32)


def load(paths):
    """Gibt eine Liste von Einträgen zurück: dict(audio, label, table, background, motion)."""
    files = []
    for p in paths:
        files += sorted(glob.glob(p))
    items = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for ev in data.get("events", []):
            if ev.get("label") is None or not ev.get("ch0"):
                continue
            lab = int(ev["label"])
            items.append({
                "audio": decode(ev["ch0"]),
                "label": NOISE if lab < 0 else lab,
                "table": ev.get("table") or "unbenannt",
                "background": ev.get("background") or "?",
                "motion": ev.get("motion"),
                "kind": ev.get("kind"),
                "id": ev.get("id"),
            })
    # doppelte Einträge (gleiche id aus mehreren Exporten) nur einmal
    seen, uniq = set(), []
    for it in items:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); uniq.append(it)
    return uniq, files
