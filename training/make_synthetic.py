"""Erzeugt einen künstlichen Export zum Testen der Trainings-Pipeline (keine echten Daten)."""
import base64, json, sys
import numpy as np

rng = np.random.default_rng(0)
SR = 48000


def b64(x):
    return base64.b64encode(np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()).decode()


def background(kind, n):
    t = np.arange(n) / SR + rng.uniform(0, 10)
    x = rng.normal(0, 0.001, n)
    if "Musik" in kind:
        x += 0.03 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.sin(2 * np.pi * 330 * t)
        ph = t % 0.5; x += 0.12 * np.exp(-ph * 30) * np.sin(2 * np.pi * 60 * ph)
    if "Reden" in kind:
        x += 0.06 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * 150 * t + 3 * np.sin(2 * np.pi * 5 * t))
    return x


def tap(table, z):
    n = 4096; i = np.arange(n - 256)
    base = [[700, 780, 900, 1000], [500, 560, 650, 610]][table]
    f = base[z] * (1 + 0.05 * rng.normal()); dec = [500, 560, 420, 470][z] * (1 + 0.2 * rng.normal())
    x = np.zeros(n); x[256:] = 0.3 * (1 + 0.3 * rng.normal()) * np.exp(-i / dec) * (np.sin(2 * np.pi * f * i / SR) + 0.4 * rng.normal(size=len(i)))
    return x


events, eid = [], 1
for table in (0, 1):
    for bg in ("ruhig", "Reden", "Musik"):
        for k in range(60):
            lab = int(rng.integers(-1, 4))
            x = background(bg, 4096)
            if lab >= 0:
                x = x + tap(table, lab)
            else:  # Störgeräusch mit Anschlag (z. B. Klick)
                j = np.arange(3840); x[256:] += 0.2 * np.exp(-j / 150) * rng.normal(size=3840)
            mot = {"peak": float(0.08 + 0.02 * rng.normal()) if lab >= 0 else float(abs(0.004 * rng.normal())),
                   "noise": 0.004, "dir": [(-1) ** (lab % 2) * 0.5, (-1) ** (lab // 2) * 0.5, 0.7] if lab >= 0 else [0, 0, 1]}
            events.append({"id": eid, "label": lab, "table": ["Holztisch", "Glastisch"][table], "background": bg,
                           "kind": "collect", "sr": SR, "mics": 1, "motion": mot, "ch0": b64(x), "ch1": None})
            eid += 1
json.dump({"app": "tisch-maulwurf", "synthetic": True, "events": events}, open(sys.argv[1], "w"))
print(len(events), "künstliche Aufnahmen")
