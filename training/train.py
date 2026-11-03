"""Trainiert TapNet auf exportierten Daten aus dem Kalibrier-Labor.

    pip install -r requirements.txt
    python train.py daten/*.json

Schreibt ../model.onnx und ../model-meta.json (die App lädt beides automatisch)
und gibt einen Bericht aus: Genauigkeit pro Ecke, Verwechslungstabelle,
Fehlauslöser bei Störgeräuschen, und (bei mehreren Tischen) wie gut das
Modell auf einem Tisch klappt, den es nie gesehen hat.
"""
import argparse, json, os, random, datetime
from collections import Counter

import numpy as np
import torch
from torch import nn

from tapdata import CLASSES, NOISE, START, LEN, load, wave_input, extras
from model import TapNet

HERE = os.path.dirname(os.path.abspath(__file__))


def features(it, shift=0, gain=1.0, noise=None, noise_gain=0.0, drop_motion=False):
    x = it["audio"] * gain
    if noise is not None:
        x = x + noise * noise_gain
    peak = float(np.abs(x).max())
    return wave_input(x, START + shift), extras(peak, None if drop_motion else it["motion"])


class Augmenter:
    """Zufällige Verschiebung, Lautstärke, Mischen mit echten Störgeräuschen, fehlender Bewegungssensor."""

    def __init__(self, items, train=True):
        self.items, self.train = items, train
        self.noises = [it["audio"] for it in items if it["label"] == NOISE]

    def batch(self, idx):
        W, E, Y = [], [], []
        for i in idx:
            it = self.items[i]
            if self.train:
                mix = self.noises and it["label"] != NOISE and random.random() < 0.5
                w, e = features(it, shift=random.randint(-48, 48), gain=10 ** random.uniform(-0.3, 0.3),
                                noise=random.choice(self.noises) if mix else None, noise_gain=random.uniform(0.2, 1.5),
                                drop_motion=random.random() < 0.15)
            else:
                w, e = features(it)
            W.append(w); E.append(e); Y.append(it["label"])
        return torch.tensor(np.stack(W))[:, None, :], torch.tensor(np.stack(E)), torch.tensor(Y)


def fit(train_items, epochs, seed=0, log=False):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    net = TapNet()
    aug = Augmenter(train_items, train=True)
    counts = Counter(it["label"] for it in train_items)
    weight = torch.tensor([len(train_items) / (len(CLASSES) * max(1, counts[c])) for c in range(len(CLASSES))], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * max(1, (len(train_items) + 31) // 32))
    for ep in range(epochs):
        net.train()
        order = list(range(len(train_items))); random.shuffle(order)
        tot = 0.0
        for b in range(0, len(order), 32):
            w, e, y = aug.batch(order[b:b + 32])
            if len(y) < 2:
                continue
            loss = lossf(net(w, e), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y)
        if log and (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoche {ep + 1}/{epochs}  Verlust {tot / len(train_items):.3f}")
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, items, threshold):
    aug = Augmenter(items, train=False)
    w, e, y = aug.batch(range(len(items)))
    p = torch.softmax(net(w, e), dim=1)
    conf, pred = p.max(dim=1)
    pred = torch.where((conf < threshold) & (pred != NOISE), torch.full_like(pred, NOISE), pred)  # unsicher → ablehnen
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, q in zip(y.tolist(), pred.tolist()):
        cm[t, q] += 1
    taps = cm[:4].sum()
    return {
        "n": len(items),
        "ecken_richtig": float(np.trace(cm[:4, :4]) / max(1, taps)),
        "ecken_verwechselt": float((cm[:4, :4].sum() - np.trace(cm[:4, :4])) / max(1, taps)),
        "ecken_abgelehnt": float(cm[:4, NOISE].sum() / max(1, taps)),
        "fehlausloeser": float(cm[NOISE, :4].sum() / max(1, cm[NOISE].sum())),
        "pro_ecke": [float(cm[z, z] / max(1, cm[z].sum())) for z in range(4)],
        "verwechslung": cm.tolist(),
    }


def show(title, r):
    print(f"\n{title}  (n={r['n']})")
    print(f"  Ecken richtig {r['ecken_richtig']:.0%} · verwechselt {r['ecken_verwechselt']:.0%} · abgelehnt {r['ecken_abgelehnt']:.0%} · Fehlauslöser bei Störgeräuschen {r['fehlausloeser']:.0%}")
    print("  pro Ecke: " + " · ".join(f"{CLASSES[z]} {v:.0%}" for z, v in enumerate(r["pro_ecke"])))
    print("  getippt ↓ / erkannt →   " + " | ".join(c[:9].rjust(9) for c in CLASSES))
    for z, row in enumerate(r["verwechslung"]):
        print(f"  {CLASSES[z][:22].ljust(22)}  " + " | ".join(str(v).rjust(9) for v in row))


def split(items, frac=0.2, seed=1):
    rng = random.Random(seed)
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["label"], []).append(i)
    test = set()
    for idx in by.values():
        rng.shuffle(idx); test.update(idx[:max(1, int(len(idx) * frac))])
    return [it for i, it in enumerate(items) if i not in test], [it for i, it in enumerate(items) if i in test]


def export(net, path_onnx, path_meta, meta):
    w = torch.zeros(1, 1, LEN); e = torch.zeros(1, len(extras(0.1, None)))
    torch.onnx.export(net, (w, e), path_onnx, input_names=["wave", "extra"], output_names=["logits"],
                      opset_version=17, dynamic_axes=None, dynamo=False)
    with open(path_meta, "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Export-Dateien (JSON) aus der App, Platzhalter erlaubt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.6, help="Mindest-Sicherheit, sonst 'abgelehnt'")
    ap.add_argument("--out", default=os.path.join(HERE, ".."))
    args = ap.parse_args()

    items, files = load(args.files)
    c = Counter(it["label"] for it in items)
    tables = sorted(set(it["table"] for it in items))
    print(f"{len(items)} Aufnahmen aus {len(files)} Datei(en), Tische: {', '.join(tables)}")
    print("  " + " · ".join(f"{CLASSES[k]}: {c[k]}" for k in range(len(CLASSES))))
    if min(c[k] for k in range(4)) < 5:
        print("Zu wenige Klopfer pro Ecke (mindestens 5 je Ecke). Bitte mehr Daten sammeln.")
        return

    report = {}
    tr, te = split(items)
    print(f"\nTest 1: zufällige 20 % zurückgehalten ({len(te)} Aufnahmen)")
    r = evaluate(fit(tr, args.epochs), te, args.threshold); show("Ergebnis auf zurückgehaltenen Aufnahmen", r)
    report["zufaellig"] = r

    if len(tables) > 1:
        report["neuer_tisch"] = {}
        for t in tables:
            tr = [it for it in items if it["table"] != t]; te = [it for it in items if it["table"] == t]
            if min(Counter(it["label"] for it in tr)[k] for k in range(4)) < 5:
                continue
            r = evaluate(fit(tr, args.epochs), te, args.threshold)
            show(f"Test 2: Tisch „{t}“ nie gesehen", r)
            report["neuer_tisch"][t] = r

    print(f"\nEndgültiges Training auf allen {len(items)} Aufnahmen …")
    net = fit(items, args.epochs, log=True)
    meta = {
        "name": datetime.date.today().isoformat(), "classes": CLASSES, "start": START, "length": LEN,
        "threshold": args.threshold, "trainedOn": len(items), "tables": tables,
        "perClass": {CLASSES[k]: c[k] for k in range(len(CLASSES))}, "report": report,
    }
    onnx_path, meta_path = os.path.join(args.out, "model.onnx"), os.path.join(args.out, "model-meta.json")
    export(net, onnx_path, meta_path, meta)
    print(f"\nGeschrieben: {onnx_path} ({os.path.getsize(onnx_path) // 1024} KB) und {meta_path}")


if __name__ == "__main__":
    main()
