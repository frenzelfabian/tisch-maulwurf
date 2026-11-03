import torch
from torch import nn
from tapdata import LEN, N_EXTRA, CLASSES


class TapNet(nn.Module):
    """Kleines 1D-CNN direkt auf der Wellenform (~60k Parameter) + Zusatzmerkmale."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 9, stride=2, padding=4), nn.BatchNorm1d(o), nn.ReLU())
        self.conv = nn.Sequential(block(1, 16), block(16, 32), block(32, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.Linear(128 + N_EXTRA, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, wave, extra):
        h = self.conv(wave)                                   # [B, 64, 64]
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # [B, 128]
        return self.head(torch.cat([h, extra], dim=1))
