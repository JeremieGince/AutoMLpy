import numpy as np
import torch.nn as nn


class DeepMnistNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(150, 300, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(300, 300, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(300, 150, 3, padding=1),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.clf(feat)
        return logits


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(10, 50, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.fc1 = nn.Linear(50 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, np.newaxis, :, :]
        feat = self.backbone(x)
        logits = self.clf(feat)
        return logits


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(10, 50, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(50, 150, 3, padding=1),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150 * 8 * 8, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.clf(feat)
        return logits


class CifarNetBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 10, 5, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 50, 3, padding=1, stride=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 150, 3, padding=1, stride=2),
            nn.BatchNorm2d(150),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2400, 10),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.clf(feat)
        return logits
