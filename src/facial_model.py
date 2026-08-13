import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from config import EMOTION_CLASSES, EMOTION_TO_STRESS, STRESS_CLASSES


class FacialCNN(nn.Module):
    """Compact CNN for FER-style 48x48 grayscale images."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.20),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FacialDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files, self.labels, self.augment = files, labels, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("L").resize((48, 48))
        image = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
            if np.random.rand() < 0.25:
                contrast = np.random.uniform(0.85, 1.15)
                image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
            if np.random.rand() < 0.15:
                noise = np.random.normal(0, 0.015, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)
        return torch.from_numpy(image).unsqueeze(0), torch.tensor(self.labels[index], dtype=torch.long)


def emotion_probabilities_to_stress(probabilities):
    out = np.zeros(len(STRESS_CLASSES), dtype=np.float64)
    stress_to_idx = {name: i for i, name in enumerate(STRESS_CLASSES)}
    for i, emotion in enumerate(EMOTION_CLASSES):
        out[stress_to_idx[EMOTION_TO_STRESS[emotion]]] += float(probabilities[i])
    return out / out.sum()
