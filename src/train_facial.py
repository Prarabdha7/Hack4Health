import os
import numpy as np
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


IMAGE_DIR = "data/images/Extracted_images"

EMOTION_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

STRESS_CLASSES = [
    "Healthy",
    "Mild_Stress",
    "Moderate_Stress",
    "Severe_Stress"
]

EMOTION_TO_STRESS = {
    "Happy": "Healthy",
    "Neutral": "Healthy",
    "Sad": "Mild_Stress",
    "Surprise": "Mild_Stress",
    "Fear": "Moderate_Stress",
    "Disgust": "Moderate_Stress",
    "Angry": "Severe_Stress"
}

stress_to_id = {
    stress: i
    for i, stress in enumerate(STRESS_CLASSES)
}

files = []
labels = []

for emotion in EMOTION_CLASSES:
    emotion_path = os.path.join(
        IMAGE_DIR,
        emotion
    )

    if not os.path.isdir(emotion_path):
        continue

    stress = EMOTION_TO_STRESS[emotion]
    label = stress_to_id[stress]

    for filename in sorted(os.listdir(emotion_path)):
        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        files.append(
            os.path.join(
                emotion_path,
                filename
            )
        )

        labels.append(label)

train_files, test_files, train_labels, test_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training files:", len(train_files))
print("Testing files:", len(test_files))

train_counts = np.bincount(
    train_labels,
    minlength=len(STRESS_CLASSES)
)

test_counts = np.bincount(
    test_labels,
    minlength=len(STRESS_CLASSES)
)

print("Training class counts:", train_counts.tolist())
print("Testing class counts:", test_counts.tolist())

class FacialDataset(Dataset):
    def __init__(
        self,
        files,
        labels,
        augment=False
    ):
        self.files = files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        label = self.labels[index]

        image = Image.open(
            path
        ).convert("L")

        image = image.resize(
            (48, 48)
        )

        image = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        if self.augment:
            if np.random.random() < 0.5:
                image = np.fliplr(image).copy()

            if np.random.random() < 0.15:
                noise = np.random.normal(
                    0,
                    0.02,
                    image.shape
                ).astype(np.float32)

                image = np.clip(
                    image + noise,
                    0,
                    1
                )

        tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        return (
            tensor,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


class FacialCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                256,
                128
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

train_dataset = FacialDataset(
    train_files,
    train_labels,
    augment=True
)

test_dataset = FacialDataset(
    test_files,
    test_labels,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(STRESS_CLASSES)),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print(
    "Class weights:",
    class_weights.cpu().numpy()
)

model = FacialCNN(
    len(STRESS_CLASSES)
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=25,
    eta_min=1e-5
)

epochs = 25

best_f1 = -1

os.makedirs(
    "models",
    exist_ok=True
)

print("Device:", device)
print("Classes:", STRESS_CLASSES)

for epoch in range(epochs):
    model.train()

    train_predictions = []
    train_targets = []
    total_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        train_predictions.extend(
            outputs.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        train_targets.extend(
            targets
            .detach()
            .cpu()
            .numpy()
        )

    scheduler.step()

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )

    model.eval()

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            test_predictions.extend(
                outputs.argmax(dim=1)
                .cpu()
                .numpy()
            )

            test_targets.extend(
                targets
                .cpu()
                .numpy()
            )

    test_accuracy = accuracy_score(
        test_targets,
        test_predictions
    )

    test_f1 = f1_score(
        test_targets,
        test_predictions,
        average="macro"
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Test Acc: {test_accuracy:.4f} "
        f"Test Macro F1: {test_f1:.4f} "
        f"LR: {current_lr:.6f}"
    )

    if test_f1 > best_f1:
        best_f1 = test_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": STRESS_CLASSES
            },
            "models/facial_stress_cnn.pth"
        )

        print(
            f"Best model saved with Macro F1: {best_f1:.4f}"
        )

checkpoint = torch.load(
    "models/facial_stress_cnn.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

final_predictions = []
final_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)

        final_predictions.extend(
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
        )

        final_targets.extend(
            targets.numpy()
        )

final_accuracy = accuracy_score(
    final_targets,
    final_predictions
)

final_f1 = f1_score(
    final_targets,
    final_predictions,
    average="macro"
)

print("\nBest Model Results:")
print(
    f"Test Accuracy: {final_accuracy:.4f}"
)
print(
    f"Test Macro F1: {final_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        final_targets,
        final_predictions,
        target_names=STRESS_CLASSES,
        zero_division=0
    )
)

print(
    "Best model saved to models/facial_stress_cnn.pth"
)