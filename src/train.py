import torch
import timm

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from dataset import MedicalImageDataset


DATASET_PATH = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = MedicalImageDataset(
    DATASET_PATH,
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=len(dataset.classes)
)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = train_correct / train_total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {train_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    },
    "models/resnet18_baseline.pth"
)

print("\nTraining complete.")
print("Model saved to models/resnet18_baseline.pth")