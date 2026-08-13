import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn

MODEL_PATH = "models/facial_emotion_cnn.pth"

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

class FacialCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = FacialCNN(len(emotion_names)).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = emotion_names[predicted_index]

    print("\nFacial Emotion")
    print("--------------")

    for emotion, probability in zip(
        emotion_names,
        probabilities
    ):
        print(
            f"{emotion}: {probability:.4f}"
        )

    print(
        f"\nDominant: {predicted_emotion}"
    )

    return predicted_emotion, probabilities

if __name__ == "__main__":
    image_path = input(
        "Enter image path: "
    ).strip()

    predict(image_path)