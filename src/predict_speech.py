import os
import torch
import librosa
import numpy as np

from torch import nn

MODEL_PATH = "models/speech_emotion_cnn.pth"

emotion_names = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = SpeechCNN(len(emotion_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

emotion_list = list(emotion_names.values())

image_path = input("Enter audio path: ").strip()

if not os.path.exists(image_path):
    print("File not found.")
    exit()

audio, sr = librosa.load(
    image_path,
    sr=16000
)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)

mel = librosa.power_to_db(
    mel,
    ref=np.max
)

target_width = 128

if mel.shape[1] < target_width:
    mel = np.pad(
        mel,
        ((0, 0), (0, target_width - mel.shape[1]))
    )
else:
    mel = mel[:, :target_width]

mel = (mel - mel.mean()) / (mel.std() + 1e-6)

tensor = torch.tensor(
    mel,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

predicted_id = probabilities.argmax().item()
predicted_emotion = emotion_list[predicted_id]

print("\nSpeech Emotion")
print("--------------")

for emotion, probability in zip(
    emotion_list,
    probabilities.cpu().numpy()
):
    print(f"{emotion}: {probability:.4f}")

print(f"\nDominant: {predicted_emotion}")