import joblib
import librosa
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch import nn


FACIAL_MODEL_PATH = "models/facial_stress_cnn.pth"
SPEECH_MODEL_PATH = "models/speech_stress_svm.joblib"
TABULAR_MODEL_PATH = "models/best_tabular_classifier.pkl"

STRESS_CLASSES = [
    "Healthy",
    "Mild_Stress",
    "Moderate_Stress",
    "Severe_Stress"
]

TABULAR_FEATURES = [
    "Sleep_Quality",
    "Social_Engagement",
    "Daily_App_Usage_Min",
    "Typing_Speed_WPM",
    "Session_Frequency",
    "Idle_Time_Min",
    "Facial_Emotion_Variance",
    "Eye_Blink_Rate",
    "Smile_Intensity",
    "Head_Motion_Index",
    "MFCC_Mean",
    "MFCC_Variance",
    "Pitch_Mean",
    "Speech_Rate",
    "Heart_Rate_BPM",
    "HRV_Index",
    "Skin_Temperature",
    "GSR_Level"
]


class FacialCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_models():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        FACIAL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    facial_model = FacialCNN(
        len(STRESS_CLASSES)
    ).to(device)

    facial_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    facial_model.eval()

    speech_package = joblib.load(
        SPEECH_MODEL_PATH
    )

    speech_model = speech_package["model"]

    tabular_model = joblib.load(
        TABULAR_MODEL_PATH
    )

    return (
        facial_model,
        speech_model,
        tabular_model,
        device
    )


def get_facial_probabilities(
    model,
    device,
    image_path
):
    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (48, 48)
    )

    image = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    return probabilities.cpu().numpy()[0]


def extract_speech_features(path):
    audio, sr = librosa.load(
        path,
        sr=16000
    )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    if len(audio) < 1024:
        audio = np.pad(
            audio,
            (0, 1024 - len(audio))
        )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=1024,
        hop_length=256
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256
    )

    zero_crossing = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=256
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256
    )

    feature_groups = [
        mfcc,
        delta,
        delta2,
        chroma,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing,
        rms
    ]

    features = []

    for feature in feature_groups:
        features.extend(
            np.mean(
                feature,
                axis=1
            )
        )

        features.extend(
            np.std(
                feature,
                axis=1
            )
        )

    return np.array(
        features,
        dtype=np.float32
    )


def get_speech_probabilities(
    model,
    audio_path
):
    features = extract_speech_features(
        audio_path
    ).reshape(1, -1)

    probabilities = model.predict_proba(
        features
    )[0]

    return probabilities


def get_tabular_probabilities(
    model,
    tabular_values
):
    values = pd.DataFrame(
        [tabular_values],
        columns=TABULAR_FEATURES
    )

    probabilities = model.predict_proba(
        values
    )[0]

    return probabilities


def fuse_predictions(
    facial_probabilities,
    speech_probabilities,
    tabular_probabilities
):
    weights = np.array([
        0.43,
        0.42,
        0.15
    ])

    fused = (
        weights[0] * facial_probabilities
        + weights[1] * speech_probabilities
        + weights[2] * tabular_probabilities
    )

    fused = fused / fused.sum()

    return fused


def predict(
    image_path,
    audio_path,
    tabular_values
):
    (
        facial_model,
        speech_model,
        tabular_model,
        device
    ) = load_models()

    facial_probabilities = get_facial_probabilities(
        facial_model,
        device,
        image_path
    )

    speech_probabilities = get_speech_probabilities(
        speech_model,
        audio_path
    )

    tabular_probabilities = get_tabular_probabilities(
        tabular_model,
        tabular_values
    )

    fused = fuse_predictions(
        facial_probabilities,
        speech_probabilities,
        tabular_probabilities
    )

    prediction = STRESS_CLASSES[
        np.argmax(fused)
    ]

    print("\nFacial stress probabilities:")

    for stress, probability in zip(
        STRESS_CLASSES,
        facial_probabilities
    ):
        print(
            f"{stress}: {probability:.4f}"
        )

    print("\nSpeech stress probabilities:")

    for stress, probability in zip(
        STRESS_CLASSES,
        speech_probabilities
    ):
        print(
            f"{stress}: {probability:.4f}"
        )

    print("\nTabular stress probabilities:")

    for stress, probability in zip(
        STRESS_CLASSES,
        tabular_probabilities
    ):
        print(
            f"{stress}: {probability:.4f}"
        )

    print("\nFused probabilities:")

    for stress, probability in zip(
        STRESS_CLASSES,
        fused
    ):
        print(
            f"{stress}: {probability:.4f}"
        )

    print(
        "\nFinal Prediction:",
        prediction
    )

    return prediction