import os
import numpy as np
import pandas as pd
import joblib
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from fusion_model import (
    FacialCNN,
    extract_speech_features,
    STRESS_CLASSES,
    TABULAR_FEATURES,
    FACIAL_MODEL_PATH,
    SPEECH_MODEL_PATH,
    TABULAR_MODEL_PATH
)


IMAGE_DIR = "data/images/Extracted_images"
AUDIO_DIR = "data/audio/audio_speech_actors_01-24"
DATA_PATH = "data/mental_health_multimodal.csv"

EMOTION_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
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

EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

SPEECH_TO_STRESS = {
    "neutral": "Healthy",
    "calm": "Healthy",
    "happy": "Healthy",
    "sad": "Mild_Stress",
    "surprised": "Mild_Stress",
    "fearful": "Moderate_Stress",
    "angry": "Moderate_Stress",
    "disgust": "Severe_Stress"
}


def print_results(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(STRESS_CLASSES)),
            target_names=STRESS_CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(STRESS_CLASSES))
        )
    )

    return accuracy, macro_f1, weighted_f1


def evaluate_facial():
    files = []
    labels = []

    stress_to_id = {
        stress: i
        for i, stress in enumerate(STRESS_CLASSES)
    }

    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(
            IMAGE_DIR,
            emotion
        )

        if not os.path.isdir(emotion_path):
            continue

        stress = EMOTION_TO_STRESS[emotion]
        label = stress_to_id[stress]

        for filename in sorted(
            os.listdir(emotion_path)
        ):
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

    _, test_files, _, test_labels = train_test_split(
        files,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

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

    model = FacialCNN(
        len(STRESS_CLASSES)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions = []

    for path in test_files:
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

        tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)

        prediction = output.argmax(
            dim=1
        ).item()

        predictions.append(prediction)

    return print_results(
        "FACIAL CNN",
        test_labels,
        predictions
    )


def evaluate_speech():
    files = []
    labels = []

    stress_to_id = {
        stress: i
        for i, stress in enumerate(STRESS_CLASSES)
    }

    for actor in sorted(
        os.listdir(AUDIO_DIR)
    ):
        actor_path = os.path.join(
            AUDIO_DIR,
            actor
        )

        if not os.path.isdir(actor_path):
            continue

        for filename in sorted(
            os.listdir(actor_path)
        ):
            if not filename.lower().endswith(
                ".wav"
            ):
                continue

            parts = filename.replace(
                ".wav",
                ""
            ).split("-")

            if len(parts) != 7:
                continue

            emotion = EMOTION_NAMES.get(
                parts[2]
            )

            if emotion is None:
                continue

            stress = SPEECH_TO_STRESS[
                emotion
            ]

            files.append(
                os.path.join(
                    actor_path,
                    filename
                )
            )

            labels.append(
                stress_to_id[stress]
            )

    actors = sorted(
        set(
            os.path.basename(
                os.path.dirname(path)
            )
            for path in files
        )
    )

    _, test_actors = train_test_split(
        actors,
        test_size=0.2,
        random_state=42
    )

    test_actors = set(test_actors)

    test_files = []
    test_labels = []

    for path, label in zip(
        files,
        labels
    ):
        actor = os.path.basename(
            os.path.dirname(path)
        )

        if actor in test_actors:
            test_files.append(path)
            test_labels.append(label)

    package = joblib.load(
        SPEECH_MODEL_PATH
    )

    model = package["model"]

    predictions = []

    for path in test_files:
        features = extract_speech_features(
            path
        ).reshape(
            1,
            -1
        )

        prediction = model.predict(
            features
        )[0]

        predictions.append(prediction)

    return print_results(
        "SPEECH SVM",
        test_labels,
        predictions
    )


def evaluate_tabular():
    df = pd.read_csv(
        DATA_PATH
    )

    X = df[TABULAR_FEATURES]
    y = df["Mental_Health_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(
        TABULAR_MODEL_PATH
    )

    predictions = model.predict(
        X_test
    )

    print("\n" + "=" * 65)
    print("TABULAR RANDOM FOREST")
    print("=" * 65)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=STRESS_CLASSES,
            target_names=STRESS_CLASSES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=STRESS_CLASSES
        )
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1
    )


print("\n" + "=" * 65)
print("HACK4HEALTH MODEL EVALUATION")
print("=" * 65)

facial_results = evaluate_facial()
speech_results = evaluate_speech()
tabular_results = evaluate_tabular()

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    f"Facial CNN       Accuracy: {facial_results[0]:.4f}  "
    f"Macro F1: {facial_results[1]:.4f}"
)

print(
    f"Speech SVM       Accuracy: {speech_results[0]:.4f}  "
    f"Macro F1: {speech_results[1]:.4f}"
)

print(
    f"Tabular RF       Accuracy: {tabular_results[0]:.4f}  "
    f"Macro F1: {tabular_results[1]:.4f}"
)

print("\nTrue multimodal fusion evaluation requires")
print("aligned image, audio and tabular samples.")