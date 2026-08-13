import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


AUDIO_DIR = "data/audio/audio_speech_actors_01-24"

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

STRESS_CLASSES = [
    "Healthy",
    "Mild_Stress",
    "Moderate_Stress",
    "Severe_Stress"
]

EMOTION_TO_STRESS = {
    "neutral": "Healthy",
    "calm": "Healthy",
    "happy": "Healthy",
    "sad": "Mild_Stress",
    "surprised": "Mild_Stress",
    "fearful": "Moderate_Stress",
    "angry": "Moderate_Stress",
    "disgust": "Severe_Stress"
}

stress_to_id = {
    stress: i
    for i, stress in enumerate(STRESS_CLASSES)
}

files = []
labels = []

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

        stress = EMOTION_TO_STRESS[
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

train_actors, test_actors = train_test_split(
    actors,
    test_size=0.2,
    random_state=42
)

train_actors = set(
    train_actors
)

test_actors = set(
    test_actors
)

train_files = []
test_files = []
train_labels = []
test_labels = []

for path, label in zip(
    files,
    labels
):
    actor = os.path.basename(
        os.path.dirname(path)
    )

    if actor in train_actors:
        train_files.append(path)
        train_labels.append(label)
    else:
        test_files.append(path)
        test_labels.append(label)

print(
    "Training actors:",
    sorted(train_actors)
)

print(
    "Testing actors:",
    sorted(test_actors)
)

print(
    "Training files:",
    len(train_files)
)

print(
    "Testing files:",
    len(test_files)
)

print(
    "Training class counts:",
    np.bincount(
        train_labels,
        minlength=len(STRESS_CLASSES)
    ).tolist()
)

print(
    "Testing class counts:",
    np.bincount(
        test_labels,
        minlength=len(STRESS_CLASSES)
    ).tolist()
)


def extract_features(path):
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


print("\nExtracting training features...")

train_features = np.array(
    [
        extract_features(path)
        for path in train_files
    ],
    dtype=np.float32
)

print(
    "Training feature shape:",
    train_features.shape
)

print("\nExtracting testing features...")

test_features = np.array(
    [
        extract_features(path)
        for path in test_files
    ],
    dtype=np.float32
)

print(
    "Testing feature shape:",
    test_features.shape
)

print("\nTraining SVM...")

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            )
        )
    ]
)

model.fit(
    train_features,
    train_labels
)

predictions = model.predict(
    test_features
)

accuracy = accuracy_score(
    test_labels,
    predictions
)

macro_f1 = f1_score(
    test_labels,
    predictions,
    average="macro"
)

print("\nSpeech Stress SVM Results:")

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Macro F1: {macro_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        test_labels,
        predictions,
        target_names=STRESS_CLASSES,
        zero_division=0
    )
)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "classes": STRESS_CLASSES
    },
    "models/speech_stress_svm.joblib"
)

print(
    "\nModel saved to models/speech_stress_svm.joblib"
)