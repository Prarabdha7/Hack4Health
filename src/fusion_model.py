import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import (
    FACIAL_MODEL_PATH, LEGACY_FACIAL_MODEL_PATH, SPEECH_MODEL_PATH,
    TABULAR_MODEL_PATH, STRESS_CLASSES, TABULAR_FEATURES,
    EMOTION_CLASSES, EMOTION_TO_STRESS,
)
from facial_model import FacialCNN, emotion_probabilities_to_stress


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    dev = device()
    facial_path = FACIAL_MODEL_PATH if FACIAL_MODEL_PATH.exists() else LEGACY_FACIAL_MODEL_PATH
    checkpoint = torch.load(facial_path, map_location=dev, weights_only=False)
    n = len(checkpoint.get("classes", EMOTION_CLASSES if checkpoint.get("task") == "7_class_emotion" else STRESS_CLASSES))
    facial = FacialCNN(n).to(dev)
    facial.load_state_dict(checkpoint["model_state_dict"])
    facial.eval()
    speech_pkg = joblib.load(SPEECH_MODEL_PATH)
    tabular = joblib.load(TABULAR_MODEL_PATH)
    return facial, speech_pkg, tabular, dev, checkpoint


def get_facial_probabilities(model, dev, image_path):
    image = Image.open(image_path).convert("L").resize((48, 48))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    # New model is 7-class emotion; old checkpoint is 4-class stress.
    if len(p) == len(EMOTION_CLASSES):
        return emotion_probabilities_to_stress(p), p
    return p / p.sum(), p


def extract_speech_features(path):
    audio, sr = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < 1024: audio = np.pad(audio, (0, 1024 - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=1024, hop_length=256)
    groups = [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2),
              librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=1024, hop_length=256),
              librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=1024, hop_length=256),
              librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=1024, hop_length=256),
              librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=1024, hop_length=256),
              librosa.feature.zero_crossing_rate(audio, hop_length=256),
              librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)]
    features = []
    for g in groups:
        features.extend(np.mean(g, axis=1)); features.extend(np.std(g, axis=1))
    return np.asarray(features, dtype=np.float32)


def get_speech_probabilities(package, audio_path):
    model = package["model"] if isinstance(package, dict) else package
    p = model.predict_proba(extract_speech_features(audio_path).reshape(1, -1))[0]
    return np.asarray(p, dtype=float)


def get_tabular_probabilities(model, values):
    X = pd.DataFrame([values], columns=TABULAR_FEATURES)
    return np.asarray(model.predict_proba(X)[0], dtype=float)


def fuse_predictions(facial, speech, tabular, weights=(0.45, 0.40, 0.15)):
    # Weights are intentionally explicit and easy to recalibrate later on aligned validation data.
    p = np.vstack([facial, speech, tabular])
    w = np.asarray(weights, dtype=float); w /= w.sum()
    fused = (p * w[:, None]).sum(axis=0)
    return fused / fused.sum()


def predict(image_path, audio_path, tabular_values):
    facial_model, speech_package, tabular_model, dev, checkpoint = load_models()
    fp, raw_face = get_facial_probabilities(facial_model, dev, image_path)
    sp = get_speech_probabilities(speech_package, audio_path)
    tp = get_tabular_probabilities(tabular_model, tabular_values)
    fused = fuse_predictions(fp, sp, tp)
    return {
        "prediction": STRESS_CLASSES[int(np.argmax(fused))],
        "facial": fp, "speech": sp, "tabular": tp, "fused": fused,
        "raw_facial": raw_face, "checkpoint": checkpoint,
    }
