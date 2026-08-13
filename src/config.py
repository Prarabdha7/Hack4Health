from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

IMAGE_DIR = DATA_DIR / "images" / "Extracted_images"
AUDIO_DIR = DATA_DIR / "audio" / "audio_speech_actors_01-24"
TABULAR_PATH = DATA_DIR / "mental_health_multimodal.csv"

STRESS_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_TO_STRESS = {
    "Happy": "Healthy",
    "Neutral": "Healthy",
    "Sad": "Mild_Stress",
    "Surprise": "Mild_Stress",
    "Fear": "Moderate_Stress",
    "Disgust": "Moderate_Stress",
    "Angry": "Severe_Stress",
}

SPEECH_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
SPEECH_TO_STRESS = {
    "neutral": "Healthy", "calm": "Healthy", "happy": "Healthy",
    "sad": "Mild_Stress", "surprised": "Mild_Stress",
    "fearful": "Moderate_Stress", "angry": "Moderate_Stress",
    "disgust": "Severe_Stress",
}

TABULAR_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", "Typing_Speed_WPM",
    "Session_Frequency", "Idle_Time_Min", "Facial_Emotion_Variance", "Eye_Blink_Rate",
    "Smile_Intensity", "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean",
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature", "GSR_Level",
]
REGRESSION_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

FACIAL_MODEL_PATH = MODEL_DIR / "facial_emotion_cnn.pth"
LEGACY_FACIAL_MODEL_PATH = MODEL_DIR / "facial_stress_cnn.pth"
SPEECH_MODEL_PATH = MODEL_DIR / "speech_stress_svm.joblib"
TABULAR_MODEL_PATH = MODEL_DIR / "best_tabular_classifier.pkl"
TABULAR_REGRESSOR_PATH = MODEL_DIR / "tabular_regressor.pkl"

for path in (MODEL_DIR, REPORT_DIR):
    path.mkdir(parents=True, exist_ok=True)
