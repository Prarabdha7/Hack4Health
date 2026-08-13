import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

DATA_PATH = "data/mental_health_multimodal.csv"

df = pd.read_csv(DATA_PATH)

features = [
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

targets = [
    "Depression_Score",
    "Anxiety_Score",
    "Stress_Score"
]

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("=" * 60)
print("TABULAR REGRESSION RESULTS")
print("=" * 60)

for i, target in enumerate(targets):
    actual = y_test.iloc[:, i]
    predicted = predictions[:, i]

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)
    ev = explained_variance_score(actual, predicted)

    print(f"\n{target}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Explained Variance: {ev:.4f}")

joblib.dump(model, "models/tabular_regressor.pkl")

print("\n" + "=" * 60)
print("Model saved to models/tabular_regressor.pkl")