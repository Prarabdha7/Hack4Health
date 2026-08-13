import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

from config import (
    TABULAR_PATH,
    TABULAR_FEATURES,
    REGRESSION_TARGETS,
    TABULAR_REGRESSOR_PATH,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TABULAR_PATH)

X = df[TABULAR_FEATURES]
y = df[REGRESSION_TARGETS]

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ============================================================
# COMPACT REGRESSION MODELS
# ============================================================

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    ),
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

best = None
best_name = None
best_score = float("inf")

for name, base_model in models.items():

    model = MultiOutputRegressor(base_model)

    model.fit(Xtr, ytr)

    predictions = model.predict(Xte)

    mean_mae = sum(
        mean_absolute_error(
            yte.iloc[:, i],
            predictions[:, i],
        )
        for i in range(len(REGRESSION_TARGETS))
    ) / len(REGRESSION_TARGETS)

    print(f"\n{name} mean MAE={mean_mae:.4f}")

    for i, target in enumerate(REGRESSION_TARGETS):

        actual = yte.iloc[:, i]
        predicted = predictions[:, i]

        mse = mean_squared_error(actual, predicted)

        print(
            f"{target}: "
            f"MAE={mean_absolute_error(actual, predicted):.4f} "
            f"MSE={mse:.4f} "
            f"RMSE={mse ** 0.5:.4f} "
            f"R2={r2_score(actual, predicted):.4f} "
            f"ExplainedVariance="
            f"{explained_variance_score(actual, predicted):.4f}"
        )

    if mean_mae < best_score:
        best_score = mean_mae
        best = model
        best_name = name


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best,
    TABULAR_REGRESSOR_PATH,
    compress=3,
)

print(f"\nBest regression model: {best_name}")
print(f"Saved: {TABULAR_REGRESSOR_PATH}")