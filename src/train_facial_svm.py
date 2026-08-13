import os
import cv2
import joblib
import numpy as np

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

IMAGE_DIR = "data/images/Extracted_images"

classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

class_to_id = {
    class_name: i
    for i, class_name in enumerate(classes)
}

files = []
labels = []

for class_name in classes:
    class_path = os.path.join(
        IMAGE_DIR,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    for filename in sorted(
        os.listdir(class_path)
    ):
        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        files.append(
            os.path.join(
                class_path,
                filename
            )
        )

        labels.append(
            class_to_id[class_name]
        )

train_files, test_files, train_labels, test_labels = train_test_split(
    files,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(
    "Training files:",
    len(train_files)
)

print(
    "Testing files:",
    len(test_files)
)

def extract_features(path):
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (48, 48)
    )

    image_float = image.astype(
        np.float32
    ) / 255.0

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(6, 6),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    lbp = local_binary_pattern(
        image,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(11),
        range=(0, 10),
        density=True
    )

    small_image = cv2.resize(
        image_float,
        (24, 24)
    ).flatten()

    features = np.concatenate(
        [
            hog_features,
            lbp_histogram,
            small_image
        ]
    )

    return features.astype(
        np.float32
    )

print(
    "\nExtracting training features..."
)

X_train = np.array(
    [
        extract_features(path)
        for path in train_files
    ],
    dtype=np.float32
)

print(
    "Training feature shape:",
    X_train.shape
)

print(
    "\nExtracting testing features..."
)

X_test = np.array(
    [
        extract_features(path)
        for path in test_files
    ],
    dtype=np.float32
)

print(
    "Testing feature shape:",
    X_test.shape
)

y_train = np.array(
    train_labels
)

y_test = np.array(
    test_labels
)

print(
    "\nTraining facial SVM..."
)

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                dual="auto",
                random_state=42
            )
        )
    ]
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print(
    "\nFacial SVM Results:"
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Macro F1: {macro_f1:.4f}"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=classes,
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
        "classes": classes
    },
    "models/facial_emotion_svm.joblib"
)

np.save(
    "models/facial_svm_test_features.npy",
    X_test
)

np.save(
    "models/facial_svm_test_labels.npy",
    y_test
)

print(
    "\nModel saved to models/facial_emotion_svm.joblib"
)