# Hack4Health — Explainable Multimodal Mental-Health Assessment

An explainable multimodal machine-learning framework for four-class mental-health status assessment using **facial expression, speech emotion, and 18 behavioral/acoustic/physiological features**. The project also includes a separate symptom-severity regression head for Depression, Anxiety, and Stress scores.

> **Research/demo system only. It is not a clinical diagnostic tool or medical advice.**

## What the project implements

### Objective 1 — Multimodal classification
The three modalities produce class probabilities for:

- Healthy
- Mild Stress
- Moderate Stress
- Severe Stress

The facial pipeline now trains a **7-class emotion model first** and maps emotion probabilities to the required four stress classes. This preserves the original emotion information instead of collapsing Fear, Anger, Disgust, etc. before the model learns their visual distinctions.

The final demo combines the modality probabilities using an explicit weighted fusion. The weights are deliberately kept configurable so they can be recalibrated if participant-aligned multimodal validation data becomes available.

### Objective 2 — Severity estimation
The supplied numerical dataset contains Depression, Anxiety, and Stress targets. The repository provides a multi-output regression head and reports MAE, MSE, RMSE, R² and Explained Variance.

The supplied documentation does **not provide participant IDs linking the 28,709 facial images and 1,440 speech files to the 4,000 numerical rows**. Therefore the project does not fabricate a participant-aligned multimodal regression score. The demo exposes the regression head separately and clearly labels it.

### Objective 3 — Explainability
The demo reports:

- modality-level contribution to the fused prediction,
- tabular feature importance when supported by the fitted estimator,
- facial probability distribution,
- speech probability distribution,
- fused probability distribution.

## Data quality and Fear/Anger handling

The provided facial dataset contains seven emotion folders. The official dataset description maps **Fear → Moderate Stress** and **Angry → Severe Stress**, so label ambiguity between these emotions can directly affect the stress classifier.

`src/data_quality.py` performs:

- per-class image counts,
- corrupted-image detection,
- exact duplicate detection using SHA-256,
- perceptual duplicate detection using a compact pHash implementation.

`src/augment_facial.py` creates a reproducible augmented **training-only** dataset. Validation and test images are split before augmentation so augmented copies cannot leak into evaluation.

The training script also performs on-the-fly augmentation, which is generally preferable for training because it does not require storing tens of thousands of generated images.

## Dataset layout

Place the supplied data at:

```text
data/
├── images/
│   └── Extracted_images/
│       ├── Angry/
│       ├── Disgust/
│       ├── Fear/
│       ├── Happy/
│       ├── Neutral/
│       ├── Sad/
│       └── Surprise/
├── audio/
│   └── audio_speech_actors_01-24/
│       ├── Actor_01/
│       └── ...
└── mental_health_multimodal.csv
```

The dataset documentation describes 28,709 facial images, 1,440 speech files, and 4,000 numerical rows.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, PyTorch should expose the MPS backend. Verify with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Recommended workflow

### 1. Audit the data

```bash
python src/data_quality.py --audio
```

This creates `reports/data_quality_report.json`.

### 2. Generate a reproducible augmented training set

```bash
python src/augment_facial.py --copies 1
```

Do not put the generated dataset in the Git repository unless the organizers explicitly require the generated files and your repository/storage limits allow it. The manifest and deterministic generation code are the reproducible artifact.

### 3. Train the facial model

```bash
python src/train_facial.py
```

The script creates a train/validation/test split, augments only training samples, uses class-weighted loss, early stopping and selects the checkpoint using validation macro F1. The held-out test set is evaluated only after model selection.

Output:

```text
models/facial_emotion_cnn.pth
```

### 4. Train speech model

```bash
python src/train_speech.py
```

The speech split is actor-independent to avoid placing the same speaker in both training and testing.

### 5. Train numerical models

```bash
python src/tabular_classification.py
python src/tabular_regression.py
```

### 6. Evaluate

```bash
python src/evaluate_models.py
```

The evaluator reports Accuracy, Precision, Recall, Macro F1, Weighted F1, ROC-AUC and confusion matrices for classification, plus MAE, MSE, RMSE, R² and Explained Variance for regression.

### 7. Run the multimodal CLI

```bash
python src/run_fusion.py IMAGE_PATH AUDIO_PATH \
  3 3 180 40 15 120 0.5 14 0.2 0.2 10 5 180 4 75 50 34 1
```

### 8. Launch the demo frontend

```bash
streamlit run app/streamlit_app.py
```

The interface accepts a facial image, a WAV recording and the 18 numerical features, then displays modality predictions, fused prediction and explainability information.

## Evaluation integrity

The repository intentionally avoids test-set model selection and fabricated multimodal metrics. A genuine multimodal evaluation requires participant-level alignment across image, audio and tabular records. If the organizers provide such an alignment key, the evaluation script should be extended to use it rather than inventing pairings.

## Repository hygiene

Large raw datasets and generated model artifacts are ignored by default. If organizers explicitly require model weights or an augmented dataset to be committed, use Git LFS or the storage mechanism specified by the organizers instead of forcing large binaries into ordinary Git history.
