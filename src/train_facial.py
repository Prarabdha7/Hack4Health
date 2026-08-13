import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from config import IMAGE_DIR, MODEL_DIR, EMOTION_CLASSES
from facial_model import FacialCNN, FacialDataset

SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 6

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

files, labels = [], []
for idx, emotion in enumerate(EMOTION_CLASSES):
    folder = IMAGE_DIR / emotion
    if not folder.is_dir():
        continue
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            files.append(str(path)); labels.append(idx)

if not files:
    raise FileNotFoundError(f"No facial images found under {IMAGE_DIR}")

# Train/validation/test split: test remains untouched until the end.
train_files, temp_files, train_labels, temp_labels = train_test_split(
    files, labels, test_size=0.20, random_state=SEED, stratify=labels
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)

print(f"Train: {len(train_files)} | Validation: {len(val_files)} | Test: {len(test_files)}")

train_loader = DataLoader(FacialDataset(train_files, train_labels, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(FacialDataset(val_files, val_labels, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(FacialDataset(test_files, test_labels, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
weights = compute_class_weight("balanced", classes=np.arange(len(EMOTION_CLASSES)), y=train_labels)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.05)
model = FacialCNN(len(EMOTION_CLASSES)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


def run_eval(loader):
    model.eval(); ys, ps, total = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); total += criterion(logits, y).item() * len(y)
            ys.extend(y.cpu().numpy()); ps.extend(logits.argmax(1).cpu().numpy())
    return total / len(ys), accuracy_score(ys, ps), f1_score(ys, ps, average="macro"), ys, ps

best_f1, best_epoch, stale = -1, -1, 0
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for epoch in range(1, EPOCHS + 1):
    model.train(); running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step(); running += loss.item() * len(y)
    scheduler.step()
    val_loss, val_acc, val_f1, _, _ = run_eval(val_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} train_loss={running/len(train_files):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1, best_epoch, stale = val_f1, epoch, 0
        torch.save({"model_state_dict": model.state_dict(), "task": "7_class_emotion", "classes": EMOTION_CLASSES, "seed": SEED}, MODEL_DIR / "facial_emotion_cnn.pth")
    else:
        stale += 1
        if stale >= PATIENCE:
            print("Early stopping."); break

checkpoint = torch.load(MODEL_DIR / "facial_emotion_cnn.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
test_loss, test_acc, test_f1, y_true, y_pred = run_eval(test_loader)
print("\nFINAL HELD-OUT FACIAL EMOTION TEST")
print(f"Best validation epoch: {best_epoch} | Test accuracy: {test_acc:.4f} | Test macro F1: {test_f1:.4f}")
print(classification_report(y_true, y_pred, labels=range(7), target_names=EMOTION_CLASSES, zero_division=0))
with open(MODEL_DIR / "facial_training_split.json", "w") as f:
    json.dump({"seed": SEED, "train": len(train_files), "validation": len(val_files), "test": len(test_files), "best_epoch": best_epoch}, f, indent=2)
