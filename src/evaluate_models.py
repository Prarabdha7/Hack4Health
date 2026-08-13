"""Leakage-aware evaluation for each modality.
Facial: 7-class emotion model evaluated once on held-out test, then mapped to stress.
Speech: actor-independent held-out test.
Tabular: fixed stratified split plus required classification/regression metrics.
"""
import os, json
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score, explained_variance_score)
from config import *
from facial_model import FacialCNN, emotion_probabilities_to_stress
from fusion_model import extract_speech_features


def classification_metrics(y, pred, prob, classes=STRESS_CLASSES):
    result = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        "report": classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0),
    }
    try: result["roc_auc_ovr_macro"] = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError: result["roc_auc_ovr_macro"] = None
    return result


def evaluate_facial():
    files, labels = [], []
    for i, emotion in enumerate(EMOTION_CLASSES):
        folder = IMAGE_DIR / emotion
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}: files.append(str(p)); labels.append(i)
    train, temp, y_train, y_temp = train_test_split(files, labels, test_size=.20, random_state=42, stratify=labels)
    _, test, _, y_test = train_test_split(temp, y_temp, test_size=.50, random_state=42, stratify=y_temp)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    path = FACIAL_MODEL_PATH if FACIAL_MODEL_PATH.exists() else LEGACY_FACIAL_MODEL_PATH
    ck = torch.load(path, map_location=dev, weights_only=False)
    n = len(ck.get("classes", STRESS_CLASSES))
    model = FacialCNN(n).to(dev); model.load_state_dict(ck["model_state_dict"]); model.eval()
    raw, stress_p, stress_y = [], [], []
    with torch.no_grad():
        for p in test:
            arr = np.asarray(Image.open(p).convert("L").resize((48,48)), dtype=np.float32)/255
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(dev)
            probs = torch.softmax(model(x),1).cpu().numpy()[0]
            if len(probs) == 7: sp = emotion_probabilities_to_stress(probs)
            else: sp = probs
            raw.append(probs); stress_p.append(sp); stress_y.append(STRESS_CLASSES.index(EMOTION_TO_STRESS[EMOTION_CLASSES[y_test[len(stress_y)]]]))
    pred = np.argmax(stress_p,1)
    return classification_metrics(stress_y, pred, np.asarray(stress_p)), classification_metrics(y_test, np.argmax(raw,1), np.asarray(raw), EMOTION_CLASSES)


def evaluate_speech():
    files, labels, actors = [], [], []
    for actor_dir in sorted(AUDIO_DIR.iterdir()) if AUDIO_DIR.exists() else []:
        if not actor_dir.is_dir(): continue
        for p in sorted(actor_dir.glob("*.wav")):
            parts=p.stem.split("-")
            if len(parts)!=7 or parts[2] not in SPEECH_EMOTIONS: continue
            emotion=SPEECH_EMOTIONS[parts[2]]; files.append(str(p)); labels.append(STRESS_CLASSES.index(SPEECH_TO_STRESS[emotion])); actors.append(actor_dir.name)
    actor_names=sorted(set(actors)); _, test_actors=train_test_split(actor_names,test_size=.20,random_state=42)
    test=set(test_actors); package=joblib.load(SPEECH_MODEL_PATH); model=package["model"]
    y=[]; probs=[]
    for p,label,actor in zip(files,labels,actors):
        if actor in test:
            y.append(label); probs.append(model.predict_proba(extract_speech_features(p).reshape(1,-1))[0])
    probs=np.asarray(probs); pred=np.argmax(probs,1)
    return classification_metrics(y,pred,probs)


def evaluate_tabular():
    df=pd.read_csv(TABULAR_PATH); X=df[TABULAR_FEATURES]; y=df["Mental_Health_Status"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    clf=joblib.load(TABULAR_MODEL_PATH); prob=clf.predict_proba(Xte); pred=clf.predict(Xte)
    cls={s:i for i,s in enumerate(STRESS_CLASSES)}
    yid=[cls[v] for v in yte]; pid=[cls[v] for v in pred]
    classification=classification_metrics(yid,pid,prob)
    reg=joblib.load(TABULAR_REGRESSOR_PATH) if TABULAR_REGRESSOR_PATH.exists() else None
    regression={}
    if reg:
        targets=df[REGRESSION_TARGETS]; _, Xr, _, yr=train_test_split(X,targets,test_size=.20,random_state=42)
        rp=reg.predict(Xr)
        for i,t in enumerate(REGRESSION_TARGETS):
            a=yr.iloc[:,i]; p=rp[:,i]; mse=mean_squared_error(a,p)
            regression[t]={"MAE":mean_absolute_error(a,p),"MSE":mse,"RMSE":mse**.5,"R2":r2_score(a,p),"Explained_Variance":explained_variance_score(a,p)}
    return classification,regression


def main():
    out={}
    print("\nHACK4HEALTH LEAKAGE-AWARE EVALUATION\n")
    if FACIAL_MODEL_PATH.exists() or LEGACY_FACIAL_MODEL_PATH.exists(): out["facial"]=evaluate_facial(); print("Facial evaluation complete")
    if SPEECH_MODEL_PATH.exists() and AUDIO_DIR.exists(): out["speech"]=evaluate_speech(); print("Speech evaluation complete")
    if TABULAR_MODEL_PATH.exists() and TABULAR_PATH.exists(): out["tabular"]=evaluate_tabular(); print("Tabular evaluation complete")
    REPORT_DIR.mkdir(parents=True,exist_ok=True); (REPORT_DIR/"evaluation.json").write_text(json.dumps(out,indent=2,default=float))
    print(f"Saved {REPORT_DIR/'evaluation.json'}")
if __name__=="__main__": main()
