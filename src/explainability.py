"""Lightweight, dependency-free explainability utilities for the demo."""
import numpy as np
import torch
from config import STRESS_CLASSES, TABULAR_FEATURES


def modality_contributions(facial, speech, tabular, weights=(.45,.40,.15)):
    probs=np.vstack([facial,speech,tabular]); w=np.asarray(weights); fused=(probs*w[:,None]).sum(0); target=int(np.argmax(fused))
    raw=np.abs(probs[:,target] - 0.25) * w
    raw=raw/raw.sum() if raw.sum() else np.ones(3)/3
    return {"facial":float(raw[0]),"speech":float(raw[1]),"tabular":float(raw[2]),"target_class":STRESS_CLASSES[target]}


def tabular_importance(model):
    est=model
    if hasattr(model,"named_steps"): est=model.named_steps.get("model") or model.named_steps.get("classifier") or model.named_steps.get("regressor")
    if hasattr(est,"feature_importances_"): vals=np.asarray(est.feature_importances_,dtype=float)
    elif hasattr(est,"coef_"): vals=np.mean(np.abs(est.coef_),axis=0); vals=vals/vals.sum()
    else: return []
    vals=vals/vals.sum() if vals.sum() else vals
    return sorted(zip(TABULAR_FEATURES,vals.tolist()), key=lambda x:x[1], reverse=True)


def facial_saliency(model, image_tensor, class_index, device):
    x=image_tensor.clone().detach().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True); score=model(x)[0,class_index]; score.backward()
    sal=x.grad.detach().abs().squeeze().cpu().numpy()
    sal=(sal-sal.min())/(sal.max()-sal.min()+1e-8)
    return sal
