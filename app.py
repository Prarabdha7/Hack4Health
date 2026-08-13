import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import streamlit as st
from PIL import Image

from config import STRESS_CLASSES, TABULAR_FEATURES, TABULAR_REGRESSOR_PATH
from fusion_model import load_models, get_facial_probabilities, get_speech_probabilities, get_tabular_probabilities, fuse_predictions
from explainability import modality_contributions, tabular_importance
import joblib

st.set_page_config(page_title="Hack4Health | Multimodal Assessment", page_icon="🧠", layout="wide")
st.title("Hack4Health")
st.caption("Explainable multimodal mental-health assessment demo")
st.warning("Research/demo system only. This is not a clinical diagnosis or medical advice.")

with st.sidebar:
    st.header("Input data")
    image_file=st.file_uploader("Facial image", type=["png","jpg","jpeg"])
    audio_file=st.file_uploader("Speech recording", type=["wav"])
    st.subheader("Behavioral & physiological features")
    defaults={
        "Sleep_Quality":3.0,"Social_Engagement":3.0,"Daily_App_Usage_Min":180.0,"Typing_Speed_WPM":40.0,
        "Session_Frequency":15.0,"Idle_Time_Min":120.0,"Facial_Emotion_Variance":0.5,"Eye_Blink_Rate":14.0,
        "Smile_Intensity":0.2,"Head_Motion_Index":0.2,"MFCC_Mean":10.0,"MFCC_Variance":5.0,
        "Pitch_Mean":180.0,"Speech_Rate":4.0,"Heart_Rate_BPM":75.0,"HRV_Index":50.0,"Skin_Temperature":34.0,"GSR_Level":1.0}
    vals=[]
    for f in TABULAR_FEATURES:
        vals.append(st.number_input(f, value=float(defaults[f])))
    run=st.button("Assess", type="primary", use_container_width=True)

if run:
    if not image_file or not audio_file:
        st.error("Please provide both a facial image and a WAV speech recording.")
        st.stop()
    tmp=Path(".streamlit_tmp"); tmp.mkdir(exist_ok=True)
    ip=tmp/"input_face.png"; ap=tmp/"input_audio.wav"; image_file.seek(0); ip.write_bytes(image_file.read()); audio_file.seek(0); ap.write_bytes(audio_file.read())
    try:
        facial_model, speech_pkg, tab_model, dev, ck=load_models()
        fp, raw=get_facial_probabilities(facial_model,dev,str(ip)); sp=get_speech_probabilities(speech_pkg,str(ap)); tp=get_tabular_probabilities(tab_model,vals); fused=fuse_predictions(fp,sp,tp)
    except Exception as e:
        st.error(f"Model loading/inference failed: {e}")
        st.stop()
    pred=STRESS_CLASSES[int(np.argmax(fused))]
    st.subheader("Assessment")
    st.metric("Predicted mental-health status", pred)
    c1,c2,c3=st.columns(3)
    for c,name,p in zip((c1,c2,c3),("Facial","Speech","Tabular"),(fp,sp,tp)):
        c.write(f"**{name}**"); c.bar_chart({s:float(v) for s,v in zip(STRESS_CLASSES,p)})
    st.write("### Fused probabilities")
    st.bar_chart({s:float(v) for s,v in zip(STRESS_CLASSES,fused)})
    st.write("### Explainability")
    contrib=modality_contributions(fp,sp,tp); st.write({k:f"{v:.1%}" if k!="target_class" else v for k,v in contrib.items()})
    st.caption("Modality contribution is a transparent probability-based attribution, not a clinical causal explanation.")
    if TABULAR_REGRESSOR_PATH.exists():
        reg=joblib.load(TABULAR_REGRESSOR_PATH); scores=reg.predict(np.asarray(vals).reshape(1,-1))[0]
        st.write("### Symptom severity estimates (tabular regression head)")
        st.bar_chart({"Depression":float(scores[0]),"Anxiety":float(scores[1]),"Stress":float(scores[2])})
        st.caption("The supplied dataset does not provide participant-level links between the image/audio files and the 4,000 tabular rows, so these regression estimates are kept as a separate tabular head rather than falsely claiming a trained aligned multimodal regressor.")
