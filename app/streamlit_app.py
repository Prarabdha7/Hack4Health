import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (
    TABULAR_FEATURES,
    STRESS_CLASSES,
    TABULAR_REGRESSOR_PATH,
)
from fusion_model import predict


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hack4Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin: 1rem 0;
    }

    .result-title {
        font-size: 1rem;
        opacity: 0.7;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Hack4Health</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Multimodal Mental Health Assessment using Facial, Speech and Behavioral Signals"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About Hack4Health")

    st.write(
        """
        Hack4Health combines three complementary modalities
        to estimate a person's mental-health stress state:

        **Facial Analysis**
        - Facial expression recognition
        - Emotion → stress mapping

        **Speech Analysis**
        - Acoustic feature extraction
        - Stress classification

        **Behavioral & Physiological Analysis**
        - 18 tabular features
        - Machine-learning classification

        The three predictions are combined using
        the project's multimodal fusion model.
        """
    )

    st.divider()

    st.subheader("Fusion Weights")

    st.write("Facial: **45%**")
    st.write("Speech: **40%**")
    st.write("Tabular: **15%**")

    st.caption(
        "Weights are explicitly defined in the current fusion pipeline."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Provide Assessment Inputs</div>',
    unsafe_allow_html=True,
)

image_col, audio_col = st.columns(2)

with image_col:

    st.subheader("🖼 Facial Input")

    image_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"],
        key="facial_upload",
    )

    if image_file is not None:
        st.image(
            image_file,
            caption="Uploaded facial image",
            use_container_width=True,
        )


with audio_col:

    st.subheader("🎙 Speech Input")

    audio_file = st.file_uploader(
        "Upload a speech recording",
        type=["wav", "mp3", "ogg", "m4a"],
        key="audio_upload",
    )

    if audio_file is not None:
        st.audio(audio_file)


# ============================================================
# TABULAR INPUTS
# ============================================================

st.markdown(
    '<div class="section-title">2. Behavioral & Physiological Inputs</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Enter the observed values for the participant. "
    "The input ranges correspond to the supplied dataset."
)

col1, col2, col3 = st.columns(3)


with col1:

    sleep_quality = st.slider(
        "Sleep Quality",
        1.0,
        5.0,
        3.0,
        0.1,
    )

    social_engagement = st.slider(
        "Social Engagement",
        1.0,
        5.0,
        3.0,
        0.1,
    )

    daily_app_usage = st.slider(
        "Daily App Usage (min)",
        30,
        479,
        250,
    )

    typing_speed = st.slider(
        "Typing Speed (WPM)",
        20,
        89,
        53,
    )

    session_frequency = st.slider(
        "Session Frequency",
        1,
        19,
        10,
    )

    idle_time = st.slider(
        "Idle Time (min)",
        5,
        179,
        90,
    )


with col2:

    facial_variance = st.slider(
        "Facial Emotion Variance",
        0.10,
        1.00,
        0.55,
        0.01,
    )

    eye_blink_rate = st.slider(
        "Eye Blink Rate",
        10,
        34,
        22,
    )

    smile_intensity = st.slider(
        "Smile Intensity",
        0.0,
        0.999,
        0.50,
        0.001,
    )

    head_motion = st.slider(
        "Head Motion Index",
        0.0,
        1.0,
        0.50,
        0.01,
    )

    mfcc_mean = st.slider(
        "MFCC Mean",
        -49.991,
        49.997,
        0.0,
        0.1,
    )

    mfcc_variance = st.slider(
        "MFCC Variance",
        1.003,
        29.999,
        15.0,
        0.1,
    )


with col3:

    pitch_mean = st.slider(
        "Pitch Mean",
        80.0,
        299.97,
        190.0,
        0.1,
    )

    speech_rate = st.slider(
        "Speech Rate",
        2.0,
        14.0,
        8.0,
        0.1,
    )

    heart_rate = st.slider(
        "Heart Rate (BPM)",
        55,
        119,
        87,
    )

    hrv_index = st.slider(
        "HRV Index",
        10,
        99,
        55,
    )

    skin_temperature = st.slider(
        "Skin Temperature (°C)",
        32.0,
        37.0,
        34.5,
        0.1,
    )

    gsr_level = st.slider(
        "GSR Level",
        0.101,
        4.999,
        2.5,
        0.001,
    )


# ============================================================
# CREATE TABULAR VECTOR
# ============================================================

tabular_values = [
    sleep_quality,
    social_engagement,
    daily_app_usage,
    typing_speed,
    session_frequency,
    idle_time,
    facial_variance,
    eye_blink_rate,
    smile_intensity,
    head_motion,
    mfcc_mean,
    mfcc_variance,
    pitch_mean,
    speech_rate,
    heart_rate,
    hrv_index,
    skin_temperature,
    gsr_level,
]


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 ANALYZE MENTAL HEALTH",
    type="primary",
    use_container_width=True,
)


if analyze:

    if image_file is None:
        st.error("Please upload a facial image.")

    elif audio_file is None:
        st.error("Please upload a speech recording.")

    else:

        with st.spinner(
            "Running facial, speech, tabular and multimodal analysis..."
        ):

            try:

                # ------------------------------------------------
                # Save uploaded files temporarily
                # ------------------------------------------------

                temp_dir = ROOT / "outputs" / "gui_inputs"
                temp_dir.mkdir(parents=True, exist_ok=True)

                image_path = temp_dir / image_file.name
                audio_path = temp_dir / audio_file.name

                image_path.write_bytes(image_file.getbuffer())
                audio_path.write_bytes(audio_file.getbuffer())

                # ------------------------------------------------
                # Run actual multimodal model
                # ------------------------------------------------

                result = predict(
                    image_path,
                    audio_path,
                    tabular_values,
                )

                # ------------------------------------------------
                # Regression model
                # ------------------------------------------------

                regression_model = joblib.load(
                    TABULAR_REGRESSOR_PATH
                )

                regression_input = pd.DataFrame(
                    [tabular_values],
                    columns=TABULAR_FEATURES,
                )

                regression_prediction = regression_model.predict(
                    regression_input
                )[0]

                st.session_state["result"] = result
                st.session_state["regression"] = regression_prediction

                st.success("Analysis completed successfully.")

            except Exception as exc:

                st.error(
                    "The analysis could not be completed."
                )

                st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]
    regression = st.session_state["regression"]

    st.divider()

    st.markdown(
        '<div class="section-title">3. Multimodal Assessment</div>',
        unsafe_allow_html=True,
    )

    prediction = result["prediction"]
    fused = np.asarray(result["fused"])

    confidence = float(np.max(fused)) * 100

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">
                Overall Predicted Stress State
            </div>
            <div class="result-value">
                {prediction.replace("_", " ")}
            </div>
            <div>
                Fusion confidence: {confidence:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # MODALITY RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">Modality Predictions</div>',
        unsafe_allow_html=True,
    )

    facial_col, speech_col, tabular_col = st.columns(3)

    modality_data = {
        "Facial": result["facial"],
        "Speech": result["speech"],
        "Tabular": result["tabular"],
    }

    for column, (name, probabilities) in zip(
        [facial_col, speech_col, tabular_col],
        modality_data.items(),
    ):

        with column:

            st.subheader(name)

            best_idx = int(np.argmax(probabilities))

            st.metric(
                "Prediction",
                STRESS_CLASSES[best_idx].replace("_", " "),
            )

            chart_df = pd.DataFrame(
                {
                    "Stress State": [
                        c.replace("_", " ")
                        for c in STRESS_CLASSES
                    ],
                    "Probability": probabilities,
                }
            )

            st.bar_chart(
                chart_df.set_index("Stress State")
            )


    # ========================================================
    # FUSION DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">Fusion Probability Distribution</div>',
        unsafe_allow_html=True,
    )

    fusion_df = pd.DataFrame(
        {
            "Stress State": [
                c.replace("_", " ")
                for c in STRESS_CLASSES
            ],
            "Probability": fused,
        }
    )

    st.bar_chart(
        fusion_df.set_index("Stress State")
    )


    # ========================================================
    # REGRESSION
    # ========================================================

    st.markdown(
        '<div class="section-title">Mental Health Score Estimates</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)

    regression_names = [
        "Depression Score",
        "Anxiety Score",
        "Stress Score",
    ]

    for column, name, value in zip(
        [r1, r2, r3],
        regression_names,
        regression,
    ):

        with column:

            st.metric(
                name,
                f"{float(value):.2f}",
            )


    # ========================================================
    # RAW DATA
    # ========================================================

    with st.expander("View detailed model probabilities"):

        detailed_df = pd.DataFrame(
            {
                "Stress State": STRESS_CLASSES,
                "Facial": result["facial"],
                "Speech": result["speech"],
                "Tabular": result["tabular"],
                "Fused": result["fused"],
            }
        )

        st.dataframe(
            detailed_df.style.format(
                {
                    "Facial": "{:.4f}",
                    "Speech": "{:.4f}",
                    "Tabular": "{:.4f}",
                    "Fused": "{:.4f}",
                }
            ),
            use_container_width=True,
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Hack4Health is a research/demo system and is not a medical "
    "diagnostic tool. Predictions should not be used as a substitute "
    "for professional clinical assessment."
)