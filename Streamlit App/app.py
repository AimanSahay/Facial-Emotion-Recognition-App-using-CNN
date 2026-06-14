import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion AI Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* Hide Streamlit default menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.hero-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero-subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.2rem;
    margin-bottom: 40px;
}

.prediction-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
}

.emotion-text {
    font-size: 2rem;
    font-weight: bold;
    color: white;
}

.confidence-text {
    font-size: 1.2rem;
    color: #dbeafe;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:40px;
    padding-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("cnn_model_new_.h5")

model = load_emotion_model()

# -----------------------------
# LABELS
# -----------------------------
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

emotion_emojis = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img, target_size=(48, 48)):
    img = img.convert("L")
    img = img.resize(target_size)

    img_arr = img_to_array(img)
    img_arr = img_arr / 255.0

    img_arr = np.reshape(img_arr, (1, 48, 48, 1))

    return img_arr

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    """
    <div class='hero-title'>
        🤖 Emotion AI Detector
    </div>

    <div class='hero-subtitle'>
        Upload a face image and let the CNN model predict the emotion.
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if st.button("🔍 Analyze Emotion", use_container_width=True):

        with st.spinner("Analyzing emotion..."):

            processed = preprocess_image(image)

            preds = model.predict(processed, verbose=0)

            label = np.argmax(preds)

            confidence = float(np.max(preds) * 100)

            emotion = emotion_labels[label]

            emoji = emotion_emojis[emotion]

        # -----------------------------
        # IMAGE + RESULT
        # -----------------------------
        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                width=300
            )

        with col2:

            st.markdown("## Emotion Prediction")

            st.markdown(f"# {emoji}")

            st.markdown(
                f"""
                <h2 style='color:white'>
                    {emotion}
                </h2>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence / 100)

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

        # -----------------------------
        # PROBABILITY CHART
        # -----------------------------
        st.subheader("📊 Emotion Probability Distribution")

        prob_df = pd.DataFrame({
            "Emotion": list(emotion_labels.values()),
            "Probability": preds[0]
        })

        st.bar_chart(
            prob_df.set_index("Emotion")
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div class='footer'>
        Built using CNN • TensorFlow • Keras • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)