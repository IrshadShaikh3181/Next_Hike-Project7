import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

# ----------------------- App Setup -----------------------
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Disaster Tweet Classifier")
st.write("This app uses a deep learning model trained on BERT embeddings to predict whether a tweet is about a real disaster or not.")

# -------------------- Model Loading ----------------------
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        model = load_model('disaster_bert_model.h5')
        return embedder, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

embedder, model = load_models()

if embedder is None or model is None:
    st.stop()

# --------------------- Preprocessing ---------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s!?]', '', text)
    text = ' '.join(text.split())
    return text

# --------------------- Prediction ---------------------
def predict(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    prediction = model.predict(embedding)
    prob = float(prediction[0][0])
    return prob > 0.5, prob

# --------------------- UI Section ---------------------
st.subheader("Enter tweet text:")
user_input = st.text_area("Type a tweet or message to classify", height=100)

if st.button("Predict"):
    if user_input.strip():
        with st.spinner("Predicting..."):
            is_disaster, confidence = predict(user_input)

        st.subheader("Result:")
        if is_disaster:
            st.error(f"🚨 Disaster Tweet Detected (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Not a Disaster Tweet (Confidence: {confidence:.2%})")

        st.markdown("**Cleaned Text:**")
        st.code(clean_text(user_input))
    else:
        st.warning("Please enter some text first.")

# --------------------- Sidebar ---------------------
st.sidebar.header("📄 Info")
st.sidebar.markdown("""
This model was trained using:
- BERT sentence transformer (all-MiniLM-L6-v2)
- Deep learning model with:
    - Dense(128, relu) → Dropout → Dense(64, relu) → Dropout → Dense(1, sigmoid)
- Trained on labeled tweets (`target`: 0 = not disaster, 1 = disaster)
""")
