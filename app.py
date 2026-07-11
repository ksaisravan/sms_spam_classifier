import pickle
import string
import os

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()


# IMPORTANT: This must EXACTLY match your training preprocessing
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
        st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the current directory.")
        return None, None

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


vectorizer, model = load_artifacts()

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

st.title("📩 SMS Spam Classifier")
st.write("Paste a text message below and check whether it's **Spam** or **Not Spam**.")

# --- FIX: Lower the threshold ---
SPAM_THRESHOLD = 0.25  # Default was 0.5, now lowered to 25%

# Example messages
st.markdown("### Try these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📧 Normal"):
        st.session_state.input_sms = "Hey, are we still meeting for lunch tomorrow?"
with col2:
    if st.button("📱 Loan Spam"):
        st.session_state.input_sms = "a loan for $950 is approved for you if you receive this sms. 1 min verification & cash in 1 hr at www.example.co.uk to opt out reply stop"
with col3:
    if st.button("🎯 Prize Spam"):
        st.session_state.input_sms = "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize now!"

if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

input_sms = st.text_area("Enter the message", value=st.session_state.input_sms, height=150)

if st.button("Predict"):
    if vectorizer is None or model is None:
        st.error("Cannot make predictions - model files are missing.")
    elif not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])

        # Get probabilities
        proba = model.predict_proba(vector_input)[0]

        # --- FIX: Use custom threshold ---
        if proba[1] > SPAM_THRESHOLD:
            result = 1
            confidence = proba[1] * 100
        else:
            result = 0
            confidence = proba[0] * 100

        # Display result
        if result == 1:
            st.error(f"🚨 **SPAM DETECTED!** (confidence: {confidence:.1f}%)")

            # Show why it was flagged
            with st.expander("🔍 Why was this flagged as spam?"):
                feature_names = vectorizer.get_feature_names_out()
                nonzero = vector_input.nonzero()
                st.write("**Spam indicators found:**")
                for idx in nonzero[1][:10]:
                    st.write(f"- '{feature_names[idx]}' (weight: {vector_input[0, idx]:.3f})")
        else:
            st.success(f"✅ **Not Spam** (confidence: {confidence:.1f}%)")

        # Show raw probabilities
        with st.expander("📊 Model Details"):
            st.write(f"**Spam probability:** {proba[1] * 100:.1f}%")
            st.write(f"**Not Spam probability:** {proba[0] * 100:.1f}%")
            st.write(f"**Threshold:** {SPAM_THRESHOLD * 100:.0f}%")
            st.write(f"**Decision:** {'SPAM' if result == 1 else 'NOT SPAM'}")

            # Visual progress bar
            st.progress(proba[1])
            st.caption("🔴 = Spam probability")

            # Show preprocessed text
            st.write("**Preprocessed text:**")
            st.code(transformed_sms or "(empty after preprocessing)")

# Sidebar with info
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown(f"**Spam Threshold:** {SPAM_THRESHOLD * 100:.0f}%")
    st.caption("Lower threshold = more sensitive to spam")

    st.markdown("---")
    st.markdown("## 📊 Current Model Performance")
    st.markdown("Based on your test results:")
    st.markdown(f"- Loan spam detected at **35.5%** ✅")
    st.markdown(f"- Prize spam at **5.1%** ⚠️ (needs retraining)")
    st.markdown(f"- Normal messages below threshold ✅")

    st.markdown("---")
    st.markdown("## 🚨 Spam Indicators")
    st.markdown("""
    - Financial offers (loan, cash)
    - Urgency (approved, verify)
    - Suspicious links
    - "Opt out" replies
    - Prizes/winning
    """)