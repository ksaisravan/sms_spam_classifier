import os
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")

# Download required NLTK data (punkt_tab is needed by newer NLTK versions)
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

ps = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


# IMPORTANT: this must exactly match the preprocessing used during training
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
        return None, None
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def spam_indicators(vectorizer, model, vector_input, top_n=8):
    """Words in the message that the model associates more with spam than ham."""
    if not hasattr(model, "feature_log_prob_"):
        return []
    names = vectorizer.get_feature_names_out()
    scored = []
    for idx in vector_input.nonzero()[1]:
        score = model.feature_log_prob_[1, idx] - model.feature_log_prob_[0, idx]
        if score > 0:
            scored.append((names[idx], score))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]


vectorizer, model = load_artifacts()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Spam threshold",
        min_value=0.15,
        max_value=0.75,
        value=0.25,
        step=0.05,
        help="A message is flagged as spam if its spam probability is above this value.",
    )
    st.caption("Lower = catches more spam, but more false alarms.")
    st.caption(
        "Naive Bayes probabilities are not perfectly calibrated, "
        "so treat the percentages as a rough score."
    )

    st.markdown("---")
    st.markdown("## 🚨 Common spam signals")
    st.markdown(
        """
- Financial offers (loan, cash)
- Urgency (approved, verify, claim)
- Suspicious links
- "Opt out" replies
- Prizes and winnings
"""
    )

# ---------- Main page ----------
st.title("📩 SMS Spam Classifier")
st.write("Paste a text message below and check whether it's **Spam** or **Not Spam**.")

if model is None or vectorizer is None:
    st.error(
        "Model files not found. Make sure 'vectorizer.pkl' and 'model.pkl' "
        "are in the project folder."
    )
    st.stop()

if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

st.markdown("### Try these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📧 Normal"):
        st.session_state.input_sms = "Hey, are we still meeting for lunch tomorrow?"
with col2:
    if st.button("📱 Loan Spam"):
        st.session_state.input_sms = (
            "a loan for $950 is approved for you if you receive this sms. "
            "1 min verification & cash in 1 hr at www.example.co.uk to opt out reply stop"
        )
with col3:
    if st.button("🎯 Prize Spam"):
        st.session_state.input_sms = (
            "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize now!"
        )

input_sms = st.text_area("Enter the message", key="input_sms", height=150)

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        proba = model.predict_proba(vector_input)[0]

        is_spam = proba[1] > threshold
        confidence = (proba[1] if is_spam else proba[0]) * 100

        if is_spam:
            st.error(f"🚨 **SPAM DETECTED!** (confidence: {confidence:.1f}%)")
            with st.expander("🔍 Why was this flagged as spam?"):
                indicators = spam_indicators(vectorizer, model, vector_input)
                if indicators:
                    st.write("**Words that pushed the score toward spam:**")
                    for word, score in indicators:
                        st.write(f"- `{word}` (spam lean: {score:.2f})")
                else:
                    st.write("No single word stood out.")
        else:
            st.success(f"✅ **Not Spam** (confidence: {confidence:.1f}%)")

        with st.expander("📊 Model Details"):
            st.write(f"**Spam probability:** {proba[1] * 100:.1f}%")
            st.write(f"**Not Spam probability:** {proba[0] * 100:.1f}%")
            st.write(f"**Threshold:** {threshold * 100:.0f}%")
            st.write(f"**Decision:** {'SPAM' if is_spam else 'NOT SPAM'}")
            st.progress(float(proba[1]))
            st.caption("Bar shows the spam probability")
            st.write("**Preprocessed text:**")
            st.code(transformed_sms or "(empty after preprocessing)")
