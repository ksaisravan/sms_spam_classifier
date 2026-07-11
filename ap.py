import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Your preprocessing function
ps = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


# Load model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Test your loan message
test_messages = [
    "a loan for $950 is approved for you if you receive this sms. 1 min verification & cash in 1 hr at www.example.co.uk to opt out reply stop",
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize now!",
    "Can you pick up some milk on your way home?"
]

for msg in test_messages:
    transformed = transform_text(msg)
    vectorized = vectorizer.transform([transformed])

    # Check which features were found
    feature_names = vectorizer.get_feature_names_out()
    nonzero = vectorized.nonzero()

    print(f"\nMessage: {msg[:50]}...")
    print(f"Transformed: {transformed[:100]}...")
    print(f"Features found: {len(nonzero[1])}")

    if len(nonzero[1]) > 0:
        print("Top features detected:")
        for idx in nonzero[1][:5]:
            print(f"  - '{feature_names[idx]}' (value: {vectorized[0, idx]:.3f})")

    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    print(f"Prediction: {'SPAM' if pred == 1 else 'NOT SPAM'}")
    print(f"Spam prob: {proba[1]:.3f}, Ham prob: {proba[0]:.3f}")