# 📩 SMS Spam Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smsspamclassifier-q8inyensj3iqge7yjgbvto.streamlit.app/)

A machine learning web app that classifies an SMS message as **Spam** or **Not Spam** using NLP preprocessing, TF-IDF features, and a Multinomial Naive Bayes model.

---

## ✨ Features
- **Live Demo:** Try the app instantly at [smsspamclassifier.streamlit.app](https://smsspamclassifier-q8inyensj3iqge7yjgbvto.streamlit.app/)
- **Real-time classification:** Get predictions with spam probability and confidence scores
- **Adjustable sensitivity:** Custom spam threshold slider (0.15 to 0.75) in the sidebar
- **Detailed analysis:** View preprocessed text tokens and model probability breakdowns
- **One-click testing:** Built-in example messages for quick spam and ham testing
- **Spam indicators:** Highlights specific words that pushed a message toward a spam rating

---

## 🛠️ Tech Stack

| Area | Tools |
| :--- | :--- |
| **Language** | Python |
| **Web App** | Streamlit |
| **Machine Learning** | Scikit-learn (TF-IDF Vectorizer, Multinomial Naive Bayes) |
| **NLP** | NLTK (tokenization, stopword removal, Porter stemming) |
| **Data Processing** | NumPy, Pandas |

---

## ⚙️ How It Works
1. **Preprocess:** Lowercase conversion, tokenization, removal of stopwords and punctuation, followed by Porter stemming.
2. **Vectorize:** Convert the cleaned text into numerical TF-IDF feature vectors.
3. **Classify:** Pass feature vectors to the Multinomial Naive Bayes model to obtain probabilities.
4. **Decide:** Flag the message as spam if the predicted spam probability exceeds the configured threshold (default `0.25`).

---

## 📊 Model Performance

Trained on the **SMS Spam Collection Dataset**: 5,574 messages (747 spam, 4,827 ham).

| Metric (Spam Class) | Score |
| :--- | :--- |
| **Accuracy** | 98.2% |
| **Precision** | 96.3% |
| **F1-Score** | 91.7% |

*Note: Because the dataset is imbalanced (~13% spam), precision and F1-score are emphasized over raw accuracy.*

### Known Limitations
- May occasionally miss novel prize-style scams with non-traditional wording (lowering the sensitivity threshold catches more at the cost of potential false positives).
- Trained primarily on standard English text formats.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/ksaisravan/sms_spam_classifier.git](https://github.com/ksaisravan/sms_spam_classifier.git)
cd sms_spam_classifier
2. Set up virtual environment (Optional)Bashpython -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate
3. Install dependenciesBashpip install -r requirements.txt
4. Download NLTK dataBashpython -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
5. Run the web app locallyBashstreamlit run app.py
The pre-trained artifacts model.pkl and vectorizer.pkl are bundled in the repository, so no additional training step is required.📁 Project StructurePlaintextsms_spam_classifier/
├── app.py              # Streamlit web app
├── model.pkl           # Trained Multinomial Naive Bayes model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
🔧 TroubleshootingProblemFixModel files not foundEnsure model.pkl and vectorizer.pkl remain in the root directory.LookupError: Resource punkt not foundRe-run the NLTK download command from Step 4.Spam missedLower the spam threshold slider in the sidebar.Too many false alarmsIncrease the spam threshold slider in the sidebar.🔮 Future ImprovementsTest performance against Logistic Regression and Support Vector Classifier (SVC) models.Implement automated cross-validation pipelines.Explore advanced class re-balancing techniques (SMOTE / custom class weights).👤 AuthorKillamsetty Sai SravanGitHub📄 LicenseLicensed under the MIT License.🙏 AcknowledgmentsSMS Spam Collection Dataset (UCI Machine Learning Repository)NLTK, Scikit-learn, and Streamlit teams
