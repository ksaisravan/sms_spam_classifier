 SMS Spam Classifier
A machine learning web app that classifies an SMS message as Spam or Not Spam using NLP preprocessing, TF-IDF features and a Multinomial Naive Bayes model.
✨ Features
- Real-time classification with a spam probability and confidence score
- Adjustable sensitivity: spam threshold slider (0.15 to 0.75)
- Detailed analysis: view the preprocessed text and model probabilities
- Example messages: one-click spam and ham samples for quick testing
- Spam indicators: shows the words that pushed a message toward spam
🛠️ Tech Stack
AreaToolsLanguagePythonWeb appStreamlitMLScikit-learn (TF-IDF, Multinomial Naive Bayes)NLPNLTK (tokenization, stopword removal, Porter stemming)DataNumPy, Pandas
⚙️ How It Works
1. Preprocess: lowercase, tokenize, remove stopwords and punctuation, apply Porter stemming.
2. Vectorize: convert the cleaned text to TF-IDF features.
3. Classify: a Multinomial Naive Bayes model returns the spam probability.
4. Decide: the message is flagged as spam if the probability is above the threshold (default 0.25, adjustable from 0.15 to 0.75 in the sidebar).
📊 Model Performance
Trained on the SMS Spam Collection Dataset: 5,574 messages (747 spam, 4,827 ham).
Metric (spam class)ScoreAccuracy98.2%Precision96.3%F1-score91.7%
The dataset is imbalanced (about 13% spam), so precision and F1 matter more than accuracy alone.
Known Limitations
- The model can miss prize-style scams that use unusual wording, so lowering the threshold in the sidebar catches more spam at the cost of more false positives.
- It is trained on a small, older English dataset and may not reflect modern spam.
🚀 Getting Started
1. #Clone the repository

git clone https://github.com/ksaisravan/sms_spam_classifier.gitcd sms_spam_classifier# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows# source venv/bin/activate   # Mac/Linux# 3. Install dependencies
pip install -r requirements.txt# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"# 5. Run the app
streamlit run app.py
The trained files model.pkl and vectorizer.pkl are included in the repository, so no extra download is needed.
📁 Project Structure
sms_spam_classifier/
├── app.py              # Streamlit web app
├── model.pkl           # Trained Multinomial Naive Bayes model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
🔧 Troubleshooting
ProblemFixModel files not foundMake sure model.pkl and vectorizer.pkl are in the project rootLookupError: Resource punkt not foundRun the NLTK download command from step 4Spam is being missedLower the spam threshold in the app sidebarToo many false alarmsRaise the spam threshold
🔮 Future Improvements
- Try Logistic Regression or an ensemble and compare results
- Add cross-validation and a full classification report
- Handle class imbalance (for example, class weights or resampling)
- Deploy on Streamlit Community Cloud
👤 Author
Killamsetty Sai Sravan GitHub · LinkedIn
📄 License
Licensed under the MIT License. See LICENSE for details.
🙏 Acknowledgments
- SMS Spam Collection Dataset (UCI Machine Learning Repository)
- NLTK, Scikit-learn and Streamlit


is this readme is enoygh
