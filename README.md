# 📩 SMS Spam Classifier

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP), TF-IDF feature extraction, and a Multinomial Naive Bayes classifier.

The application provides real-time predictions, spam probability, adjustable classification sensitivity, and insights into the words influencing the prediction.

## 🌐 Live Demo

🚀 **[Try SMS Spam Classifier](https://smsspamclassifier-q8inyensj3iqge7yjgbvto.streamlit.app/)**

Explore real-time SMS spam detection, view prediction probabilities, adjust classification sensitivity, and test example messages directly in your browser.

## ✨ Features

- **Real-Time Classification:** Instantly predicts whether an SMS message is Spam or Not Spam.
- **Spam Probability:** Displays the probability of a message being spam along with a confidence score.
- **Adjustable Sensitivity:** Customize the spam detection threshold using a slider (0.15–0.75).
- **Text Analysis:** View preprocessed text and model prediction probabilities.
- **Example Messages:** Test the application using built-in spam and ham examples.
- **Spam Indicators:** Highlights words that contribute to the spam prediction.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Feature Extraction | TF-IDF Vectorization |
| Classification Algorithm | Multinomial Naive Bayes |
| NLP | NLTK |
| Data Processing | NumPy, Pandas |

## ⚙️ How It Works

The application follows a four-step machine learning pipeline.

### 1. Text Preprocessing

- Convert text to lowercase.
- Tokenize SMS messages.
- Remove stopwords and punctuation.
- Apply Porter Stemming to reduce words to their root forms.

### 2. Feature Extraction

- Use TF-IDF (Term Frequency–Inverse Document Frequency) to convert preprocessed text into numerical feature vectors.
- Transform text into a format suitable for machine learning.

### 3. Model Prediction

- Use a trained Multinomial Naive Bayes classifier.
- Calculate the probability of a message belonging to the spam class.

### 4. Classification

- Compare the predicted spam probability against the selected threshold.
- Classify the message as Spam or Not Spam.
- The default threshold is 0.25 and can be adjusted between 0.15 and 0.75.

## 📊 Model Performance

The model was trained using the **SMS Spam Collection Dataset**, containing 5,574 SMS messages.

### Dataset Distribution

| Dataset Category | Number of Messages |
|---|---:|
| Spam | 747 |
| Ham (Not Spam) | 4,827 |
| Total | 5,574 |

### Evaluation Metrics

| Metric (Spam Class) | Score |
|---|---:|
| Accuracy | 98.2% |
| Precision | 96.3% |
| F1-Score | 91.7% |

Since the dataset is imbalanced, precision and F1-score are important evaluation metrics rather than relying only on accuracy.

## ⚠️ Known Limitations

- The model may miss certain prize-style scams or spam messages with unusual wording.
- Lowering the spam threshold can help detect more spam messages but may increase false positives.
- The model is trained on a relatively small and older English SMS dataset.
- It may not accurately identify modern spam patterns or messages in other languages.

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/ksaisravan/sms_spam_classifier.git
cd sms_spam_classifier
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

**Note:** The trained model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) are included in the repository. No additional model download is required.

## 📁 Project Structure

```text
sms_spam_classifier/
│
├── app.py                # Streamlit application
├── model.pkl             # Trained Naive Bayes model
├── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── requirements.txt      # Project dependencies
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Model files not found | Ensure `model.pkl` and `vectorizer.pkl` are in the project root directory. |
| NLTK LookupError | Run the NLTK resource download command mentioned above. |
| Spam messages are being missed | Try lowering the spam threshold. |
| Too many false positives | Try increasing the spam threshold. |

## 🔮 Future Improvements

- Experiment with Logistic Regression and other classification algorithms.
- Implement cross-validation and a detailed classification report.
- Explore techniques to handle class imbalance.
- Improve spam detection using larger and more recent datasets.
- Further improve the application and model performance.

## 👨‍💻 Author

**Killamsetty Sai Sravan**

- GitHub: [ksaisravan](https://github.com/ksaisravan)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SMS Spam Collection Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Streamlit](https://streamlit.io/)
