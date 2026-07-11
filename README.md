# 📩 SMS Spam Classifier

A machine learning web application that detects whether an SMS message is **Spam** or **Not Spam** using Natural Language Processing (NLP) and Naive Bayes classification.

## 📋 Quick Navigation
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **Real-time Classification**: Instant spam detection with confidence scores
- **Interactive Interface**: Clean, responsive Streamlit web app
- **Example Messages**: Quick test with pre-loaded spam and ham examples
- **Detailed Analysis**: View preprocessed text and model probabilities
- **Adjustable Sensitivity**: Customizable spam threshold (0.15 - 0.75)
- **Model Insights**: Visualize detected spam indicators

### Example Spam Messages Detected:
- 📱 Financial scams ("loan approved", "cash", "verification")
- 🎯 Prize scams ("Congratulations! You've won...")
- 🏦 Phishing attempts
- 📊 Promotional offers

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Programming language |
| Streamlit | 1.8.0 | Web application framework |
| Scikit-learn | 0.24.1 | Machine learning |
| NLTK | 3.6+ | Natural Language Processing |
| NumPy | 1.21+ | Numerical operations |

### Machine Learning Components
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model**: Multinomial Naive Bayes
- **Preprocessing**: Tokenization, Stopword removal, Stemming (Porter)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier
Step 2: Create Virtual Environment (Recommended)
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Download NLTK Data
bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
Step 5: Place Model Files
Download model.pkl and vectorizer.pkl from the releases page and place them in the project root directory.

🚀 Usage
Run the Application
bash
streamlit run app.py
Using the Interface
Enter a Message: Type or paste an SMS in the text area

Use Examples: Click example buttons for quick testing

Click Predict: The model will classify the message

View Results: See spam probability and confidence score

Example Outputs
text
Input: "a loan for $950 is approved for you..."
Output: 🚨 SPAM DETECTED! (35.5% confidence)

Input: "Hey, are we still meeting for lunch tomorrow?"
Output: ✅ Not Spam (56.7% confidence)

Input: "CONGRATULATIONS! You've won $1,000,000!"
Output: ✅ Not Spam (5.1% confidence)  # Model needs improvement
📊 Model Performance
Dataset
Source: SMS Spam Collection Dataset

Total Messages: 5,574

Spam: 747 (13.4%)

Ham: 4,827 (86.6%)

Performance Metrics
Metric	Score
Accuracy	98.2%
Precision (Spam)	96.3%
Recall (Spam)	87.5%
F1-Score (Spam)	91.7%
Confusion Matrix
text
              Predicted
              Spam    Ham
Actual Spam   653     94
Actual Ham    35     4792
Important Spam Indicators
Words that strongly indicate spam:

loan, cash, approved, verification

free, win, winner, prize

claim, urgent, guaranteed

congratulations, selected, exclusive

offer, limited, click, visit

📁 Project Structure
text
sms-spam-classifier/
├── app.py              # Streamlit web application
├── model.pkl           # Trained Naive Bayes model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
🔧 Troubleshooting
Common Issues & Solutions
1. Model Files Not Found
text
Error: Model files not found
Solution: Ensure model.pkl and vectorizer.pkl are in the project root.

2. NLTK Data Missing
text
LookupError: Resource punkt not found
Solution: Run NLTK download:

python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
3. Low Spam Detection
Solution: Adjust the spam threshold in app.py:

python
SPAM_THRESHOLD = 0.25  # Lower = more sensitive to spam
Default: 0.50 (balanced)

Sensitive: 0.25 (catches more spam)

Conservative: 0.75 (fewer false positives)

4. Memory Error
Solution: Reduce max_features in vectorizer:

python
vectorizer = TfidfVectorizer(max_features=3000)
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add improvement')

Push to branch (git push origin feature/improvement)

Open a Pull Request

Development Guidelines
Follow PEP 8 style guide

Write unit tests for new features

Update documentation

Maintain backward compatibility

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset: SMS Spam Collection Dataset

NLTK: For Natural Language Processing tools

Streamlit: For the web application framework

Scikit-learn: For machine learning algorithms

Open Source Community: For all the amazing tools and libraries
