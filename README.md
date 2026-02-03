📧 Spam Email Classifier

A machine learning project to detect spam emails using Python, Scikit-learn, and Flask. This project automatically classifies emails as Spam or Ham to help users manage their inbox efficiently and securely.

🚀 Project Motivation

Emails are an essential communication tool, but spam and unsolicited emails create several problems:

Waste time and reduce productivity

Increase risk of phishing and scams

Overload inboxes, making it hard to identify important emails

Goal: Build a solution that automatically identifies spam emails, helping users focus on important messages while minimizing risks.

🎯 Problem Statement

Manually checking hundreds of emails every day is inefficient and error-prone. Traditional filters might not detect all spam types.

Solution:

Train a machine learning model to classify emails accurately

Provide a simple web interface for real-time predictions

📂 Dataset

Source: Kaggle Spam Emails

Columns:

text → full email content

spam → 1 = spam, 0 = ham

🛠 Methods & Implementation
1. Preprocessing

Lowercasing text

Removing punctuation, numbers, and HTML tags

Removing stopwords

Stemming words to their root form

Reason: Reduces noise and keeps only meaningful words for classification.

2. Feature Extraction

Method: TF-IDF Vectorization

Converts text to numerical features

Highlights important words that are frequent in one email but rare in all emails

Reason: Captures patterns that distinguish spam from ham without overemphasizing common words.

3. Machine Learning Models

Naive Bayes (MultinomialNB)

Fast, interpretable, excellent for short spam messages

Logistic Regression

Models probability of spam

Handles large feature sets and subtle patterns

Effectiveness: Both models are widely used in text classification and show high accuracy for spam detection.

4. Web Interface

Built using Flask

Users can paste email content and instantly get a Spam/Ham prediction

📈 Project Workflow

Load Dataset → Read emails from CSV

Preprocess Text → Clean and stem words

Vectorize Text → Convert to TF-IDF features

Train Models → Naive Bayes & Logistic Regression

Evaluate → Precision, Recall, F1-score

Save Models → model.pkl & vectorizer.pkl

Web App → Real-time email classification using Flask

💡 Key Outcomes / Delivery Report

Why we built it: Automate spam detection, improve inbox management, and reduce security risks.

How it solves the problem: Uses ML to identify spam patterns and classify emails automatically.

Methods used: Preprocessing, TF-IDF vectorization, Naive Bayes, Logistic Regression, Flask.

Why these methods: Proven efficiency in text classification; lightweight and interpretable.

Effectiveness: Can detect classic spam emails with high accuracy; professional or legitimate emails are correctly classified as Ham.

Future improvements:

Detect promotional emails separately

Use deep learning for higher accuracy

Multi-class classification (ham, spam, promotion, phishing)

📂 Folder Structure
spam-email-classifier/
│
├── data/
│   └── emails.csv            # Kaggle dataset
├── src/
│   ├── preprocess.py         # text cleaning functions
│   ├── train.py              # train ML models
│   └── predict.py            # predict new emails
├── templates/
│   └── index.html            # Flask web interface
├── app.py                    # Flask app
├── model.pkl                 # trained Naive Bayes model
├── vectorizer.pkl            # TF-IDF vectorizer
├── requirements.txt          # project dependencies
└── README.md                 # this file

⚡ Installation

Clone repository:

git clone https://github.com/ummehabiba-m/Spam-Email-Classifier.git
cd Spam-Email-Classifier


Install dependencies:

pip install -r requirements.txt

▶️ Usage
Train the Model
python src/train.py

Test Prediction
python src/predict.py

Run Web App
python app.py


Open browser: http://127.0.0.1:5000/

Paste an email → click Check → see Spam/Ham

📊 Evaluation Metrics

Precision, Recall, F1-score

Both models achieve high accuracy on the Kaggle dataset

Naive Bayes is faster, Logistic Regression handles complex patterns

📌 License

MIT License
