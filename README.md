Spam Email Classifier

A machine learning project to detect spam emails using Python, Scikit-learn, and Flask. The project trains a model on a Kaggle dataset of emails and provides a simple web interface to check if an email is spam or ham.

Features

Preprocesses email text (lowercase, remove punctuation, stopwords, numbers, HTML)

Converts text to TF-IDF features

Trains Naive Bayes & Logistic Regression models

Compares models using precision, recall, and F1-score

Web interface using Flask for real-time email classification

Dataset

This project uses the Kaggle email dataset:

Columns: text (email content), spam (1 = spam, 0 = ham)

You can download it from: Kaggle Spam Emails

Folder Structure
spam-email-classifier/
│
├── data/
│   └── emails.csv             # Kaggle dataset
├── notebooks/
│   └── EDA.ipynb              # Optional: explore dataset
├── src/
│   ├── __init__.py            # empty file to make src a module
│   ├── preprocess.py          # text cleaning functions
│   ├── train.py               # train ML models
│   └── predict.py             # predict new emails
├── templates/
│   └── index.html             # Flask web interface
├── app.py                     # Flask app
├── model.pkl                  # trained Naive Bayes model
├── vectorizer.pkl             # TF-IDF vectorizer
├── requirements.txt           # project dependencies
└── README.md                  # this file

Installation

Clone the repository:

git clone <your-repo-url>
cd spam-email-classifier


Install dependencies:

pip install -r requirements.txt

Usage
1. Train the model
python src/train.py


This will train the Naive Bayes and Logistic Regression models

Saves the trained model as model.pkl and vectorizer as vectorizer.pkl in the project root

2. Test predictions
python src/predict.py


Test the model with a sample email

3. Run the Flask web app
python app.py


Open your browser: http://127.0.0.1:5000/

Paste an email and click Check to see if it’s Spam or Ham

How it works

Preprocessing: Removes stopwords, punctuation, numbers, HTML, and performs stemming

Vectorization: Converts cleaned text to numerical features using TF-IDF

Modeling: Trains Naive Bayes and Logistic Regression models

Prediction: Web app takes input email, preprocesses, vectorizes, and predicts using the saved model

Future Improvements

Detect promotional emails separately

Add deep learning-based classifier for higher accuracy

Deploy online for public use

License

MIT License