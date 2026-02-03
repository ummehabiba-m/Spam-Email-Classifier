import sys
import os
sys.path.append(os.path.dirname(__file__))

from preprocess import clean_text
import joblib

# Load saved model and vectorizer
model = joblib.load("../model.pkl")
vectorizer = joblib.load("../vectorizer.pkl")

def predict_spam(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "Spam" if pred==1 else "Ham"

if __name__ == "__main__":
    email_text = "Congratulations! You won a free iPhone. Click here."
    print(predict_spam(email_text))
