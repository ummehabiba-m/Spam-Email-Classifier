import sys
import os
sys.path.append(os.path.dirname(__file__))

from preprocess import clean_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("D:\one drive\OneDrive\Desktop\spam\data\emails.csv")  # adjust path if needed

# 2️⃣ Use existing columns
df['label'] = df['spam']  # 1 = spam, 0 = ham

# 3️⃣ Clean text
df['clean_text'] = df['text'].apply(clean_text)

# 4️⃣ Features & Labels
X_text = df['clean_text']
y = df['label']

# 5️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# 6️⃣ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train Models
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 8️⃣ Evaluate
print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

joblib.dump(nb, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved successfully!")
