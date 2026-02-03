import joblib
from src.preprocess import clean_text
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    if request.method == "POST":
        email_text = request.form["message"]
        cleaned = clean_text(email_text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        result = "Spam" if pred==1 else "Ham"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
