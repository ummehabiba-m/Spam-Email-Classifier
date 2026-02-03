import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    """
    Clean email text:
    - Lowercase
    - Remove HTML tags
    - Remove numbers
    - Remove punctuation
    - Remove stopwords
    - Stemming
    """
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'\d+', '', text)    # remove numbers
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)
