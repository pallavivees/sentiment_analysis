import os, pickle, pandas as pd, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import streamlit as st

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Load or train model
if os.path.exists("nb_model.pkl") and os.path.exists("tfidf.pkl"):
    model = pickle.load(open("nb_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
else:
    df = pd.read_csv("output.csv")
    df.columns = df.columns.str.strip()
    df['cleaned'] = df['Review'].apply(clean_text)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned'])
    y = df['Liked']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = BernoulliNB()
    model.fit(X_train, y_train)

    pickle.dump(model, open("nb_model.pkl", "wb"))
    pickle.dump(tfidf, open("tfidf.pkl", "wb"))

# Streamlit UI
st.title("Naïve Bayes Sentiment Classifier")

user_input = st.text_area("Enter a review:")
if st.button("Predict"):
    cleaned = clean_text(user_input)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]
    st.write("Sentiment:", "👍 Positive" if prediction==1 else "👎 Negative")