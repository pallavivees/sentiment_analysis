import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("output.csv")
df.columns = df.columns.str.strip()

# Preprocessing
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

df['cleaned'] = df['Review'].apply(clean_text)

# Features + labels
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['cleaned'])
y = df['Liked']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Save model + vectorizer
pickle.dump(model, open("nb_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("✅ Model and TF-IDF vectorizer saved successfully!")
