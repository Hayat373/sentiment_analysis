import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|@\w+|#[^\s]+|[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove stopwords (optional for Transformers, useful for NLTK)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_data('../data/reviews.csv')
    print(df[['text', 'cleaned_text']].head())