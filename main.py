from src.preprocess import load_data, clean_text
from src.model import load_sentiment_model, predict_sentiment

def analyze_dataset(file_path):
    # Load and preprocess data
    df = load_data(file_path)
    model = load_sentiment_model()
    # Predict sentiment for each text
    df['sentiment'], df['confidence'] = zip(*df['cleaned_text'].apply(lambda x: predict_sentiment(model, x)))
    return df

def analyze_single_text(text):
    model = load_sentiment_model()
    cleaned_text = clean_text(text)
    label, score = predict_sentiment(model, cleaned_text)
    return label, score

if __name__ == "__main__":
    # Analyze dataset
    df = analyze_dataset('data/reviews.csv')
    print(df[['text', 'sentiment', 'confidence']].head())
    
    # Analyze single text input
    user_input = input("Enter a review or tweet: ")
    label, score = analyze_single_text(user_input)
    print(f"Sentiment: {label}, Confidence: {score:.2f}")