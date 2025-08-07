from transformers import pipeline

def load_sentiment_model():
    # Load pre-trained sentiment analysis model
    model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return model

def predict_sentiment(model, text):
    # Predict sentiment for a single text
    result = model(text)
    return result[0]['label'], result[0]['score']

if __name__ == "__main__":
    model = load_sentiment_model()
    sample_text = "This product is amazing and works perfectly!"
    label, score = predict_sentiment(model, sample_text)
    print(f"Text: {sample_text}")
    print(f"Sentiment: {label}, Confidence: {score:.2f}")