import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
from src.model import load_sentiment_model, predict_sentiment
from src.preprocess import clean_text

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'))
model = load_sentiment_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        label, score = predict_sentiment(model, cleaned_text)
        return render_template('index.html', text=text, sentiment=label, confidence=f"{score:.2f}")
    return render_template('index.html', text='', sentiment='', confidence='')

if __name__ == '__main__':
    app.run(debug=True, port=5001)