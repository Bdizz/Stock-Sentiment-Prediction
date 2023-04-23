from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import TRAIN_FILE, MAX_FEATURES, NGRAM_RANGE, ALPHA, RANDOM_STATE, MODEL_FILE, VOCAB_FILE
from utils import load_data,

# load the trained model and vocab
model, vocab = load_data(MODEL_FILE, VOCAB_FILE)

def predict_sentiment(text):
    # preprocess the text
    text = preprocess_text(text)

    # convert text to bag of words representation using vocab
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.transform([text])

    # use the trained model to predict the sentiment
    sentiment = model.predict(x)[0]

    return sentiment



