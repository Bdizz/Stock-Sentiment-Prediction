import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import TRAIN_FILE, MAX_FEATURES, NGRAM_RANGE, ALPHA, RANDOM_STATE, MODEL_FILE, VOCAB_FILE
from utils import load_data, save_model


def train_model():

    # load data
    train_df = load_data(TRAIN_FILE)

    # setting random state using numpy random
    np.random.seed(RANDOM_STATE)

    # define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
        ('clf', MultinomialNB(alpha=ALPHA)),
    ])

    # train the model
    pipeline.fit(train_df['description'], train_df['label'])

    # save the trained model
    model = pipeline
    vocab = pipeline.named_steps['vect'].vocabulary_
    save_model(model, vocab, MODEL_FILE, VOCAB_FILE)

