import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import TRAIN_FILE, MAX_FEATURES, NGRAM_RANGE, ALPHA, RANDOM_STATE, MODEL_FILE, VOCAB_FILE


def train_model():

    # load data
    train_df = pd.read_csv(TRAIN_FILE, names=['description', 'label'], header=None)

    # set random state using numpy random
    np.random.seed(RANDOM_STATE)

    # define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
        ('clf', MultinomialNB(alpha=ALPHA)),
    ])

    # train the model
    pipeline.fit(train_df['description'], train_df['label'])

    # save the trained model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(pipeline, f)

    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(pipeline.named_steps['vect'].vocabulary_, f)



