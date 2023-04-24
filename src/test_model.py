import pandas as pd
import pickle

from config import TEST_FILE, MODEL_FILE, VOCAB_FILE


def test_model():

    # load test data
    test_df = pd.read_csv(TEST_FILE, names=['description', 'label'], header=None)

    # load the trained model and vocab
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)

    # predict labels for test data
    predicted_labels = model.predict(test_df['description'])

    # make a dataframe with predicted labels
    predicted_df = pd.DataFrame({
        'description': test_df['description'],
        'predicted_label': predicted_labels
    })

    # merge predicted with original test data
    results_df = pd.merge(test_df, predicted_df, on='description')

    return results_df


