import pandas as pd
import pickle
import os


def load_data():
    """
    Loads data from CSV file and returns the data as a pandas dataframe
    """
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train.csv'))
    return data


def save_model(model, vocab, model_file, vocab_file):
    """
     Saves the trained model to a file
    """

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"Model saved to {model_file}")
    print(f"Vocabulary saved to {vocab_file}")

