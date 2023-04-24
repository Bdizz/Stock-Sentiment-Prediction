import pandas as pd
import pickle
import os


def load_data(train_file):
    """
    Loads data from CSV file and returns the data as a pandas dataframe
    """
    data = pd.read_csv(train_file)
    return data


def load_model():
    with open(os.path.join(os.path.dirname(__file__), 'models/model.pth')) as f:
        model = pickle.load(f)

    with open(os.path.join(os.path.dirname(__file__), 'models/vocab.pkl')) as f:
        vocab = pickle.load(f)

    return model, vocab


def save_model(model, vocab, model_file, vocab_file):
    """
     Saves the trained model to a file
    """

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)



