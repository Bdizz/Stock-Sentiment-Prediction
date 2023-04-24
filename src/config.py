import os

# paths to data files
DATA_DIR = "/home/brandon/Desktop/Sentiment_Analysis/data"
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Hyperparms
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 1)
ALPHA = .1
RANDOM_STATE = 42

# Model Paths
MODEL_DIR = "/home/brandon/Desktop/Sentiment_Analysis/models"
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pth')
VOCAB_FILE = os.path.join(MODEL_DIR, 'vocab.pkl')

