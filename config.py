import os

# paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Hyperparms
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 1)
ALPHA = .1
RANDOM_STATE = 42

# Model Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pth')
VOCAB_FILE = os.path.join(MODEL_DIR, 'vocab.pkl')

