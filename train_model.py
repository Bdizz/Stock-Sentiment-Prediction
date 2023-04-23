import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

from config import TRAIN_FILE, MAX_FEATURES, NGRAM_RANGE, ALPHA, RANDOM_STATE
from utils import load_data, save_model

