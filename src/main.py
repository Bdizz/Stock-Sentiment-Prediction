from data_scripts.scraping import scrape
from src.test_model import test_model
from src.train_model import train_model
from data_scripts.prepare_data import data_prep


def main():

    # scrape the data
    scrape()

    train_model()

    results_df = test_model()

    # print the sentiment
    sentiment_counts = results_df['predicted_label'].value_counts()
    print("Sentiment about the stock:")
    if sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0):
        print("BUY")
    elif sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
        print("SELL")
    else:
        print("HOLD")


if __name__ == '__main__':
    main()
