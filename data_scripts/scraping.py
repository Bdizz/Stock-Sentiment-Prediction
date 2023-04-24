import nltk
import requests
import re
import os
import csv
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

url = "https://finance.yahoo.com/rss/headline?s=LUMN"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.content, features="xml")

# finds all news article items in rss feed
items = soup.find_all('item')

# define a list of stop words
stop_words = set(stopwords.words('english'))



# initialize empty lists for title tokens, desc tokens, and categories
title_tokens_list = []
desc_tokens_list = []
categories = []

# loops through each item and extracts relevant data
def scrape():
    for item in items:
        title = item.find('title').text
        link = item.find('link').text
        description = item.find('description').text
        pub_date = item.find('pubDate').text

        # remove HTML tags
        title = re.sub('<.*?>', '', title)
        description = re.sub('<.*?>', '', description)

        # remove punctuation and special characters
        title = re.sub(r'[^\w\s]', '', title)
        description = re.sub(r'[^\w\s]', '', description)

        # tokenize the text
        title_tokens = word_tokenize((title.lower()))
        desc_tokens = word_tokenize(description.lower())

        # remove stop words
        title_tokens = [word for word in title_tokens if word not in stop_words]
        desc_tokens = [word for word in desc_tokens if word not in stop_words]

        sentiment = TextBlob(description).sentiment.polarity

        # assign category based on keywords
        if sentiment > 0:
            sentiment_label = "positive"
        elif sentiment < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # write the data to the CSV
        with open(os.path.join(os.path.dirname(__file__), '../data/labeled_data.csv'), "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([title, description, sentiment_label])

        # add the title tokens, desc tokens, and category to the respective lists
        title_tokens_list.append(title_tokens)
        desc_tokens_list.append(desc_tokens)
        categories.append(sentiment_label)

        print("Title Tokens:", title_tokens)
        print("Description tokens:", desc_tokens)
        print("Category:", sentiment_label)
        print("\n")

    # return a list of tuples containing title tokens, desc tokens, and category
    return [(title_tokens, desc_tokens, category) for title_tokens, desc_tokens, category in zip(title_tokens_list, desc_tokens_list, categories)]



