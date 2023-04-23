import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


url = "https://finance.yahoo.com/rss/headline?s=TRMB"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.content, features="xml")

# finds all news article items in rss feed
items = soup.find_all('item')

# define a list of stop words
stop_words = set(stopwords.words('english'))

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

        print("Title Tokens", title_tokens)
        print("Description tokens", desc_tokens)
        print("\n")

        return title_tokens, desc_tokens
